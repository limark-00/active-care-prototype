"""Loopback-only frame inference; in-memory processing, no image/video persistence."""
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import secrets
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from pose_logic import PoseTracker

ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']
MAX_BYTES = 1_500_000
SESSION_TTL = 120


def decode_frame(data):
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != 'JPEG' or not (64 <= image.width <= 1280 and 64 <= image.height <= 960):
                raise ValueError('Expected JPEG, dimensions 64–1280 × 64–960')
            rgb = np.asarray(image.convert('RGB'))
            return np.ascontiguousarray(rgb[:, :, ::-1])
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, '无效 JPEG，或画面尺寸超过限制。') from exc


def create_app(detector_factory=None):
    @asynccontextmanager
    async def lifespan(app):
        if detector_factory is None:
            from detector import Detector
            factory = Detector
        else:
            factory = detector_factory
        app.state.detector = await run_in_threadpool(factory)
        yield
        app.state.sessions.clear()

    app = FastAPI(title='Active Care Local Vision', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.sessions = {}
    app.state.inference_lock = asyncio.Lock()
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type', 'X-Care-Client', 'X-Session-Id', 'X-Frame-Id'], max_age=600)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def local_browser_only(request, call_next):
        from starlette.responses import JSONResponse
        origin = request.headers.get('origin')
        if origin is not None and origin not in ORIGINS:
            return JSONResponse({'detail': '只允许本地网页访问视觉服务。'}, status_code=403)
        if request.method in ('POST', 'DELETE') and request.headers.get('x-care-client') != 'active-care-web':
            return JSONResponse({'detail': '缺少本地客户端标识。'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/health')
    def health():
        detector = app.state.detector
        return {'status': 'ready', 'model': detector.model_name, 'device': detector.device, 'schema_version': 1}

    @app.post('/sessions')
    def start_session():
        now = time.monotonic()
        sessions = app.state.sessions
        for key in list(sessions):
            if now - sessions[key]['at'] > SESSION_TTL:
                del sessions[key]
        if len(sessions) >= 4:
            raise HTTPException(429, '最多同时开启 4 个本地会话，请先停止其他页面。')
        token = secrets.token_urlsafe(24)
        sessions[token] = {'tracker': PoseTracker(), 'at': now, 'last_frame': -1}
        return {'session_id': token}

    @app.delete('/sessions')
    def stop_session(request: Request):
        app.state.sessions.pop(request.headers.get('x-session-id'), None)
        return {'stopped': True}

    @app.post('/frames')
    async def frame(request: Request):
        session = app.state.sessions.get(request.headers.get('x-session-id'))
        if session is None or time.monotonic() - session['at'] > SESSION_TTL:
            raise HTTPException(410, '会话已过期，请重新开启摄像头。')
        if request.headers.get('content-type') != 'image/jpeg':
            raise HTTPException(415, '只接受 image/jpeg。')
        try:
            frame_id = int(request.headers.get('x-frame-id', '-1'))
        except ValueError as exc:
            raise HTTPException(400, '无效帧编号。') from exc
        if not 0 <= frame_id <= 2**53 - 1 or frame_id <= session['last_frame']:
            raise HTTPException(409, '重复或乱序帧。')
        if app.state.inference_lock.locked():
            raise HTTPException(429, '识别忙碌，请丢弃旧帧后重试。')
        async with app.state.inference_lock:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise HTTPException(413, '单帧超过大小限制。')
            image = await run_in_threadpool(decode_frame, data)
            start = time.monotonic()
            try:
                persons = await run_in_threadpool(app.state.detector.predict, image)
            except Exception as exc:
                raise HTTPException(503, '模型推理失败，请检查本地服务终端并重启。') from exc
            elapsed = (time.monotonic() - start) * 1000
            height, width = image.shape[:2]
            persons = session['tracker'].update(persons, width, height, start)
            session['at'], session['last_frame'] = time.monotonic(), frame_id
            return {'schema_version': 1, 'source': 'camera', 'model': app.state.detector.model_name, 'device': app.state.detector.device, 'frame_id': frame_id, 'width': width, 'height': height, 'inference_ms': round(elapsed, 1), 'persons': persons}

    return app


app = create_app()
