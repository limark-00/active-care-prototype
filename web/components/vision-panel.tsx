'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  ScanLine,
  ShieldAlert,
  VideoOff,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import {
  ACTION_LABELS,
  cameraError,
  drawFrame,
  parseVisionFrame,
  VISION_URL,
} from '@/lib/vision';
import type { VisionFrame, VisionLog } from '@/lib/vision';
import type { CameraObservation, ZoneAction, ZoneState } from '@/lib/zones';
import ZoneEditor from '@/components/zone-editor';
import { startFramePump, VISION_TIMING } from '@/lib/vision-runtime';

type Phase = 'off' | 'connecting' | 'running' | 'error';
const CLIENT_HEADERS = { 'X-Care-Client': 'active-care-web' };

export default function VisionPanel({
  zoneState,
  onZoneAction,
  onObservation,
  pendingFalls,
  now,
}: {
  zoneState: ZoneState;
  onZoneAction: (action: ZoneAction) => void;
  onObservation: (observation: CameraObservation | null) => void;
  pendingFalls: number;
  now: () => number;
}) {
  const [phase, setPhase] = useState<Phase>('off');
  const [message, setMessage] = useState(
    '点击开启后，浏览器会请求摄像头权限。',
  );
  const [frame, setFrame] = useState<VisionFrame | null>(null);
  const [fps, setFps] = useState(0);
  const [skeleton, setSkeleton] = useState(true);
  const [flipped, setFlipped] = useState(true);
  const [view, setView] = useState<'live' | 'analysis'>('live');
  const [timing, setTiming] = useState<{
    encodingMs: number;
    transportMs: number;
    totalMs: number;
  } | null>(null);
  const [resultAge, setResultAge] = useState<number | null>(null);
  const [logs, setLogs] = useState<VisionLog[]>([]);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef<string | null>(null);
  const generation = useRef(0);
  const stopPump = useRef<(() => void) | null>(null);
  const controller = useRef<AbortController | null>(null);
  const skeletonRef = useRef(true);
  const flippedRef = useRef(true);
  const lastSnapshot = useRef<{
    source: HTMLCanvasElement;
    frame: VisionFrame;
  } | null>(null);
  const lastCaptureAt = useRef<number | null>(null);
  const codes = useRef(new Map<number, Set<string>>());
  const logSequence = useRef(0);
  const previousFrameTime = useRef<number | null>(null);

  const release = useCallback(() => {
    generation.current += 1;
    stopPump.current?.();
    stopPump.current = null;
    controller.current?.abort();
    controller.current = null;
    for (const track of stream.current?.getTracks() ?? []) {
      track.onended = null;
      track.onmute = null;
      track.stop();
    }
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    const context = canvas.current?.getContext('2d');
    if (canvas.current)
      context?.clearRect(0, 0, canvas.current.width, canvas.current.height);
    if (session.current) {
      void fetch(`${VISION_URL}/sessions`, {
        method: 'DELETE',
        headers: { ...CLIENT_HEADERS, 'X-Session-Id': session.current },
        keepalive: true,
      }).catch(() => undefined);
      session.current = null;
    }
    codes.current.clear();
    previousFrameTime.current = null;
    lastSnapshot.current = null;
    lastCaptureAt.current = null;
    onObservation(null);
  }, [onObservation]);
  const stop = useCallback(
    (reason: string, error = false) => {
      release();
      setPhase(error ? 'error' : 'off');
      setFrame(null);
      setFps(0);
      setTiming(null);
      setResultAge(null);
      setMessage(reason);
    },
    [release],
  );
  useEffect(() => {
    const onHidden = () => {
      if (document.hidden)
        stop('页面已进入后台，摄像头已停止。重新开启才能继续识别。');
    };
    const onPageHide = () => stop('页面已离开，摄像头已停止。');
    document.addEventListener('visibilitychange', onHidden);
    window.addEventListener('pagehide', onPageHide);
    return () => {
      document.removeEventListener('visibilitychange', onHidden);
      window.removeEventListener('pagehide', onPageHide);
      release();
    };
  }, [release, stop]);

  useEffect(() => {
    const timer = setInterval(() => {
      setResultAge(
        lastCaptureAt.current === null
          ? null
          : Math.round(performance.now() - lastCaptureAt.current),
      );
    }, 250);
    return () => clearInterval(timer);
  }, []);

  function redraw() {
    const snapshot = lastSnapshot.current;
    if (canvas.current && snapshot)
      drawFrame(
        canvas.current,
        snapshot.source,
        snapshot.frame,
        skeletonRef.current,
        flippedRef.current,
      );
  }

  async function request(path: string, options: RequestInit, timeout = 6000) {
    const abort = new AbortController();
    controller.current = abort;
    const timer = setTimeout(() => abort.abort(), timeout);
    try {
      const response = await fetch(`${VISION_URL}${path}`, {
        ...options,
        signal: abort.signal,
      });
      const body: unknown = await response.json();
      if (!response.ok) {
        const detail =
          typeof body === 'object' &&
          body !== null &&
          'detail' in body &&
          typeof body.detail === 'string'
            ? body.detail
            : `视觉服务返回 ${response.status}`;
        throw new Error(detail);
      }
      return body;
    } catch (error) {
      if (error instanceof TypeError)
        throw new Error(
          '无法连接本地视觉服务。请先启动 vision 服务（127.0.0.1:8001），并允许浏览器访问本地网络。',
        );
      throw error;
    } finally {
      clearTimeout(timer);
      if (controller.current === abort) controller.current = null;
    }
  }

  async function start() {
    release();
    const current = generation.current;
    setPhase('connecting');
    setFrame(null);
    setFps(0);
    setTiming(null);
    setResultAge(null);
    setMessage('正在检查本地模型…');
    try {
      if (
        !['localhost', '127.0.0.1'].includes(location.hostname) ||
        location.port !== '3000'
      )
        throw new Error(
          '请从 http://localhost:3000 打开此功能；不允许远程网页读取本机摄像头服务。',
        );
      if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext)
        throw new Error(
          '此浏览器环境无法调用摄像头。请使用本机浏览器打开 localhost:3000。',
        );
      const health = await request('/health', {}, 4000);
      if (current !== generation.current) return;
      if (
        typeof health !== 'object' ||
        health === null ||
        !('status' in health) ||
        health.status !== 'ready'
      )
        throw new Error('本地模型尚未准备就绪。');
      const data = await request('/sessions', {
        method: 'POST',
        headers: CLIENT_HEADERS,
      });
      if (current !== generation.current) return;
      if (
        typeof data !== 'object' ||
        data === null ||
        !('session_id' in data) ||
        typeof data.session_id !== 'string'
      )
        throw new Error('无法创建本地识别会话。');
      session.current = data.session_id;
      setMessage('请允许摄像头权限。画面仅送到本机，不录制。');
      const media = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      });
      if (current !== generation.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      for (const track of media.getVideoTracks()) {
        track.onended = () => stop('摄像头连接已结束，状态未知。', true);
        track.onmute = () =>
          stop('摄像头画面被暂停，状态未知，请重新开启。', true);
      }
      const element = video.current;
      if (!element) throw new Error('视频组件未准备好，请刷新页面。');
      element.srcObject = media;
      await element.play();
      if (current !== generation.current) return;
      setMessage('正在读取第一帧，等待模型返回。');
      const observationSession = `CAM-${crypto.randomUUID().slice(0, 8)}`;
      const capture = document.createElement('canvas');
      const completedSource = document.createElement('canvas');
      let frameId = 0;
      const analyze = async () => {
        if (current !== generation.current) return;
        const started = performance.now();
        if (!element.videoWidth || !element.videoHeight) return;
        const scale = Math.min(
          640 / element.videoWidth,
          480 / element.videoHeight,
          1,
        );
        const width = Math.round(element.videoWidth * scale);
        const height = Math.round(element.videoHeight * scale);
        if (capture.width !== width) capture.width = width;
        if (capture.height !== height) capture.height = height;
        const ctx = capture.getContext('2d');
        if (!ctx) throw new Error('浏览器无法读取视频画面。');
        const capturedAt = now();
        ctx.drawImage(element, 0, 0, capture.width, capture.height);
        const blob = await new Promise<Blob>((resolve, reject) =>
          capture.toBlob(
            (value) =>
              value ? resolve(value) : reject(new Error('视频帧编码失败。')),
            'image/jpeg',
            0.8,
          ),
        );
        if (current !== generation.current || !session.current) return;
        const encoded = performance.now();
        const id = ++frameId;
        const result = parseVisionFrame(
          await request('/frames', {
            method: 'POST',
            headers: {
              ...CLIENT_HEADERS,
              'Content-Type': 'image/jpeg',
              'X-Session-Id': session.current,
              'X-Frame-Id': String(id),
            },
            body: blob,
          }),
          id,
        );
        if (current !== generation.current) return;
        if (result.width !== capture.width || result.height !== capture.height)
          throw new Error('识别画面尺寸不匹配，已停止。');
        const received = performance.now();
        if (completedSource.width !== width) completedSource.width = width;
        if (completedSource.height !== height) completedSource.height = height;
        const completedContext = completedSource.getContext('2d');
        if (!completedContext) throw new Error('浏览器无法绘制识别画面。');
        completedContext.drawImage(capture, 0, 0);
        lastSnapshot.current = { source: completedSource, frame: result };
        if (canvas.current)
          drawFrame(
            canvas.current,
            completedSource,
            result,
            skeletonRef.current,
            flippedRef.current,
          );
        lastCaptureAt.current = started;
        setResultAge(Math.round(performance.now() - started));
        setTiming({
          encodingMs: encoded - started,
          transportMs: received - encoded,
          totalMs: performance.now() - started,
        });
        setFps(
          previousFrameTime.current === null
            ? 0
            : 1000 / (received - previousFrameTime.current),
        );
        previousFrameTime.current = received;
        setFrame(result);
        onObservation({
          frame: result,
          sessionId: observationSession,
          capturedAt,
        });
        setPhase('running');
        setMessage(
          '本机识别中 · 只处理最新帧，不排队积压；实时预览与分析结果独立刷新。',
        );
        const newLogs: VisionLog[] = [];
        const nextCodes = new Map<number, Set<string>>();
        for (const person of result.persons) {
          const previous = codes.current.get(person.id) ?? new Set<string>();
          nextCodes.set(
            person.id,
            new Set(person.actions.map((action) => action.code)),
          );
          for (const action of person.actions)
            if (!previous.has(action.code))
              newLogs.push({
                id: ++logSequence.current,
                at: capturedAt,
                person: person.id,
                label: ACTION_LABELS[action.code],
                evidence: action.evidence,
                urgent: action.code === 'fall_candidate',
              });
        }
        codes.current = nextCodes;
        if (newLogs.length) {
          setLogs((old) => [...newLogs.reverse(), ...old].slice(0, 80));
        }
      };
      stopPump.current = startFramePump(element, analyze, (error) => {
        if (current === generation.current) stop(cameraError(error), true);
      });
    } catch (error) {
      if (current === generation.current) stop(cameraError(error), true);
    }
  }

  const active = phase === 'connecting' || phase === 'running';
  const showAnalysis = view === 'analysis' && !!frame;
  return (
    <section className="panel vision-panel" id="vision">
      <div className="panel-heading">
        <div>
          <h2>
            视觉感知 <span className="vision-ai-tag">YOLO POSE</span>
          </h2>
          <p>真实人员检测 · 17 个身体关键点 · 连续动作候选</p>
        </div>
        <Badge variant={phase === 'running' ? 'secondary' : 'outline'}>
          {phase === 'running'
            ? '本机识别中'
            : phase === 'connecting'
              ? '连接中'
              : '未采集'}
        </Badge>
      </div>
      <div className="vision-toolbar">
        <Button
          onClick={() =>
            active
              ? stop('摄像头已停止，未保留画面；已有候选提示仍需核查。')
              : void start()
          }
          variant={active ? 'outline' : 'default'}
          size="sm"
        >
          {active ? <CameraOff size={15} /> : <Camera size={15} />}{' '}
          {active ? '停止摄像头' : '开启摄像头与识别'}
        </Button>
        <label htmlFor="skeleton-toggle">
          <Switch
            id="skeleton-toggle"
            checked={skeleton}
            disabled={view !== 'analysis'}
            onCheckedChange={(value) => {
              skeletonRef.current = value;
              setSkeleton(value);
              redraw();
            }}
          />
          骨架（识别画面）
        </label>
        <label htmlFor="flip-toggle">
          <Switch
            id="flip-toggle"
            checked={flipped}
            onCheckedChange={(value) => {
              flippedRef.current = value;
              setFlipped(value);
              redraw();
            }}
          />
          左右翻转
        </label>
      </div>
      <div className="vision-view-controls" aria-label="画面模式">
        <Button
          size="sm"
          variant={view === 'live' ? 'secondary' : 'outline'}
          aria-pressed={view === 'live'}
          onClick={() => setView('live')}
        >
          实时预览 · 低延迟
        </Button>
        <Button
          size="sm"
          variant={view === 'analysis' ? 'secondary' : 'outline'}
          aria-pressed={view === 'analysis'}
          onClick={() => setView('analysis')}
        >
          识别画面 · 人物与骨架
        </Button>
      </div>
      <p className="vision-view-note">
        {view === 'live'
          ? '实时预览不等待 YOLO，也不叠加滞后的骨架。查看人物框与骨架请切换“识别画面”；后台识别与事件判断持续运行。'
          : '画面、人物框和骨架来自同一分析帧。结果刷新速度取决于设备，未将旧骨架叠到实时视频上。'}{' '}
        左右翻转同步区域与标注，不改变判断坐标。
      </p>
      <ZoneEditor
        state={zoneState}
        onAction={onZoneAction}
        now={now}
        frame={frame}
        flipped={flipped}
        showObservation={showAnalysis}
      >
        <video
          ref={video}
          muted
          playsInline
          className={
            active
              ? `vision-video ${flipped ? 'vision-flipped' : ''}`
              : 'vision-hidden'
          }
          aria-label="本机摄像头预览"
        />
        <canvas
          ref={canvas}
          className={showAnalysis ? 'vision-canvas' : 'vision-hidden'}
          aria-label="YOLO 人员框与关键点识别画面"
        />
        {!frame && (
          <div
            className={`vision-placeholder ${active ? 'vision-connecting' : ''}`}
          >
            <VideoOff size={30} />
            <h3>{active ? '等待首帧识别' : '摄像头尚未开启'}</h3>
            <p>将电脑固定，让肩部、髋部与腿部尽量进入画面。</p>
            <small>仅本机处理 · 不录制 · 不识别身份</small>
          </div>
        )}
        {frame && (
          <div className="vision-frame-label">
            <ScanLine size={13} />
            {view === 'live'
              ? '实时预览 · 识别在后台运行'
              : `分析帧 #${frame.frame_id} · ${frame.persons.length} 人`}
            {resultAge !== null && ` · 结果年龄 ${resultAge} ms`}
          </div>
        )}
      </ZoneEditor>
      <output
        className={`vision-status ${phase === 'error' ? 'vision-error' : ''}`}
      >
        {message}
      </output>
      {resultAge !== null && resultAge > 500 && (
        <p className="vision-lag-warning">
          识别结果落后当前画面 {resultAge} ms。实时预览不代表识别已同步；超过
          3.5 秒的区域证据会按未知处理。
        </p>
      )}
      <div className="vision-stats">
        <span>
          识别帧率 <b>{frame && fps ? fps.toFixed(1) : '—'} FPS</b>
        </span>
        <span>
          模型耗时 <b>{frame ? `${frame.inference_ms.toFixed(0)} ms` : '—'}</b>
        </span>
        <span>
          计算设备 <b>{frame?.device.toUpperCase() ?? '—'}</b>
        </span>
        <span>
          编码 <b>{timing ? `${timing.encodingMs.toFixed(0)} ms` : '—'}</b>
        </span>
        <span>
          本机请求往返{' '}
          <b>{timing ? `${timing.transportMs.toFixed(0)} ms` : '—'}</b>
        </span>
        <span>
          采集至结果 <b>{timing ? `${timing.totalMs.toFixed(0)} ms` : '—'}</b>
        </span>
      </div>
      <p className="vision-timing-note">
        分析目标上限 {VISION_TIMING.maxAnalysisFps}{' '}
        FPS，实际值以上方为准。耗时从浏览器复制画面开始，不含摄像头曝光与硬件缓冲延迟；动作确认仍需满足持续时间。
      </p>
      {pendingFalls > 0 && (
        <div className="vision-alerts" aria-live="assertive">
          <article>
            <div>
              <ShieldAlert size={17} />
              <strong>{pendingFalls} 条疑似跌倒事件待处理</strong>
            </div>
            <p>候选证据、接手和核查记录已合并到事件中心。</p>
            <small>
              断流或停止摄像头不会自动删除此提示。仅网页提示，未联系任何人。
            </small>
            <a href="#events">前往事件中心核查 →</a>
          </article>
        </div>
      )}
      <div className="person-list">
        {frame ? (
          frame.persons.length ? (
            frame.persons.map((person) => (
              <article key={person.id}>
                <div className="person-heading">
                  <strong>人物 P{person.id}</strong>
                  <span>检测置信度 {Math.round(person.confidence * 100)}%</span>
                </div>
                <div className="pose-actions">
                  {person.actions.length ? (
                    person.actions.map((action) => (
                      <span key={action.code}>
                        {ACTION_LABELS[action.code]}
                      </span>
                    ))
                  ) : (
                    <span className="pose-unknown">
                      {person.pose_reliable
                        ? '观察中 / 动作不明确'
                        : '关键点不足，动作未知'}
                    </span>
                  )}
                </div>
              </article>
            ))
          ) : (
            <p className="vision-empty">
              本帧未检测到人物，不代表画面外无人或人员安全。
            </p>
          )
        ) : (
          <p className="vision-empty">视觉状态未知，尚无有效识别结果。</p>
        )}
      </div>
      <details className="vision-log">
        <summary>
          动作记录 <span>{logs.length}</span>
        </summary>
        {logs.length ? (
          <ol>
            {logs.map((log) => (
              <li key={log.id}>
                <time>
                  {new Date(log.at).toLocaleTimeString('zh-CN', {
                    hour12: false,
                  })}
                </time>
                <strong>
                  P{log.person} · {log.label}
                </strong>
                <p>{log.evidence}</p>
                {log.urgent && <small>处理状态请查看统一事件中心</small>}
              </li>
            ))}
          </ol>
        ) : (
          <p>稳定识别到动作后才记录，不生成演示结果。</p>
        )}
      </details>
      <p className="vision-disclaimer">
        AI
        检测人物和关键点；动作由几何与时序规则判断，可能误报或漏报。区域停留与疑似跌倒均进入事件中心；跌倒候选覆盖当前画面全部可识别人物，区域停留仅针对人工选定对象。人物编号不是身份识别。请勿为了演示而真实摔倒。
      </p>
    </section>
  );
}
