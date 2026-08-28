"""Local Ultralytics adapter. Never downloads weights during a camera session."""
import os
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
(ROOT / '.runtime' / 'ultralytics').mkdir(parents=True, exist_ok=True)
(ROOT / '.runtime' / 'matplotlib').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime' / 'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime' / 'matplotlib'))
# The camera path is deliberately offline, including vendor telemetry/connectivity checks.
os.environ['YOLO_OFFLINE'] = 'true'


class Detector:
    model_name = 'YOLO11n-pose'

    def __init__(self):
        from ultralytics import YOLO, settings
        settings.update({'sync': False})
        from ultralytics.utils.events import events
        events.enabled = False
        weights = ROOT / 'models' / 'yolo11n-pose.pt'
        if not weights.is_file():
            raise FileNotFoundError('模型未下载，请运行 python download_model.py')
        expected = (ROOT / 'model.sha256').read_text().split()[0]
        if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
            raise ValueError('模型校验失败，拒绝加载。请重新从官方来源下载。')
        self.device = os.environ.get('CARE_DEVICE', 'cpu')
        self.model = YOLO(str(weights), task='pose')
        # Warm up before reporting ready, so the first camera frame does not trigger model setup.
        self.predict(np.zeros((480, 640, 3), dtype=np.uint8))

    def predict(self, bgr):
        result = self.model.predict(bgr, imgsz=640, conf=0.4, iou=0.5, max_det=8, device=self.device, verbose=False, save=False)[0]
        if result.boxes is None or result.keypoints is None:
            return []
        boxes = result.boxes.xyxyn.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        points = result.keypoints.xyn.cpu().tolist()
        confidences = result.keypoints.conf.cpu().tolist()
        people = []
        for box, score, keypoints, confidence in zip(boxes, scores, points, confidences):
            if len(keypoints) != 17:
                continue
            people.append({'bbox': [min(1.0, max(0.0, v)) for v in box], 'confidence': float(score), 'keypoints': [
                {'x': min(1.0, max(0.0, p[0])), 'y': min(1.0, max(0.0, p[1])), 'confidence': float(c)} for p, c in zip(keypoints, confidence)
            ]})
        return people
