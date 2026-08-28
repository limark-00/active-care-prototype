import { displayRect, displayX } from './vision-runtime.ts';
export const VISION_URL = 'http://127.0.0.1:8001';
export const ACTION_LABELS = {
  standing: '站姿候选',
  seated_candidate: '屈膝 / 坐姿候选',
  leaning: '躯干倾斜',
  horizontal: '横卧姿态候选',
  one_hand_up: '单手举起',
  hands_up: '双手举起',
  moving: '画面内明显位移',
  fall_candidate: '疑似跌倒 · 请人工核查',
} as const;
export type ActionCode = keyof typeof ACTION_LABELS;
export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}
export interface PoseAction {
  code: ActionCode;
  label: string;
  evidence: string;
}
export interface PersonPose {
  id: number;
  bbox: [number, number, number, number];
  confidence: number;
  keypoints: Keypoint[];
  actions: PoseAction[];
  pose_reliable: boolean;
}
export interface VisionFrame {
  schema_version: 1;
  source: 'camera';
  model: string;
  device: string;
  frame_id: number;
  width: number;
  height: number;
  inference_ms: number;
  persons: PersonPose[];
}
export interface VisionLog {
  id: number;
  at: number;
  person: number;
  label: string;
  evidence: string;
  urgent: boolean;
}

const object = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null;
const finite = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v);
const unit = (v: unknown): v is number => finite(v) && v >= 0 && v <= 1;
const integer = (v: unknown): v is number =>
  finite(v) && Number.isSafeInteger(v) && v >= 0;
export function parseVisionFrame(
  value: unknown,
  expectedId: number,
): VisionFrame {
  const fail = () => {
    throw new Error('视觉服务返回格式不兼容，请重启前后端。');
  };
  if (
    !object(value) ||
    value.schema_version !== 1 ||
    value.source !== 'camera' ||
    value.frame_id !== expectedId ||
    !integer(value.frame_id) ||
    typeof value.model !== 'string' ||
    typeof value.device !== 'string' ||
    !integer(value.width) ||
    value.width < 64 ||
    value.width > 1280 ||
    !integer(value.height) ||
    value.height < 64 ||
    value.height > 960 ||
    !finite(value.inference_ms) ||
    value.inference_ms < 0 ||
    !Array.isArray(value.persons) ||
    value.persons.length > 8
  )
    return fail();
  const ids = new Set<number>();
  for (const p of value.persons as unknown[]) {
    if (
      !object(p) ||
      !integer(p.id) ||
      ids.has(p.id) ||
      !unit(p.confidence) ||
      typeof p.pose_reliable !== 'boolean' ||
      !Array.isArray(p.bbox) ||
      p.bbox.length !== 4 ||
      !p.bbox.every(unit) ||
      p.bbox[2] <= p.bbox[0] ||
      p.bbox[3] <= p.bbox[1] ||
      !Array.isArray(p.keypoints) ||
      p.keypoints.length !== 17 ||
      !Array.isArray(p.actions)
    )
      return fail();
    ids.add(p.id);
    if (
      !p.keypoints.every(
        (k: unknown) =>
          object(k) && unit(k.x) && unit(k.y) && unit(k.confidence),
      )
    )
      return fail();
    if (
      !p.actions.every(
        (a: unknown) =>
          object(a) &&
          typeof a.code === 'string' &&
          Object.hasOwn(ACTION_LABELS, a.code) &&
          typeof a.label === 'string' &&
          typeof a.evidence === 'string' &&
          a.evidence.length <= 500,
      )
    )
      return fail();
  }
  return value as unknown as VisionFrame;
}

// COCO-17 skeleton: confidence is checked at BOTH endpoints before drawing.
const EDGES = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
];
export function drawFrame(
  canvas: HTMLCanvasElement,
  source: HTMLCanvasElement,
  frame: VisionFrame,
  skeleton: boolean,
  flipped = false,
) {
  if (canvas.width !== frame.width) canvas.width = frame.width;
  if (canvas.height !== frame.height) canvas.height = frame.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, frame.width, frame.height);
  ctx.save();
  if (flipped) {
    ctx.translate(frame.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(source, 0, 0);
  ctx.restore();
  ctx.font = '14px sans-serif';
  ctx.lineWidth = 2;
  for (const person of frame.persons) {
    const [left, top, right, bottom] = displayRect(person.bbox, flipped);
    ctx.strokeStyle = '#a3edb6';
    ctx.strokeRect(
      left * frame.width,
      top * frame.height,
      (right - left) * frame.width,
      (bottom - top) * frame.height,
    );
    const text = `P${person.id} · ${Math.round(person.confidence * 100)}%`;
    const labelY = Math.max(18, top * frame.height);
    ctx.fillStyle = '#173c30';
    ctx.fillRect(
      left * frame.width,
      labelY - 18,
      ctx.measureText(text).width + 12,
      21,
    );
    ctx.fillStyle = '#fff';
    ctx.fillText(text, left * frame.width + 6, labelY - 3);
    if (!skeleton) continue;
    for (const [a, b] of EDGES) {
      const start = person.keypoints[a],
        end = person.keypoints[b];
      if (start.confidence < 0.45 || end.confidence < 0.45) continue;
      ctx.beginPath();
      ctx.moveTo(
        displayX(start.x, flipped) * frame.width,
        start.y * frame.height,
      );
      ctx.lineTo(displayX(end.x, flipped) * frame.width, end.y * frame.height);
      ctx.stroke();
    }
    for (const k of person.keypoints) {
      if (k.confidence < 0.45) continue;
      ctx.fillStyle = '#ffe7a7';
      ctx.beginPath();
      ctx.arc(
        displayX(k.x, flipped) * frame.width,
        k.y * frame.height,
        3,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }
}

export function cameraError(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError')
      return '摄像头权限未获允许。请在浏览器 / 系统设置中允许摄像头，再重试。';
    if (error.name === 'NotFoundError') return '未找到摄像头，请检查电脑设备。';
    if (error.name === 'NotReadableError')
      return '摄像头无法读取，可能正在被其他应用占用。';
    if (error.name === 'AbortError' || error.name === 'TimeoutError')
      return '视觉请求超时，已停止摄像头；状态未知，请检查本地服务。';
  }
  return error instanceof Error ? error.message : '视觉连接失败，状态未知。';
}
