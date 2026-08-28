import type { VisionFrame } from './vision.ts';

export const ZONE_POLICY = {
  staleMs: 3500,
  confirmMs: 600,
  margin: 0.015,
  maxZones: 3,
} as const;
export type Rect = [number, number, number, number];
export interface Zone {
  id: string;
  name: string;
  rect: Rect;
  dwellMs: number;
}
export interface CameraObservation {
  frame: VisionFrame;
  sessionId: string;
  capturedAt: number;
}
export interface ZoneEvidence {
  key: string;
  title: string;
  target: string;
  zoneId: string;
  signal: 'abnormal' | 'normal' | 'unknown';
  triggered: boolean;
  capturedAt: number;
  evidence: string;
}
type Position = 'inside' | 'outside';
export interface ZoneStatus {
  zoneId: string;
  stable: Position | null;
  candidate: Position | null;
  candidateSince: number | null;
  candidateFrames: number;
  enteredAt: number | null;
  dwellMs: number;
  position: Position | 'unknown';
  reason: string;
}
export interface ZoneLog {
  id: number;
  at: number;
  message: string;
}
export interface ZoneState {
  zones: Zone[];
  observation: CameraObservation | null;
  calibrated: string | null;
  selected: { sessionId: string; personId: number } | null;
  armed: boolean;
  statuses: ZoneStatus[];
  logs: ZoneLog[];
  sequence: number;
  lastNow: number;
  notice: string;
}
export type ZoneAction =
  | { type: 'frame'; observation: CameraObservation | null; now: number }
  | { type: 'tick'; now: number }
  | { type: 'select'; personId: number | null; now: number }
  | { type: 'add'; name: string; rect: Rect; dwellMs: number; now: number }
  | { type: 'delete'; id: string; now: number }
  | { type: 'calibrate'; now: number }
  | { type: 'arm'; enabled: boolean; now: number };

export function initialZoneState(): ZoneState {
  return {
    zones: [],
    observation: null,
    calibrated: null,
    selected: null,
    armed: false,
    statuses: [],
    logs: [],
    sequence: 0,
    lastNow: 0,
    notice: '',
  };
}
const calibrationKey = (o: CameraObservation) =>
  `${o.sessionId}:${o.frame.width}x${o.frame.height}`;
export function freshObservation(
  o: CameraObservation | null,
  now: number,
): o is CameraObservation {
  return (
    !!o &&
    Number.isFinite(o.capturedAt) &&
    o.capturedAt <= now &&
    now - o.capturedAt <= ZONE_POLICY.staleMs
  );
}
export function validRect(rect: Rect): boolean {
  return (
    rect.length === 4 &&
    rect.every((v) => Number.isFinite(v) && v >= 0 && v <= 1) &&
    rect[2] - rect[0] >= 0.06 &&
    rect[3] - rect[1] >= 0.06
  );
}
export function normalizeRect(a: [number, number], b: [number, number]): Rect {
  return [
    Math.min(a[0], b[0]),
    Math.min(a[1], b[1]),
    Math.max(a[0], b[0]),
    Math.max(a[1], b[1]),
  ];
}
export function positionInZone(
  point: [number, number],
  rect: Rect,
  previous: Position | null,
): Position | 'unknown' {
  const [x, y] = point,
    [left, top, right, bottom] = rect,
    m = ZONE_POLICY.margin;
  if (x < left - m || x > right + m || y < top - m || y > bottom + m)
    return 'outside';
  if (
    previous === 'inside' ||
    (x > left + m && x < right - m && y > top + m && y < bottom - m)
  )
    return 'inside';
  return 'unknown';
}
function blank(zone: Zone, reason = '等待新的有效观测'): ZoneStatus {
  return {
    zoneId: zone.id,
    stable: null,
    candidate: null,
    candidateSince: null,
    candidateFrames: 0,
    enteredAt: null,
    dwellMs: 0,
    position: 'unknown',
    reason,
  };
}
function note(s: ZoneState, now: number, message: string) {
  s.logs = [{ id: ++s.sequence, at: now, message }, ...s.logs].slice(0, 80);
}
function invalidate(s: ZoneState, reason = '等待新的有效观测') {
  s.statuses = s.zones.map((zone) => blank(zone, reason));
}
function targetPresent(s: ZoneState, now: number): boolean {
  return (
    freshObservation(s.observation, now) &&
    s.selected?.sessionId === s.observation.sessionId &&
    s.observation.frame.persons.some((p) => p.id === s.selected?.personId)
  );
}

export function zoneEvidence(s: ZoneState, now: number): ZoneEvidence[] {
  if (!s.selected) return [];
  const target = `${s.selected.sessionId} / P${s.selected.personId}`;
  return s.zones.map((zone) => {
    const row = s.statuses.find((r) => r.zoneId === zone.id);
    const available =
      s.armed &&
      freshObservation(s.observation, now) &&
      s.calibrated === calibrationKey(s.observation) &&
      targetPresent(s, now);
    const position = available ? (row?.position ?? 'unknown') : 'unknown';
    return {
      key: `${target}/${zone.id}`,
      title: `${zone.name} · 持续停留`,
      target,
      zoneId: zone.id,
      signal:
        position === 'inside'
          ? 'abnormal'
          : position === 'outside'
            ? 'normal'
            : 'unknown',
      triggered: position === 'inside' && (row?.dwellMs ?? 0) >= zone.dwellMs,
      capturedAt: s.observation?.capturedAt ?? 0,
      evidence:
        position === 'inside'
          ? `${target} 的脚踝中点位于「${zone.name}」；连续观测 ${(row!.dwellMs / 1000).toFixed(1)} 秒，演示阈值 ${zone.dwellMs / 1000} 秒。`
          : position === 'outside'
            ? `${target} 的脚踝中点已在「${zone.name}」外；不代表离家或人员安全。`
            : `${target}：${available ? (row?.reason ?? '等待观测') : '观测暂停、过期、对象丢失或机位未确认'}；不自动解除已有事件。`,
    };
  });
}

export function zoneReducer(
  previous: ZoneState,
  action: ZoneAction,
): ZoneState {
  if (!Number.isFinite(action.now) || action.now < previous.lastNow)
    return previous;
  const s: ZoneState = {
    ...previous,
    zones: [...previous.zones],
    statuses: previous.statuses.map((r) => ({ ...r })),
    logs: [...previous.logs],
    lastNow: action.now,
    notice: '',
  };
  if (action.type === 'tick') {
    if (s.armed && !freshObservation(s.observation, action.now)) {
      if (s.statuses.some((r) => r.position !== 'unknown'))
        note(s, action.now, '视觉观测过期：停止累计停留，保留事件。');
      invalidate(s, '视觉观测过期');
    }
    return s;
  }
  if (action.type === 'select') {
    if (
      action.personId !== null &&
      (!freshObservation(s.observation, action.now) ||
        !s.observation.frame.persons.some((p) => p.id === action.personId))
    ) {
      s.notice = '请从当前有效画面中选择人物。';
      return s;
    }
    s.selected =
      action.personId === null
        ? null
        : { sessionId: s.observation!.sessionId, personId: action.personId };
    invalidate(s, '观察对象已变更，等待新帧');
    note(
      s,
      action.now,
      action.personId === null
        ? '已取消观察对象；旧事件保留。'
        : `人工选择 P${action.personId}；不进行身份识别，也不将旧事件转移给新对象。`,
    );
    return s;
  }
  if (
    action.type === 'add' ||
    action.type === 'delete' ||
    action.type === 'calibrate'
  ) {
    if (s.armed) {
      s.notice = '请先暂停区域监测，再修改或确认机位。';
      return s;
    }
    if (action.type === 'delete') {
      s.zones = s.zones.filter((z) => z.id !== action.id);
      invalidate(s, '区域已变更');
      note(s, action.now, '区域已删除；原区域关联的事件仍保留。');
      return s;
    }
    if (!freshObservation(s.observation, action.now)) {
      s.notice = '需要有效摄像头画面才能划区或确认机位。';
      return s;
    }
    if (action.type === 'add') {
      if (
        !validRect(action.rect) ||
        !action.name.trim() ||
        !Number.isFinite(action.dwellMs) ||
        action.dwellMs < 5000 ||
        action.dwellMs > 120000 ||
        s.zones.length >= ZONE_POLICY.maxZones
      ) {
        s.notice = '区域至少占画面宽高各 6%，最多 3 个，停留阈值为 5–120 秒。';
        return s;
      }
      s.zones.push({
        id: `Z${++s.sequence}`,
        name: action.name.trim().slice(0, 24),
        rect: [...action.rect],
        dwellMs: action.dwellMs,
      });
      note(
        s,
        action.now,
        `人工划定「${action.name.trim().slice(0, 24)}」；阈值 ${action.dwellMs / 1000} 秒。`,
      );
    } else
      note(s, action.now, '人工确认当前机位与区域位置；移动电脑后需重新确认。');
    s.calibrated = calibrationKey(s.observation);
    invalidate(s);
    return s;
  }
  if (action.type === 'arm') {
    if (
      action.enabled &&
      (!targetPresent(s, action.now) ||
        !s.observation ||
        s.calibrated !== calibrationKey(s.observation) ||
        !s.zones.length)
    ) {
      s.notice = '请先选择当前人物、划定区域，并确认当前机位。';
      return s;
    }
    s.armed = action.enabled;
    invalidate(s, action.enabled ? '等待新帧开始监测' : '区域监测已暂停');
    note(
      s,
      action.now,
      action.enabled
        ? '已开始区域监测；以新采集画面计时。'
        : '已暂停区域监测；不自动解除已有事件。',
    );
    return s;
  }
  const o = action.observation;
  if (!o) {
    s.observation = null;
    s.calibrated = null;
    s.selected = null;
    s.armed = false;
    invalidate(s, '摄像头已停止');
    if (previous.observation)
      note(
        s,
        action.now,
        '摄像头停止：清除临时对象绑定，保留区域与未处理事件。',
      );
    return s;
  }
  if (
    !Number.isFinite(o.capturedAt) ||
    o.capturedAt > action.now ||
    !o.sessionId
  )
    return previous;
  if (
    s.observation?.sessionId === o.sessionId &&
    (o.frame.frame_id <= s.observation.frame.frame_id ||
      o.capturedAt <= s.observation.capturedAt)
  )
    return previous;
  const changed =
    !!s.observation && calibrationKey(s.observation) !== calibrationKey(o);
  const gap =
    !s.observation ||
    o.sessionId !== s.observation.sessionId ||
    o.capturedAt - s.observation.capturedAt > ZONE_POLICY.staleMs;
  s.observation = o;
  if (changed) {
    s.calibrated = null;
    s.selected = null;
    s.armed = false;
    invalidate(s, '会话或画面尺寸已改变，请重新确认机位与对象');
    note(s, action.now, '会话 / 尺寸发生变化，区域监测停止。');
  }
  if (!s.armed) return s;
  if (gap) invalidate(s, '观测间隔过长，重新确认位置');
  if (!freshObservation(o, action.now)) {
    invalidate(s, '返回画面已过期，不累计停留时间');
    return s;
  }
  const person = o.frame.persons.find(
    (p) =>
      s.selected?.sessionId === o.sessionId && p.id === s.selected.personId,
  );
  if (!person) {
    invalidate(s, '所选对象丢失，请人工重新选择；未自动换人');
    return s;
  }
  const a = person.keypoints[15],
    b = person.keypoints[16];
  if (!a || !b || a.confidence < 0.45 || b.confidence < 0.45) {
    invalidate(s, '脚踝关键点不足，不能确认地面位置');
    return s;
  }
  const point: [number, number] = [(a.x + b.x) / 2, (a.y + b.y) / 2];
  s.statuses = s.zones.map((zone) => {
    const row = s.statuses.find((r) => r.zoneId === zone.id) ?? blank(zone);
    const raw = positionInZone(point, zone.rect, row.stable);
    if (raw === 'unknown') return blank(zone, '位于边界缓冲带，等待确认');
    if (raw !== row.stable) {
      if (row.candidate !== raw) {
        row.candidate = raw;
        row.candidateSince = o.capturedAt;
        row.candidateFrames = 0;
      }
      row.candidateFrames += 1;
      row.position = 'unknown';
      row.reason = raw === 'inside' ? '正在确认区域内位置' : '正在确认离开';
      row.enteredAt = null;
      row.dwellMs = 0;
      if (
        row.candidateFrames < 2 ||
        o.capturedAt - row.candidateSince! < ZONE_POLICY.confirmMs
      )
        return row;
      note(
        s,
        action.now,
        raw === 'inside'
          ? `${zone.name}：${row.stable === 'outside' ? '进入区域' : '检测到区域内位置'}（P${person.id}）。`
          : `${zone.name}：${row.stable === 'inside' ? '离开区域' : '检测到区域外位置'}（P${person.id}）。`,
      );
      row.stable = raw;
      row.enteredAt = raw === 'inside' ? row.candidateSince : null;
    }
    row.candidate = null;
    row.candidateSince = null;
    row.candidateFrames = 0;
    row.position = raw;
    row.reason = '';
    if (raw === 'inside') {
      row.enteredAt ??= o.capturedAt;
      row.dwellMs = Math.max(0, o.capturedAt - row.enteredAt);
    } else {
      row.enteredAt = null;
      row.dwellMs = 0;
    }
    return row;
  });
  return s;
}
