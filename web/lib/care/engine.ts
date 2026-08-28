import type {
  CareAction,
  CareEvent,
  CareState,
  EnvironmentInput,
  EventKind,
  Intervention,
  Signal,
  Snapshot,
  VisionEvidence,
} from './types.ts';
import { ZONE_POLICY } from '../zones.ts';
import {
  deriveOverview,
  EMPTY_COVERAGE,
  INITIAL_OVERVIEW,
} from './overview.ts';
import {
  adapterResult,
  currentOutput,
  outputKey,
  responseWindowStarted,
} from './outputs.ts';

/** Demonstration settings only; not medical/environmental safety standards. */
export const POLICY = {
  confirmMs: 5000,
  responseMs: 20000,
  recoveryMs: 5000,
  staleMs: 3500,
} as const;
export const INITIAL_INPUT: EnvironmentInput = {
  temperature: 24,
  humidity: 50,
  smoke: 'normal',
  gas: 'normal',
  doorOpen: false,
  online: true,
};
export const TITLES: Record<EventKind, string> = {
  temperature: '温度超出演示范围',
  humidity: '湿度超出演示范围',
  smoke: '烟雾模拟报警',
  gas: '燃气模拟报警',
  device: '模拟输入不可用',
  zone_dwell: '区域持续停留',
  fall_candidate: '疑似跌倒 · 待人工核查',
};
export const PHASE_LABELS = {
  CONFIRMING: '观察确认中',
  INTERVENING: '等待干预呈现',
  WAITING_RESPONSE: '等待回应',
  ESCALATED: '需要协助',
  RECOVERING: '恢复观察中',
  CLOSED: '已结束',
} as const;
export const INTERVENTION_LABELS: Record<Intervention, string> = {
  I0: '观察',
  I1: '温和提醒',
  I2: '图示引导',
  I3: '保护请求（未接入）',
  I4: '网页求助',
};
const KINDS: EventKind[] = [
  'temperature',
  'humidity',
  'smoke',
  'gas',
  'device',
];
const urgent = (kind: EventKind) => kind === 'smoke' || kind === 'gas';

export function initialState(): CareState {
  return {
    snapshot: null,
    camera: [],
    vision: null,
    visionCursor: null,
    coverage: { ...EMPTY_COVERAGE },
    actions: [],
    overview: { ...INITIAL_OVERVIEW },
    transitions: [],
    profile: 'voice',
    events: [],
    logs: [],
    sequence: 0,
    lastNow: 0,
    notice: '',
  };
}
export function inputIsAvailable(s: Snapshot | null, now: number): boolean {
  return !!s && s.online && now - s.sampledAt <= POLICY.staleMs;
}
function signalFor(
  kind: EventKind,
  s: Snapshot,
  now: number,
  active: boolean,
): { signal: Signal; evidence: string } {
  const live = inputIsAvailable(s, now);
  const validTemperature =
    Number.isFinite(s.temperature) &&
    s.temperature >= -40 &&
    s.temperature <= 80;
  const validHumidity =
    Number.isFinite(s.humidity) && s.humidity >= 0 && s.humidity <= 100;
  const alarmKnown = (value: string) => value === 'normal' || value === 'alarm';
  if (kind === 'device') {
    const good =
      live &&
      validTemperature &&
      validHumidity &&
      alarmKnown(s.smoke) &&
      alarmKnown(s.gas);
    return {
      signal: good ? 'normal' : 'abnormal',
      evidence: !live
        ? '模拟采样暂停或数据过期；不能据此判断环境正常。'
        : good
          ? '模拟采样与输入状态已恢复。'
          : '存在无效读数或模拟报警器故障/离线。',
    };
  }
  if (!live)
    return {
      signal: 'unknown',
      evidence: '模拟采样不可用；保留已有事件，等待有效数据。',
    };
  if (kind === 'smoke' || kind === 'gas') {
    if (!alarmKnown(s[kind]))
      return {
        signal: 'unknown',
        evidence: '报警器模拟状态为故障或离线，不能确认解除。',
      };
    return {
      signal: s[kind] === 'alarm' ? 'abnormal' : 'normal',
      evidence:
        s[kind] === 'alarm'
          ? '模拟报警输入已激活，直接创建网页求助事件。'
          : '模拟报警输入已恢复正常；仍须完成事件处理。',
    };
  }
  const temp = kind === 'temperature';
  if (!(temp ? validTemperature : validHumidity))
    return { signal: 'unknown', evidence: '读数无效，不能判断当前状态。' };
  const value = temp ? s.temperature : s.humidity;
  const bounds = temp
    ? active
      ? [19, 27]
      : [18, 28]
    : active
      ? [35, 65]
      : [30, 70];
  const bad = value < bounds[0] || value > bounds[1];
  return {
    signal: bad ? 'abnormal' : 'normal',
    evidence: `${temp ? '温度' : '湿度'} ${value}${temp ? '°C' : '%'}；${active ? '恢复' : '触发'}演示范围 ${bounds[0]}–${bounds[1]}${temp ? '°C' : '%'}。非安全标准。`,
  };
}

function log(
  s: CareState,
  now: number,
  eventId: string | null,
  message: string,
) {
  s.sequence += 1;
  s.logs = [{ id: s.sequence, at: now, eventId, message }, ...s.logs].slice(
    0,
    250,
  );
}
function changeIntervention(
  s: CareState,
  e: CareEvent,
  value: Intervention,
  now: number,
  reason: string,
) {
  if (e.intervention === value) return;
  e.intervention = value;
  e.interventionAt = now;
  e.phase =
    e.signal === 'normal' && e.normalSince !== null
      ? 'RECOVERING'
      : value === 'I4'
        ? 'ESCALATED'
        : 'INTERVENING';
  log(
    s,
    now,
    e.id,
    `${value} ${INTERVENTION_LABELS[value]}：${reason}（仅网页呈现）`,
  );
}
function evaluateEnvironment(s: CareState, now: number) {
  if (!s.snapshot) return;
  for (const kind of KINDS) {
    let e = s.events.find(
      (item) => item.kind === kind && item.phase !== 'CLOSED',
    );
    const result = signalFor(kind, s.snapshot, now, !!e);
    if (!e && result.signal === 'abnormal') {
      s.sequence += 1;
      e = {
        id: `EV-${String(s.sequence).padStart(4, '0')}`,
        kind,
        title: TITLES[kind],
        risk: urgent(kind) ? 'L4' : 'L1',
        intervention: urgent(kind) ? 'I4' : 'I0',
        phase: urgent(kind) ? 'ESCALATED' : 'CONFIRMING',
        signal: 'abnormal',
        evidence: result.evidence,
        source: 'simulated',
        profile: s.profile,
        createdAt: now,
        abnormalSince: now,
        updatedAt: now,
        interventionAt: urgent(kind) ? now : null,
        normalSince: null,
        acknowledgedAt: null,
        feedback: 'none',
        closedAt: null,
        falseAlarmNote: null,
      };
      s.events = [e, ...s.events];
      log(
        s,
        now,
        e.id,
        urgent(kind)
          ? `${e.title} → L4 / I4；网页待处理，未联系外部人员。`
          : `${e.title} → L1 / I0，开始观察。`,
      );
    }
    if (!e) continue;
    const previousSignal = e.signal;
    e.evidence = result.evidence;
    e.signal = result.signal;
    e.updatedAt = now;
    if (result.signal === 'unknown') {
      e.normalSince = null;
      e.abnormalSince = null;
      if (previousSignal !== 'unknown')
        log(s, now, e.id, '观测丢失：事件保留，不自动解除。');
      continue;
    }
    if (result.signal === 'normal') {
      e.abnormalSince = null;
      if (e.normalSince === null) {
        e.normalSince = now;
        e.phase = 'RECOVERING';
        log(s, now, e.id, '输入恢复，开始稳定观察；尚未关闭事件。');
      }
      continue;
    }
    if (previousSignal !== 'abnormal') {
      e.normalSince = null;
      e.abnormalSince = now;
      e.phase =
        e.intervention === 'I4'
          ? 'ESCALATED'
          : e.intervention === 'I0'
            ? 'CONFIRMING'
            : 'INTERVENING';
      e.interventionAt = e.intervention === 'I0' ? null : now;
      log(s, now, e.id, '有效异常输入再次出现，重新开始连续观察。');
    }
    if (urgent(kind)) continue;
    if (
      e.intervention === 'I0' &&
      e.abnormalSince !== null &&
      now - e.abnormalSince >= POLICY.confirmMs
    ) {
      e.risk = kind === 'device' ? 'L1' : 'L2';
      changeIntervention(
        s,
        e,
        e.profile === 'visual' ? 'I2' : 'I1',
        now,
        e.profile === 'visual'
          ? '档案预设需要图示支持。'
          : '持续异常，先给出简短提醒。',
      );
    } else if (
      e.interventionAt !== null &&
      e.acknowledgedAt === null &&
      e.feedback === 'none' &&
      responseWindowStarted(s, e) !== null &&
      now - responseWindowStarted(s, e)! >= POLICY.responseMs &&
      e.intervention !== 'I4'
    ) {
      changeIntervention(
        s,
        e,
        e.intervention === 'I1' ? 'I2' : 'I4',
        now,
        '未收到人工反馈，增加支持；风险等级不因未回应而提高。',
      );
    }
  }
}

function evaluateCamera(s: CareState, now: number) {
  // Only a threshold-crossing observation may create a camera event.
  for (const o of s.camera) {
    if (
      o.signal !== 'abnormal' ||
      !o.triggered ||
      !Number.isFinite(o.capturedAt) ||
      now - o.capturedAt > ZONE_POLICY.staleMs ||
      o.capturedAt > now
    )
      continue;
    if (
      s.events.some(
        (e) =>
          e.source === 'camera' &&
          e.cameraKey === o.key &&
          e.phase !== 'CLOSED',
      )
    )
      continue;
    const intervention = s.profile === 'visual' ? 'I2' : 'I1';
    const e: CareEvent = {
      id: `EV-${String(++s.sequence).padStart(4, '0')}`,
      kind: 'zone_dwell',
      title: o.title,
      risk: 'L2',
      intervention,
      phase: 'INTERVENING',
      signal: 'abnormal',
      evidence: o.evidence,
      source: 'camera',
      cameraKey: o.key,
      target: o.target,
      profile: s.profile,
      createdAt: now,
      updatedAt: now,
      abnormalSince: now,
      interventionAt: now,
      normalSince: null,
      acknowledgedAt: null,
      feedback: 'none',
      closedAt: null,
      falseAlarmNote: null,
    };
    s.events = [e, ...s.events];
    log(
      s,
      now,
      e.id,
      `摄像头：${o.title}达到停留阈值 → L2 / ${intervention}；仅网页提示。`,
    );
  }
  for (const e of s.events.filter(
    (e) => e.kind === 'zone_dwell' && e.phase !== 'CLOSED',
  )) {
    const o = s.camera.find((o) => o.key === e.cameraKey);
    const signal =
      o &&
      Number.isFinite(o.capturedAt) &&
      o.capturedAt <= now &&
      now - o.capturedAt <= ZONE_POLICY.staleMs
        ? o.signal
        : 'unknown';
    const old = e.signal;
    e.signal = signal;
    e.updatedAt = now;
    e.evidence =
      signal === 'unknown'
        ? `${e.target}：观测暂停、对象丢失、返回过期或位置无法确认；保留此事件。`
        : o!.evidence;
    if (signal === 'unknown') {
      e.normalSince = null;
      e.abnormalSince = null;
      if (old !== 'unknown')
        log(s, now, e.id, '视觉证据不可用：暂停推断，已有区域事件不自动解除。');
      continue;
    }
    if (signal === 'normal') {
      e.abnormalSince = null;
      if (e.normalSince === null) {
        e.normalSince = o!.capturedAt;
        e.phase = 'RECOVERING';
        log(s, now, e.id, '同一观察对象已确认在区域外，开始恢复观察。');
      }
      continue;
    }
    if (old !== 'abnormal') {
      e.normalSince = null;
      e.abnormalSince = now;
      e.interventionAt = now;
      e.phase = e.intervention === 'I4' ? 'ESCALATED' : 'INTERVENING';
      log(
        s,
        now,
        e.id,
        '重新观测到原对象仍在原区域内：保留原事件，重新等待反馈。',
      );
    }
    if (
      e.intervention !== 'I4' &&
      e.interventionAt !== null &&
      e.acknowledgedAt === null &&
      e.feedback === 'none' &&
      responseWindowStarted(s, e) !== null &&
      now - responseWindowStarted(s, e)! >= POLICY.responseMs
    )
      changeIntervention(
        s,
        e,
        e.intervention === 'I1' ? 'I2' : 'I4',
        now,
        '区域停留仍有视觉证据，未收到人工反馈；增加支持，不提高风险等级。',
      );
  }
}
function evaluate(s: CareState, now: number) {
  evaluateEnvironment(s, now);
  evaluateCamera(s, now);
  for (const e of s.events.filter(
    (e) => e.kind === 'fall_candidate' && e.phase !== 'CLOSED',
  )) {
    const v = s.vision;
    const observed =
      v &&
      v.capturedAt <= now &&
      now - v.capturedAt <= POLICY.staleMs &&
      v.persons.some(
        (p) => `${v.sessionId}:P${p.id}` === e.cameraKey && p.reliable,
      );
    const next = observed ? 'abnormal' : 'unknown';
    if (e.signal !== next)
      log(
        s,
        now,
        e.id,
        observed
          ? '该人物重新可见；疑似跌倒仍须人工核查，姿态变化不等于风险解除。'
          : '疑似跌倒后观测不足；保留高关注事件与网页求助。',
      );
    e.signal = next;
    e.updatedAt = now;
  }
}

function acceptVision(s: CareState, v: VisionEvidence | null, now: number) {
  if (!v) {
    s.vision = null;
    return;
  }
  if (
    !Number.isFinite(v.capturedAt) ||
    v.capturedAt > now ||
    !Number.isInteger(v.frameId) ||
    v.frameId < 0
  )
    return;
  const cursor = s.visionCursor;
  if (
    cursor?.sessionId === v.sessionId &&
    (v.frameId <= cursor.frameId || v.capturedAt <= cursor.capturedAt)
  )
    return;
  const old = s.vision;
  s.vision = { ...v, persons: v.persons.map((p) => ({ ...p })) };
  s.visionCursor = {
    sessionId: v.sessionId,
    frameId: v.frameId,
    capturedAt: v.capturedAt,
  };
  for (const p of v.persons) {
    if (!p.fallEvidence || !p.reliable) continue;
    const key = `${v.sessionId}:P${p.id}`;
    if (
      s.events.some(
        (e) =>
          e.kind === 'fall_candidate' &&
          e.cameraKey === key &&
          e.phase !== 'CLOSED',
      )
    )
      continue;
    if (
      old?.sessionId === v.sessionId &&
      old.persons.some(
        (previous) => previous.id === p.id && previous.fallEvidence,
      )
    )
      continue;
    const delayed = now - v.capturedAt > POLICY.staleMs;
    const e: CareEvent = {
      id: `EV-${String(++s.sequence).padStart(4, '0')}`,
      kind: 'fall_candidate',
      title: TITLES.fall_candidate,
      risk: 'L3',
      intervention: 'I4',
      phase: 'ESCALATED',
      signal: delayed ? 'unknown' : 'abnormal',
      evidence: `${delayed ? '延迟返回的历史候选；当前观测未知。' : ''}采集时间 ${v.capturedAt}：${p.fallEvidence}。仅为规则候选，不能判断意识或伤情。`,
      source: 'camera',
      cameraKey: key,
      target: `${v.sessionId} / P${p.id}（非身份识别）`,
      profile: s.profile,
      createdAt: now,
      updatedAt: now,
      abnormalSince: null,
      interventionAt: now,
      normalSince: null,
      acknowledgedAt: null,
      feedback: 'none',
      closedAt: null,
      falseAlarmNote: null,
    };
    s.events = [e, ...s.events];
    log(
      s,
      now,
      e.id,
      '疑似跌倒候选 → L3 / I4；直接进入网页核查队列，不等待提醒。未联系任何人。',
    );
  }
}

export function canClose(e: CareEvent, s: CareState, now: number): boolean {
  if (e.kind === 'fall_candidate') return false;
  const camera = (s.camera ?? []).find((o) => o.key === e.cameraKey);
  const available =
    e.source === 'camera'
      ? !!camera &&
        camera.signal === 'normal' &&
        camera.capturedAt <= now &&
        now - camera.capturedAt <= ZONE_POLICY.staleMs
      : inputIsAvailable(s.snapshot, now);
  const recoveryTime = e.source === 'camera' ? (camera?.capturedAt ?? 0) : now;
  return (
    e.phase === 'RECOVERING' &&
    e.signal === 'normal' &&
    e.normalSince !== null &&
    recoveryTime - e.normalSince >= POLICY.recoveryMs &&
    available
  );
}
export function activeEvents(s: CareState): CareEvent[] {
  return s.events
    .filter((e) => e.phase !== 'CLOSED')
    .sort(
      (a, b) =>
        Number(b.risk.slice(1)) - Number(a.risk.slice(1)) ||
        a.createdAt - b.createdAt,
    );
}
/** Pure reducer: a deterministic timestamp makes lifecycle tests independent of UI. */
function reduceCare(previous: CareState, action: CareAction): CareState {
  if (!Number.isFinite(action.now) || action.now < previous.lastNow)
    return previous;
  const s: CareState = {
    ...previous,
    events: previous.events.map((e) => ({ ...e })),
    logs: [...previous.logs],
    camera: [...(previous.camera ?? [])],
    actions: (previous.actions ?? []).map((a) => ({ ...a })),
    coverage: { ...(previous.coverage ?? EMPTY_COVERAGE) },
    vision: previous.vision ?? null,
    visionCursor: previous.visionCursor ?? null,
    overview: previous.overview ?? { ...INITIAL_OVERVIEW },
    transitions: [...(previous.transitions ?? [])],
    lastNow: action.now,
    notice: '',
  };
  if (action.type === 'camera') {
    s.camera = action.observations.map((o) => ({ ...o }));
    if (action.coverage) s.coverage = { ...action.coverage };
    if (action.vision !== undefined) acceptVision(s, action.vision, action.now);
    evaluate(s, action.now);
    return s;
  }
  if (action.type === 'sample') {
    if (s.snapshot && action.now - s.snapshot.sampledAt > POLICY.staleMs)
      evaluate(s, action.now);
    s.snapshot = {
      ...action.input,
      source: 'simulated',
      sampledAt: action.now,
    };
    evaluate(s, action.now);
    return s;
  }
  if (action.type === 'tick') {
    evaluate(s, action.now);
    return s;
  }
  if (action.type === 'profile') {
    s.profile = action.profile;
    log(
      s,
      action.now,
      null,
      '能力预设已更新；对之后的新事件生效，已有事件保持原支持策略。',
    );
    return s;
  }
  // Re-evaluate freshness before accepting user mutations.
  evaluate(s, action.now);
  if (action.type === 'output-result') {
    const record = s.actions.find(
      (a) =>
        a.id === action.actionId &&
        a.channel === 'web' &&
        a.status === 'requested',
    );
    const event =
      record &&
      s.events.find((e) => e.id === record.eventId && e.phase !== 'CLOSED');
    if (!record || !event || record.requestKey !== outputKey(event)) return s;
    record.status = action.status;
    record.completed_at = action.now;
    record.result =
      action.status === 'executed'
        ? '提示已在可见网页呈现；不代表患者已看见、听见或理解。'
        : '网页提示未能呈现；不能作为患者未回应的依据。';
    log(
      s,
      action.now,
      event.id,
      `${record.action_type} 执行回执：${record.result}`,
    );
    if (action.status === 'failed')
      changeIntervention(
        s,
        event,
        'I4',
        action.now,
        '提示呈现失败，请人工接手；不提高风险等级。',
      );
    else if (event.phase === 'INTERVENING') event.phase = 'WAITING_RESPONSE';
    return s;
  }
  const e = s.events.find(
    (item) => item.id === action.id && item.phase !== 'CLOSED',
  );
  if (!e) return s;
  if (action.type === 'request-output') {
    if (action.channel === 'protection' && e.risk !== 'L3' && e.risk !== 'L4') {
      s.notice = '仅 L3 / L4 可演示保护动作建议；拒绝提醒不构成限制理由。';
      return s;
    }
    const key = `${e.id}:${action.channel}`;
    if (s.actions.some((a) => a.requestKey === key)) return s;
    const result = adapterResult(action.channel);
    s.actions = [
      {
        id: `ACT-${++s.sequence}`,
        eventId: e.id,
        requestKey: key,
        action_type: action.channel === 'protection' ? 'I3' : 'I4',
        channel: action.channel,
        requested_at: action.now,
        completed_at: action.now,
        ...result,
      },
      ...s.actions,
    ];
    log(s, action.now, e.id, result.result);
  } else if (action.type === 'acknowledge' && e.acknowledgedAt === null) {
    e.acknowledgedAt = action.now;
    log(s, action.now, e.id, '演示者/照护者确认接手；不代表风险解除。');
  } else if (action.type === 'feedback') {
    e.feedback = action.feedback;
    log(
      s,
      action.now,
      e.id,
      `人工代填反馈：${{ responded: '已回应（不代表安全）', declined: '拒绝提醒（不等于无自主能力）', help: '请求帮助' }[action.feedback]}。`,
    );
    if (action.feedback === 'help')
      changeIntervention(s, e, 'I4', action.now, '收到人工求助请求。');
  } else if (action.type === 'false-alarm') {
    const note = action.note.trim().slice(0, 200);
    if (!note) {
      s.notice = '请先填写误报原因。';
      return s;
    }
    e.falseAlarmNote = note;
    log(
      s,
      action.now,
      e.id,
      `人工误报标记：${note}。仅作标注，不覆盖持续异常输入。`,
    );
  } else if (action.type === 'close') {
    if (!canClose(e, s, action.now)) {
      s.notice = '输入尚未稳定恢复，或当前数据不可用，不能结束事件。';
      return s;
    }
    e.phase = 'CLOSED';
    e.closedAt = action.now;
    log(s, action.now, e.id, '输入已稳定恢复，人工结束本次事件。');
  } else if (action.type === 'review-close') {
    const note = action.note.trim().slice(0, 200);
    if (
      e.source !== 'camera' ||
      (e.signal !== 'unknown' && e.kind !== 'fall_candidate') ||
      !note
    ) {
      s.notice =
        '仅疑似跌倒或观测未知的视觉事件可人工核查结束，且必须填写实际核查说明。';
      return s;
    }
    e.phase = 'CLOSED';
    e.closedAt = action.now;
    e.reviewNote = note;
    log(
      s,
      action.now,
      e.id,
      `人工核查结束：${note}。此为人工处置记录，不是系统确认风险解除。`,
    );
  }
  e.updatedAt = action.now;
  return s;
}

/** Finalize transitions and idempotent output requests after every accepted action. */
export function careReducer(
  previous: CareState,
  action: CareAction,
): CareState {
  const s = reduceCare(previous, action);
  if (s === previous) return previous;
  for (const e of s.events) {
    if (e.phase === 'CLOSED' || e.intervention === 'I0' || currentOutput(s, e))
      continue;
    s.actions = [
      {
        id: `ACT-${++s.sequence}`,
        eventId: e.id,
        requestKey: outputKey(e),
        action_type: e.intervention,
        channel: 'web',
        requested_at: action.now,
        completed_at: null,
        status: 'requested',
        result: '已决策，等待可见网页呈现回执。',
      },
      ...s.actions,
    ];
  }
  // Superseded requests must never play later or claim execution after closure.
  for (const a of s.actions.filter((a) => a.status === 'requested')) {
    const e = s.events.find((e) => e.id === a.eventId && e.phase !== 'CLOSED');
    if (!e || a.requestKey !== outputKey(e)) {
      a.status = 'unavailable';
      a.completed_at = action.now;
      a.result = '事件已结束或动作已被更新，未执行旧请求。';
    }
  }
  const overview = deriveOverview(s, action.now);
  if (
    overview.code !== s.overview.code ||
    overview.observation !== s.overview.observation ||
    overview.risk !== s.overview.risk
  )
    s.transitions = [
      { ...overview, at: action.now, from: s.overview.code },
      ...s.transitions,
    ].slice(0, 100);
  s.overview = overview;
  return s;
}
