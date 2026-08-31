import type {
  AIDecisionPayload,
  CareAction,
  CareEvent,
  CareState,
  EnvironmentInput,
  EventKind,
  FeedbackOutcome,
  FeedbackSource,
  Intervention,
  PatientFeedback,
  Signal,
  Snapshot,
  VisionEvidence,
} from './types.ts';
import {
  createEventContext,
  decisionRequestKey,
  evidenceFromManualText,
  manualTitle,
  sceneFromText,
} from './ai.ts';
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
  visualResponseMs: 15000,
  followupMs: 10000,
  recoveryMs: 5000,
  staleMs: 3500,
} as const;
export function responseWindowMs(profile: CareEvent['profile']): number {
  return profile === 'visual' ? POLICY.visualResponseMs : POLICY.responseMs;
}
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
  manual_text: '人工文字事件',
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
export const PROFILE_LABELS: Record<CareEvent['profile'], string> = {
  voice: '可理解简短提醒，可主动求助',
  visual: '需要分步图示，求助能力待确认',
};
export const FEEDBACK_LABELS: Record<PatientFeedback, string> = {
  responded: '已回应，正在处理',
  improved: '已按提醒行动',
  no_response: '响应窗口内无回应',
  risk_persisted: '复核后风险仍在',
  declined: '拒绝本次提醒',
  help: '主动请求帮助',
};
export const FEEDBACK_OUTCOME_LABELS: Record<FeedbackOutcome, string> = {
  pending_verification: '等待现场证据验证',
  verified_recovery: '后续证据已恢复',
  risk_persisted: '后续证据仍异常',
  intervention_escalated: '已增加支持',
  recorded: '已记录，不自动改变风险',
};
const KINDS: EventKind[] = [
  'temperature',
  'humidity',
  'smoke',
  'gas',
  'device',
];
const urgent = (kind: EventKind) => kind === 'smoke' || kind === 'gas';
const RISK_ORDER = ['L0', 'L1', 'L2', 'L3', 'L4'] as const;
const INTERVENTION_ORDER: Intervention[] = ['I0', 'I1', 'I2', 'I3', 'I4'];
const maxRisk = (
  current: CareEvent['risk'],
  proposed: AIDecisionPayload['risk'],
) =>
  proposed === 'L0' ||
  RISK_ORDER.indexOf(proposed) <= RISK_ORDER.indexOf(current)
    ? current
    : proposed;
const maxIntervention = (current: Intervention, proposed: Intervention) =>
  INTERVENTION_ORDER.indexOf(proposed) <= INTERVENTION_ORDER.indexOf(current)
    ? current
    : proposed;

function guardedDecision(decision: AIDecisionPayload): AIDecisionPayload {
  const normalized = {
    ...decision,
    reviewReasons: [...decision.reviewReasons],
  };
  if (normalized.risk === 'L4' && normalized.intervention !== 'I4') {
    normalized.intervention = 'I4';
    normalized.alertMode = 'URGENT_HELP';
    normalized.manualReview = true;
    normalized.reviewReasons.push('网页融合校验：L4至少采用I4与紧急网页求助');
  }
  if (
    normalized.intervention === 'I4' &&
    normalized.alertMode !== 'URGENT_HELP'
  ) {
    normalized.alertMode = 'URGENT_HELP';
    normalized.manualReview = true;
    normalized.reviewReasons.push('网页融合校验：I4必须对应紧急网页求助');
  }
  normalized.reviewReasons = [...new Set(normalized.reviewReasons)].slice(0, 8);
  return normalized;
}

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
    scene: '客厅',
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

function emptyFeedbackState() {
  return {
    feedback: 'none' as const,
    feedbackAt: null,
    feedbackRequestKey: null,
    feedbackHistory: [],
  };
}

function nextSupport(e: CareEvent): Intervention {
  if (e.intervention === 'I0') return e.profile === 'visual' ? 'I2' : 'I1';
  if (e.intervention === 'I1') return 'I2';
  if (e.intervention === 'I2' || e.intervention === 'I3') return 'I4';
  return 'I4';
}

function feedbackResponse(
  feedback: PatientFeedback,
): CareEvent['context']['response'] {
  if (feedback === 'help') return 'requested_help';
  return feedback;
}

function archiveDecisionForRevision(e: CareEvent, now: number) {
  e.context.revision += 1;
  if (!e.aiDecision || e.source === 'manual') return;
  if (e.aiDecision.status === 'running') return;
  e.aiDecisionHistory = [
    { ...e.aiDecision },
    ...(e.aiDecisionHistory ?? []),
  ].slice(0, 12);
  e.aiDecision = null;
  e.updatedAt = now;
}

function feedbackForCurrentOutput(s: CareState, e: CareEvent): boolean {
  const output = currentOutput(s, e);
  return !!output && e.feedbackRequestKey === output.requestKey;
}

function recordFeedback(
  s: CareState,
  e: CareEvent,
  feedback: PatientFeedback,
  source: FeedbackSource,
  now: number,
): boolean {
  const output = currentOutput(s, e);
  const requestKey = output?.requestKey ?? outputKey(e);
  const latest = (e.feedbackHistory ?? []).find(
    (record) => record.requestKey === requestKey,
  );
  const isFollowup =
    feedback === 'risk_persisted' && latest?.outcome === 'pending_verification';
  const isHelpFollowup = feedback === 'help' && latest?.feedback !== 'help';
  if (latest && !isFollowup && !isHelpFollowup) {
    if (source === 'caregiver_report')
      s.notice = '本轮已经记录反馈，请等待新的干预轮次。';
    return false;
  }
  if (
    source === 'caregiver_report' &&
    feedback !== 'help' &&
    (!output || output.status !== 'executed')
  ) {
    s.notice = '当前提示尚未在网页呈现，不能记录患者对本轮提示的回应。';
    return false;
  }

  const before = e.intervention;
  const next =
    feedback === 'help'
      ? 'I4'
      : feedback === 'no_response' || feedback === 'risk_persisted'
        ? nextSupport(e)
        : before;
  if (isFollowup && latest) {
    latest.outcome = 'risk_persisted';
    latest.interventionAfter = next;
  }
  e.feedback = feedback;
  e.feedbackAt = now;
  e.feedbackRequestKey = requestKey;
  e.context.response = feedbackResponse(feedback);
  e.context.previousIntervention = before;
  e.context.previousOutcome =
    feedback === 'improved'
      ? 'effective_reported'
      : feedback === 'responded'
        ? 'pending'
        : feedback === 'no_response' || feedback === 'risk_persisted'
          ? 'ineffective'
          : feedback === 'declined'
            ? 'declined'
            : 'help_requested';
  if (e.source !== 'manual') e.context.evidence = e.evidence;
  if (next !== before)
    changeIntervention(
      s,
      e,
      next,
      now,
      feedback === 'help'
        ? '收到主动求助请求。'
        : '本轮没有产生可验证效果，按最低必要干预增加支持；风险等级保持不变。',
    );

  const outcome: FeedbackOutcome =
    feedback === 'responded' || feedback === 'improved'
      ? 'pending_verification'
      : next !== before
        ? 'intervention_escalated'
        : 'recorded';
  s.sequence += 1;
  e.feedbackHistory = [
    {
      id: `FDB-${s.sequence}`,
      eventId: e.id,
      requestKey,
      at: now,
      source,
      feedback,
      signal: e.signal,
      interventionBefore: before,
      interventionAfter: next,
      outcome,
      reason:
        source === 'system_timeout'
          ? feedback === 'risk_persisted'
            ? `记录回应后继续复核 ${POLICY.followupMs / 1000} 秒，现场证据仍异常。`
            : `网页提示呈现后 ${responseWindowMs(e.profile) / 1000} 秒内未记录回应，且当前证据仍异常。`
          : feedback === 'help'
            ? '演示者记录了患者主动求助。'
            : '由演示者代填患者反馈，仍需摄像头或环境证据验证结果。',
    },
    ...(e.feedbackHistory ?? []),
  ].slice(0, 30);
  log(
    s,
    now,
    e.id,
    `${source === 'system_timeout' ? '系统计时' : '人工代填'}反馈：${FEEDBACK_LABELS[feedback]}；${before === next ? `保持 ${next}` : `${before} → ${next}`}，风险保持 ${e.risk}。`,
  );
  archiveDecisionForRevision(e, now);
  return true;
}

function verifyFeedbackRecovery(s: CareState, e: CareEvent, now: number) {
  const pending = (e.feedbackHistory ?? []).find(
    (record) => record.outcome === 'pending_verification',
  );
  if (!pending) return;
  pending.outcome = 'verified_recovery';
  pending.interventionAfter = e.intervention;
  log(
    s,
    now,
    e.id,
    '后续有效证据已恢复；本轮反馈标记为已验证，事件仍需完成稳定观察。',
  );
}

function evaluateResponseLoop(s: CareState, e: CareEvent, now: number) {
  if (e.signal === 'normal') {
    verifyFeedbackRecovery(s, e, now);
    return;
  }
  if (
    e.signal !== 'abnormal' ||
    e.intervention === 'I0' ||
    e.intervention === 'I4' ||
    e.acknowledgedAt !== null
  )
    return;
  const startedAt = responseWindowStarted(s, e);
  if (startedAt === null) return;
  const output = currentOutput(s, e);
  if (!output) return;
  if (!feedbackForCurrentOutput(s, e)) {
    if (now - startedAt >= responseWindowMs(e.profile))
      recordFeedback(s, e, 'no_response', 'system_timeout', now);
    return;
  }
  const pending = (e.feedbackHistory ?? []).find(
    (record) =>
      record.requestKey === output.requestKey &&
      record.outcome === 'pending_verification',
  );
  if (pending && now - pending.at >= POLICY.followupMs)
    recordFeedback(s, e, 'risk_persisted', 'system_timeout', now);
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
      const id = `EV-${String(s.sequence).padStart(4, '0')}`;
      const risk = urgent(kind) ? 'L4' : 'L1';
      const intervention = urgent(kind) ? 'I4' : 'I0';
      e = {
        id,
        kind,
        title: TITLES[kind],
        risk,
        intervention,
        phase: urgent(kind) ? 'ESCALATED' : 'CONFIRMING',
        signal: 'abnormal',
        evidence: result.evidence,
        source: 'simulated',
        context: createEventContext({
          eventId: id,
          scene: s.scene,
          source: 'simulated',
          kind,
          title: TITLES[kind],
          evidence: result.evidence,
          observedAt: now,
          profile: s.profile,
        }),
        ruleDecision: {
          risk,
          intervention,
          reason: urgent(kind)
            ? '独立烟雾/燃气报警硬规则直接求助。'
            : '异常刚出现，先连续确认。',
        },
        aiDecision: null,
        aiDecisionHistory: [],
        profile: s.profile,
        createdAt: now,
        abnormalSince: now,
        updatedAt: now,
        interventionAt: urgent(kind) ? now : null,
        normalSince: null,
        acknowledgedAt: null,
        ...emptyFeedbackState(),
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
      evaluateResponseLoop(s, e, now);
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
      e.ruleDecision?.intervention === 'I0' &&
      e.abnormalSince !== null &&
      now - e.abnormalSince >= POLICY.confirmMs
    ) {
      const ruleRisk = kind === 'device' ? 'L1' : 'L2';
      const ruleIntervention = e.profile === 'visual' ? 'I2' : 'I1';
      const reason =
        e.profile === 'visual'
          ? '档案预设需要图示支持。'
          : '持续异常，先给出简短提醒。';
      e.ruleDecision = {
        risk: ruleRisk,
        intervention: ruleIntervention,
        reason,
      };
      e.risk = maxRisk(e.risk, ruleRisk);
      changeIntervention(
        s,
        e,
        maxIntervention(e.intervention, ruleIntervention),
        now,
        reason,
      );
    }
    evaluateResponseLoop(s, e, now);
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
    const id = `EV-${String(++s.sequence).padStart(4, '0')}`;
    const e: CareEvent = {
      id,
      kind: 'zone_dwell',
      title: o.title,
      risk: 'L2',
      intervention,
      phase: 'INTERVENING',
      signal: 'abnormal',
      evidence: o.evidence,
      source: 'camera',
      context: createEventContext({
        eventId: id,
        scene: s.scene,
        source: 'camera',
        kind: 'zone_dwell',
        title: o.title,
        evidence: o.evidence,
        observedAt: o.capturedAt,
        profile: s.profile,
      }),
      ruleDecision: {
        risk: 'L2',
        intervention,
        reason: '同一观察对象达到区域停留阈值。',
      },
      aiDecision: null,
      aiDecisionHistory: [],
      cameraKey: o.key,
      target: o.target,
      profile: s.profile,
      createdAt: now,
      updatedAt: now,
      abnormalSince: now,
      interventionAt: now,
      normalSince: null,
      acknowledgedAt: null,
      ...emptyFeedbackState(),
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
      evaluateResponseLoop(s, e, now);
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
    evaluateResponseLoop(s, e, now);
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
    const id = `EV-${String(++s.sequence).padStart(4, '0')}`;
    const evidence = `${delayed ? '延迟返回的历史候选；当前观测未知。' : ''}采集时间 ${v.capturedAt}：${p.fallEvidence}。仅为规则候选，不能判断意识或伤情。`;
    const e: CareEvent = {
      id,
      kind: 'fall_candidate',
      title: TITLES.fall_candidate,
      risk: 'L3',
      intervention: 'I4',
      phase: 'ESCALATED',
      signal: delayed ? 'unknown' : 'abnormal',
      evidence,
      source: 'camera',
      context: createEventContext({
        eventId: id,
        scene: s.scene,
        source: 'camera',
        kind: 'fall_candidate',
        title: TITLES.fall_candidate,
        evidence,
        observedAt: v.capturedAt,
        profile: s.profile,
      }),
      ruleDecision: {
        risk: 'L3',
        intervention: 'I4',
        reason: '疑似跌倒规则候选直接进入人工核查队列。',
      },
      aiDecision: null,
      aiDecisionHistory: [],
      cameraKey: key,
      target: `${v.sessionId} / P${p.id}（非身份识别）`,
      profile: s.profile,
      createdAt: now,
      updatedAt: now,
      abnormalSince: null,
      interventionAt: now,
      normalSince: null,
      acknowledgedAt: null,
      ...emptyFeedbackState(),
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
  if (e.kind === 'fall_candidate' || e.source === 'manual') return false;
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
    events: previous.events.map((e) => ({
      ...e,
      context: { ...e.context },
      ruleDecision: e.ruleDecision ? { ...e.ruleDecision } : null,
      aiDecision: e.aiDecision
        ? {
            ...e.aiDecision,
            decision: e.aiDecision.decision
              ? {
                  ...e.aiDecision.decision,
                  reviewReasons: [...e.aiDecision.decision.reviewReasons],
                }
              : null,
          }
        : null,
      aiDecisionHistory: (e.aiDecisionHistory ?? []).map((record) => ({
        ...record,
        decision: record.decision
          ? {
              ...record.decision,
              reviewReasons: [...record.decision.reviewReasons],
            }
          : null,
      })),
      feedbackHistory: (e.feedbackHistory ?? []).map((record) => ({
        ...record,
      })),
    })),
    logs: [...previous.logs],
    camera: [...(previous.camera ?? [])],
    actions: (previous.actions ?? []).map((a) => ({ ...a })),
    coverage: { ...(previous.coverage ?? EMPTY_COVERAGE) },
    vision: previous.vision ?? null,
    visionCursor: previous.visionCursor ?? null,
    overview: previous.overview ?? { ...INITIAL_OVERVIEW },
    transitions: [...(previous.transitions ?? [])],
    scene: previous.scene ?? '客厅',
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
  if (action.type === 'scene') {
    if (s.scene === action.scene) return s;
    s.scene = action.scene;
    log(
      s,
      action.now,
      null,
      `当前监测场景切换为${action.scene}；只影响之后创建的统一事件。`,
    );
    return s;
  }
  if (action.type === 'manual-ai-event') {
    const decision = guardedDecision(action.decision);
    if (decision.risk === 'L0' && decision.intervention === 'I0') {
      log(
        s,
        action.now,
        null,
        '人工文字经AI判断为 L0 / I0，仅保留本次页面结果，不创建待处理事件。',
      );
      return s;
    }
    const id = `EV-${String(++s.sequence).padStart(4, '0')}`;
    const risk = decision.risk === 'L0' ? 'L1' : decision.risk;
    if (decision.risk === 'L0') {
      decision.manualReview = true;
      decision.reviewReasons = [
        ...decision.reviewReasons,
        '风险与干预输出不一致，活动事件按最低L1呈现',
      ];
    }
    const context = createEventContext({
      eventId: id,
      scene: sceneFromText(action.text),
      source: 'manual',
      kind: 'manual_text',
      title: manualTitle(action.text),
      evidence: evidenceFromManualText(action.text),
      observedAt: action.now,
      profile: s.profile,
    });
    const intervention = decision.intervention;
    const event: CareEvent = {
      id,
      kind: 'manual_text',
      title: context.title,
      risk,
      intervention,
      phase:
        intervention === 'I4'
          ? 'ESCALATED'
          : intervention === 'I0'
            ? 'CONFIRMING'
            : 'INTERVENING',
      signal: 'unknown',
      evidence: context.evidence,
      source: 'manual',
      context,
      ruleDecision: null,
      aiDecision: {
        status: 'completed',
        requestKey: `manual:${id}`,
        requestedAt: action.now,
        completedAt: action.now,
        decision,
        applied: true,
        finalRisk: risk,
        finalIntervention: intervention,
        error: null,
      },
      aiDecisionHistory: [],
      profile: s.profile,
      createdAt: action.now,
      abnormalSince: null,
      updatedAt: action.now,
      interventionAt: intervention === 'I0' ? null : action.now,
      normalSince: null,
      acknowledgedAt: null,
      ...emptyFeedbackState(),
      closedAt: null,
      falseAlarmNote: null,
    };
    s.events = [event, ...s.events];
    log(
      s,
      action.now,
      id,
      `人工文字 → AI ${decision.risk} / ${decision.intervention} / ${decision.alertMode}；已进入事件中心，须人工核查结束。`,
    );
    return s;
  }
  // Re-evaluate freshness before accepting user mutations.
  evaluate(s, action.now);
  if (action.type === 'ai-start') {
    const event = s.events.find(
      (item) => item.id === action.id && item.phase !== 'CLOSED',
    );
    if (
      !event ||
      event.aiDecision !== null ||
      action.requestKey !== decisionRequestKey(event)
    )
      return s;
    event.aiDecision = {
      status: 'running',
      requestKey: action.requestKey,
      requestedAt: action.now,
      completedAt: null,
      decision: null,
      applied: false,
      finalRisk: null,
      finalIntervention: null,
      error: null,
    };
    event.updatedAt = action.now;
    log(s, action.now, event.id, '统一事件已进入本机AI串行判断队列。');
    return s;
  }
  if (action.type === 'ai-result') {
    const event = s.events.find(
      (item) => item.id === action.id && item.phase !== 'CLOSED',
    );
    if (
      !event ||
      event.aiDecision?.status !== 'running' ||
      event.aiDecision.requestKey !== action.requestKey
    )
      return s;
    const decision = guardedDecision(action.decision);
    if (action.requestKey !== decisionRequestKey(event)) {
      event.aiDecisionHistory = [
        {
          status: 'completed' as const,
          requestKey: action.requestKey,
          requestedAt: event.aiDecision.requestedAt,
          completedAt: action.now,
          decision,
          applied: false,
          finalRisk: event.risk,
          finalIntervention: event.intervention,
          error: '事件反馈已更新，本次旧上下文结果仅存档，未参与融合。',
        },
        ...(event.aiDecisionHistory ?? []),
      ].slice(0, 12);
      event.aiDecision = null;
      event.updatedAt = action.now;
      log(
        s,
        action.now,
        event.id,
        'AI返回时事件反馈已经更新；旧结果未融合，已按新上下文重新排队。',
      );
      return s;
    }
    const mayFuse =
      !decision.abstain &&
      event.signal !== 'normal' &&
      event.phase !== 'RECOVERING';
    if (mayFuse) {
      event.risk = maxRisk(event.risk, decision.risk);
      let nextIntervention = maxIntervention(
        event.intervention,
        decision.intervention,
      );
      if (event.profile === 'visual' && nextIntervention === 'I1')
        nextIntervention = 'I2';
      changeIntervention(
        s,
        event,
        nextIntervention,
        action.now,
        'AI建议通过一致性检查；规则与患者能力预设作为安全下限，不允许自动降级。',
      );
    }
    event.aiDecision = {
      status: 'completed',
      requestKey: action.requestKey,
      requestedAt: event.aiDecision.requestedAt,
      completedAt: action.now,
      decision,
      applied: mayFuse,
      finalRisk: event.risk,
      finalIntervention: event.intervention,
      error: null,
    };
    event.updatedAt = action.now;
    log(
      s,
      action.now,
      event.id,
      mayFuse
        ? `AI建议 ${decision.risk} / ${decision.intervention} / ${decision.alertMode}；安全融合后为 ${event.risk} / ${event.intervention}。`
        : `AI建议 ${decision.risk} / ${decision.intervention}；因拒绝自动判断或事件已恢复，仅记录建议，规则状态不变。`,
    );
    return s;
  }
  if (action.type === 'ai-failed') {
    const event = s.events.find(
      (item) => item.id === action.id && item.phase !== 'CLOSED',
    );
    if (
      !event ||
      event.aiDecision?.status !== 'running' ||
      event.aiDecision.requestKey !== action.requestKey
    )
      return s;
    const failed = {
      ...event.aiDecision,
      status: 'failed',
      completedAt: action.now,
      error: action.error.trim().slice(0, 240) || '本地AI决策服务暂时不可用。',
    } as const;
    if (action.requestKey !== decisionRequestKey(event)) {
      event.aiDecisionHistory = [
        failed,
        ...(event.aiDecisionHistory ?? []),
      ].slice(0, 12);
      event.aiDecision = null;
      event.updatedAt = action.now;
      log(
        s,
        action.now,
        event.id,
        '旧上下文AI请求失败；新反馈上下文继续重新排队。',
      );
      return s;
    }
    event.aiDecision = failed;
    event.updatedAt = action.now;
    log(
      s,
      action.now,
      event.id,
      'AI判断失败；事件继续使用可审计规则，摄像头与报警流程没有中断。',
    );
    return s;
  }
  if (action.type === 'ai-retry') {
    const event = s.events.find(
      (item) => item.id === action.id && item.phase !== 'CLOSED',
    );
    if (!event || event.aiDecision?.status !== 'failed') return s;
    event.aiDecisionHistory = [
      { ...event.aiDecision },
      ...(event.aiDecisionHistory ?? []),
    ].slice(0, 12);
    event.aiDecision = null;
    event.updatedAt = action.now;
    log(s, action.now, event.id, '已重新加入本机AI判断队列。');
    return s;
  }
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
    if (
      action.feedback !== 'help' &&
      action.requestKey !== undefined &&
      action.requestKey !== outputKey(e)
    ) {
      s.notice = '该反馈属于已经结束的干预轮次，未写入当前事件。';
      e.updatedAt = action.now;
      return s;
    }
    recordFeedback(s, e, action.feedback, 'caregiver_report', action.now);
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
    const reviewable =
      e.source === 'manual' ||
      (e.source === 'camera' &&
        (e.signal === 'unknown' || e.kind === 'fall_candidate'));
    if (!reviewable || !note) {
      s.notice =
        '仅人工文字、疑似跌倒或观测未知的视觉事件可人工核查结束，且必须填写实际核查说明。';
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
