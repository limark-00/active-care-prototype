import type { DecisionResult } from '../decision.ts';
import type {
  AIDecisionPayload,
  CareEvent,
  CareScene,
  EventKind,
  EventSource,
  Intervention,
  ModelRiskLevel,
  Profile,
  UnifiedEventContext,
} from './types.ts';

export const CARE_SCENES: CareScene[] = [
  '厨房',
  '卫生间',
  '卧室',
  '客厅',
  '入户门区域',
  '阳台',
];

const SOURCE_TEXT: Record<EventSource, string> = {
  simulated: '模拟传感器',
  camera: '摄像头',
  manual: '人工输入',
};
const RESPONSE_TEXT: Record<UnifiedEventContext['response'], string> = {
  not_requested: '尚未发出提醒，因此没有回应记录。',
  responded: '已经确认患者对之前的提醒作出回应。',
  improved: '人工报告患者行为已经改善，仍需现场证据确认风险是否解除。',
  no_response: '在当前响应窗口内没有收到患者回应。',
  risk_persisted: '患者曾有回应，但复核时危险状态仍然存在。',
  declined: '患者拒绝了之前的提醒，但拒绝本身不代表失去自主能力。',
  requested_help: '患者已经主动请求帮助。',
  unknown: '当前观测不足，无法判断患者是否回应。',
};
const PREVIOUS_TEXT: Record<Intervention | 'none', string> = {
  none: '之前还没有进行干预。',
  I0: '此前只保持观察，没有发现新的处理结果。',
  I1: '此前已经展示温和提醒，效果需要结合当前回应和现场证据判断。',
  I2: '此前已经展示分步骤引导，效果需要结合当前回应和现场证据判断。',
  I3: '此前只记录了保护动作建议，没有自动执行设备操作。',
  I4: '此前已经在网页请求照护者接手，尚未证明现场风险解除。',
};
const OUTCOME_TEXT: Record<UnifiedEventContext['previousOutcome'], string> = {
  not_available: '目前还没有可验证的处理结果。',
  pending: '患者已经回应，但处理结果仍在等待现场证据验证。',
  effective_reported: '人工报告患者已经采取行动，仍需现场证据确认是否有效。',
  ineffective: '后续复核没有看到危险状态解除。',
  declined: '患者拒绝了本次提醒，不能仅据此推断其失去自主能力。',
  help_requested: '患者主动请求了帮助。',
};

export function createEventContext(input: {
  eventId: string;
  scene: string;
  source: EventSource;
  kind: EventKind;
  title: string;
  evidence: string;
  observedAt: number;
  profile: Profile;
  rawText?: string | null;
}): UnifiedEventContext {
  return {
    schemaVersion: 2,
    revision: 0,
    eventId: input.eventId,
    scene: input.scene.trim().slice(0, 30) || '未指定场景',
    source: input.source,
    kind: input.kind,
    title: input.title.trim().slice(0, 120),
    evidence: input.evidence.trim().slice(0, 1200),
    observedAt: input.observedAt,
    profile: input.profile,
    response: 'not_requested',
    previousIntervention: 'none',
    previousOutcome: 'not_available',
    rawText: input.rawText?.trim().slice(0, 6000) || null,
  };
}

export function buildDecisionText(context: UnifiedEventContext): string {
  if (context.rawText) return context.rawText;
  const capability =
    context.profile === 'visual'
      ? '理解简短语音不稳定，需要图示和分步骤引导；主动求助能力尚不确定。'
      : '能够理解简短语音，也能够主动按键求助。';
  const previous = `${PREVIOUS_TEXT[context.previousIntervention]}${OUTCOME_TEXT[context.previousOutcome]}`;
  return [
    `地点：${context.scene}。请根据下面的现场记录进行判断。`,
    `事件1（${SOURCE_TEXT[context.source]}）：${context.title}。${context.evidence}`,
    `理解与求助能力：${capability}`,
    `当前回应：${RESPONSE_TEXT[context.response]}`,
    `之前的处理：${previous}`,
    '请给出下一步干预、网页警示，以及是否转人工核查。',
  ].join('\n');
}

export function decisionRequestKey(event: CareEvent): string {
  return `${event.id}:${event.context.schemaVersion}:${event.context.revision}:${event.context.observedAt}`;
}

export function decisionPayload(result: DecisionResult): AIDecisionPayload {
  return {
    model: result.model,
    runName: result.run_name,
    device: result.device,
    inferenceMs: result.inference_ms,
    risk: result.guarded_output.risk_level as ModelRiskLevel,
    intervention: result.guarded_output.intervention_level as Intervention,
    alertMode: result.guarded_output.alert_mode,
    manualReview: result.guarded_output.manual_review,
    abstain: result.guarded_output.abstain,
    riskConfidence: result.model_output.risk_level.confidence,
    interventionConfidence: result.model_output.intervention_level.confidence,
    alertConfidence: result.model_output.alert_mode.confidence,
    reviewReasons: result.review_reasons.slice(0, 8),
  };
}

export function sceneFromText(text: string): string {
  const match = text.match(/(?:地点|场景)\s*[：:]\s*([^。；;\n]{1,30})/);
  return match?.[1]?.trim() || '人工输入场景';
}

export function manualTitle(text: string): string {
  const eventLine = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => /^事件\s*\d*\s*[（(]?/.test(line));
  if (!eventLine) return `人工文字事件 · ${sceneFromText(text)}`;
  const title = eventLine.replace(
    /^事件\s*\d*\s*[（(][^）)]*[）)]\s*[：:]?\s*/,
    '',
  );
  return (title || eventLine).slice(0, 42);
}

export function evidenceFromManualText(text: string): string {
  const normalized = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .join(' ');
  return normalized.length > 520 ? `${normalized.slice(0, 517)}…` : normalized;
}
