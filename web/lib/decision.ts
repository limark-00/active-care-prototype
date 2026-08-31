import { VISION_URL } from './vision.ts';

export type RiskLevel = 'L0' | 'L1' | 'L2' | 'L3' | 'L4';
export type InterventionLevel = 'I0' | 'I1' | 'I2' | 'I3' | 'I4';
export type AlertMode = 'NONE' | 'PAGE_WARNING' | 'URGENT_HELP';

export interface Prediction<T> {
  label: T;
  confidence: number;
  probabilities: Record<string, number>;
}
export interface DecisionResult {
  schema_version: 1;
  source: 'local_text_model';
  model: string;
  base_model: string;
  run_name: string;
  device: string;
  input_characters: number;
  segment_count: number;
  inference_ms: number;
  model_output: {
    risk_level: Prediction<RiskLevel>;
    intervention_level: Prediction<InterventionLevel>;
    alert_mode: Prediction<AlertMode>;
    manual_review: Prediction<boolean>;
    abstain: Prediction<boolean>;
  };
  guarded_output: {
    risk_level: RiskLevel;
    intervention_level: InterventionLevel;
    alert_mode: AlertMode;
    manual_review: boolean;
    abstain: boolean;
  };
  guardrail_applied: boolean;
  review_reasons: string[];
  limitations: string;
}

const object = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const unit = (value: unknown): value is number =>
  finite(value) && value >= 0 && value <= 1;
const RISKS = new Set(['L0', 'L1', 'L2', 'L3', 'L4']);
const INTERVENTIONS = new Set(['I0', 'I1', 'I2', 'I3', 'I4']);
const ALERTS = new Set(['NONE', 'PAGE_WARNING', 'URGENT_HELP']);

function validPrediction(value: unknown, labels: Set<unknown>) {
  if (
    !object(value) ||
    !labels.has(value.label) ||
    !unit(value.confidence) ||
    !object(value.probabilities)
  )
    return false;
  const probabilities = Object.values(value.probabilities);
  return (
    probabilities.length === labels.size &&
    probabilities.every(unit) &&
    Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 0.01
  );
}

export function parseDecisionResult(value: unknown): DecisionResult {
  const fail = () => {
    throw new Error('文字决策服务返回格式不兼容，请重启本地服务。');
  };
  if (
    !object(value) ||
    value.schema_version !== 1 ||
    value.source !== 'local_text_model' ||
    typeof value.model !== 'string' ||
    typeof value.base_model !== 'string' ||
    typeof value.run_name !== 'string' ||
    typeof value.device !== 'string' ||
    !finite(value.inference_ms) ||
    value.inference_ms < 0 ||
    !Number.isSafeInteger(value.input_characters) ||
    !Number.isSafeInteger(value.segment_count) ||
    !object(value.model_output) ||
    !object(value.guarded_output) ||
    typeof value.guardrail_applied !== 'boolean' ||
    !Array.isArray(value.review_reasons) ||
    !value.review_reasons.every(
      (reason) => typeof reason === 'string' && reason.length <= 300,
    ) ||
    typeof value.limitations !== 'string'
  )
    return fail();
  const model = value.model_output;
  const guarded = value.guarded_output;
  if (
    !validPrediction(model.risk_level, RISKS) ||
    !validPrediction(model.intervention_level, INTERVENTIONS) ||
    !validPrediction(model.alert_mode, ALERTS) ||
    !validPrediction(model.manual_review, new Set([false, true])) ||
    !validPrediction(model.abstain, new Set([false, true])) ||
    typeof guarded.risk_level !== 'string' ||
    !RISKS.has(guarded.risk_level) ||
    typeof guarded.intervention_level !== 'string' ||
    !INTERVENTIONS.has(guarded.intervention_level) ||
    typeof guarded.alert_mode !== 'string' ||
    !ALERTS.has(guarded.alert_mode) ||
    typeof guarded.manual_review !== 'boolean' ||
    typeof guarded.abstain !== 'boolean'
  )
    return fail();
  return value as unknown as DecisionResult;
}

export async function requestTextDecision(
  text: string,
  signal?: AbortSignal,
): Promise<DecisionResult> {
  const response = await fetch(`${VISION_URL}/decision/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Care-Client': 'active-care-web',
    },
    body: JSON.stringify({ text }),
    cache: 'no-store',
    signal,
  });
  const value: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = object(value) && typeof value.detail === 'string'
      ? value.detail
      : '文字决策请求失败，请检查本地服务。';
    throw new Error(detail);
  }
  return parseDecisionResult(value);
}
