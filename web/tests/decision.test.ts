import test from 'node:test';
import assert from 'node:assert/strict';
import { parseDecisionResult } from '../lib/decision.ts';

const prediction = <T>(label: T, probabilities: Record<string, number>) => ({
  label,
  confidence: Math.max(...Object.values(probabilities)),
  probabilities,
});

const fixture = () => ({
  schema_version: 1,
  source: 'local_text_model',
  model: 'MacBERT + Event Transformer V3.1',
  base_model: 'hfl/chinese-macbert-base',
  run_name: 'partial-macbert-v31-fixture',
  device: 'mps',
  input_characters: 120,
  segment_count: 3,
  inference_ms: 120.5,
  model_output: {
    risk_level: prediction('L3', { L0: 0.01, L1: 0.01, L2: 0.08, L3: 0.85, L4: 0.05 }),
    intervention_level: prediction('I3', { I0: 0.01, I1: 0.01, I2: 0.1, I3: 0.83, I4: 0.05 }),
    alert_mode: prediction('PAGE_WARNING', { NONE: 0.03, PAGE_WARNING: 0.9, URGENT_HELP: 0.07 }),
    manual_review: prediction(false, { false: 0.9, true: 0.1 }),
    abstain: prediction(false, { false: 0.98, true: 0.02 }),
  },
  guarded_output: {
    risk_level: 'L3',
    intervention_level: 'I3',
    alert_mode: 'PAGE_WARNING',
    manual_review: false,
    abstain: false,
  },
  guardrail_applied: false,
  review_reasons: [],
  limitations: '合成模型',
});

await test('text decision parser accepts the local five-head model contract', () => {
  const value = parseDecisionResult(fixture());
  assert.equal(value.model_output.intervention_level.label, 'I3');
  assert.equal(value.guarded_output.risk_level, 'L3');
});

await test('text decision parser rejects invalid probabilities and remote sources', () => {
  const invalid = fixture();
  invalid.model_output.intervention_level.probabilities.I3 = 1.2;
  assert.throws(() => parseDecisionResult(invalid));
  assert.throws(() => parseDecisionResult({ ...fixture(), source: 'cloud_model' }));
});

await test('text decision parser rejects invented labels and unsafe metadata', () => {
  const invalid = fixture();
  invalid.guarded_output.intervention_level = 'I5';
  assert.throws(() => parseDecisionResult(invalid));
  assert.throws(() => parseDecisionResult({ ...fixture(), inference_ms: NaN }));
});
