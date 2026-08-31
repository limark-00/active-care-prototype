import test from 'node:test';
import assert from 'node:assert/strict';
import {
  activeEvents,
  canClose,
  careReducer,
  initialState,
  INITIAL_INPUT,
} from '../lib/care/engine.ts';
import { currentOutput } from '../lib/care/outputs.ts';
import { buildSessionReport } from '../lib/care/session.ts';
import { buildDecisionText, decisionRequestKey } from '../lib/care/ai.ts';
import { initialZoneState } from '../lib/zones.ts';
import type {
  CareState,
  AIDecisionPayload,
  EnvironmentInput,
  EventKind,
  VisionEvidence,
} from '../lib/care/types.ts';

const input = (patch: Partial<EnvironmentInput> = {}): EnvironmentInput => ({
  ...INITIAL_INPUT,
  ...patch,
});
const sample = (
  s: CareState,
  at: number,
  patch: Partial<EnvironmentInput> = {},
) => careReducer(s, { type: 'sample', input: input(patch), now: at });
const event = (s: CareState, kind: EventKind) =>
  activeEvents(s).find((e) => e.kind === kind)!;
function advance(
  s: CareState,
  from: number,
  to: number,
  patch: Partial<EnvironmentInput>,
) {
  for (let now = from; now <= to; now += 1000) s = sample(s, now, patch);
  return s;
}
await test('initial input is unknown until first sample; normal values produce no events', () => {
  assert.equal(initialState().snapshot, null);
  assert.equal(activeEvents(sample(initialState(), 0)).length, 0);
});
await test('smoke immediately creates L4/I4, without waiting for reminders', () => {
  const e = event(sample(initialState(), 0, { smoke: 'alarm' }), 'smoke');
  assert.equal(e.risk, 'L4');
  assert.equal(e.intervention, 'I4');
  assert.equal(e.phase, 'ESCALATED');
  assert.equal(e.source, 'simulated');
});
await test('same sustained alarm does not create duplicate events or logs', () => {
  const s = advance(initialState(), 0, 30000, { smoke: 'alarm' });
  assert.equal(s.events.length, 1);
  assert.equal(s.logs.length, 1);
});
await test('ordinary event waits for continuous evidence and respects capability profile', () => {
  let a = sample(initialState(), 0, { temperature: 32 });
  assert.equal(event(a, 'temperature').intervention, 'I0');
  a = advance(a, 1000, 5000, { temperature: 32 });
  assert.equal(event(a, 'temperature').risk, 'L2');
  assert.equal(event(a, 'temperature').intervention, 'I1');
  let b = careReducer(initialState(), {
    type: 'profile',
    profile: 'visual',
    now: 0,
  });
  b = advance(b, 0, 5000, { temperature: 32 });
  assert.equal(event(b, 'temperature').risk, 'L2');
  assert.equal(event(b, 'temperature').intervention, 'I2');
});
await test('no response increases support without increasing risk or adding restrictions', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  s = deliver(s, event(s, 'temperature').id, 5000);
  s = advance(s, 6000, 25000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I2');
  s = deliver(s, event(s, 'temperature').id, 25000);
  s = advance(s, 26000, 45000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I4');
  assert.equal(event(s, 'temperature').risk, 'L2');
});
await test('acknowledgement is not closure and prevents repeated unattended escalation', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  s = careReducer(s, {
    type: 'acknowledge',
    id: event(s, 'temperature').id,
    now: 5000,
  });
  s = advance(s, 6000, 60000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I1');
  assert.notEqual(event(s, 'temperature').phase, 'CLOSED');
});
await test('refusal does not cause restriction, false closure, or automatic support escalation', () => {
  let s = advance(initialState(), 0, 5000, { humidity: 80 });
  s = deliver(s, event(s, 'humidity').id, 5000);
  const requestKey = currentOutput(s, event(s, 'humidity'))!.requestKey;
  s = careReducer(s, {
    type: 'feedback',
    id: event(s, 'humidity').id,
    feedback: 'declined',
    requestKey,
    now: 6000,
  });
  s = advance(s, 7000, 40000, { humidity: 80 });
  assert.equal(event(s, 'humidity').intervention, 'I1');
  assert.equal(event(s, 'humidity').risk, 'L2');
  assert.equal(event(s, 'humidity').feedbackHistory.length, 1);
});
await test('explicit manual help request creates I4 even for lower risk', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  s = careReducer(s, {
    type: 'feedback',
    id: event(s, 'temperature').id,
    feedback: 'help',
    now: 0,
  });
  assert.equal(event(s, 'temperature').intervention, 'I4');
  assert.equal(event(s, 'temperature').risk, 'L1');
});
await test('patient feedback is bound to a presented intervention, while help is always accepted', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  const e = event(s, 'temperature');
  const pendingKey = currentOutput(s, e)!.requestKey;
  s = careReducer(s, {
    type: 'feedback',
    id: e.id,
    feedback: 'responded',
    requestKey: pendingKey,
    now: 5000,
  });
  assert.equal(event(s, 'temperature').feedback, 'none');
  assert.equal(event(s, 'temperature').feedbackHistory.length, 0);
  assert.match(s.notice, /尚未在网页呈现/);
  s = careReducer(s, {
    type: 'feedback',
    id: e.id,
    feedback: 'help',
    requestKey: pendingKey,
    now: 5000,
  });
  assert.equal(event(s, 'temperature').intervention, 'I4');
  assert.equal(event(s, 'temperature').risk, 'L2');
  assert.equal(event(s, 'temperature').feedbackHistory.length, 1);
});
await test('responded feedback is verified against evidence and persistent risk opens a new round', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  const id = event(s, 'temperature').id;
  s = deliver(s, id, 5000);
  const firstKey = currentOutput(s, event(s, 'temperature'))!.requestKey;
  s = careReducer(s, {
    type: 'feedback',
    id,
    feedback: 'improved',
    requestKey: firstKey,
    now: 6000,
  });
  assert.equal(event(s, 'temperature').intervention, 'I1');
  assert.equal(
    event(s, 'temperature').feedbackHistory[0].outcome,
    'pending_verification',
  );
  s = advance(s, 7000, 15000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I1');
  s = sample(s, 16000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I2');
  assert.equal(event(s, 'temperature').risk, 'L2');
  assert.equal(event(s, 'temperature').feedbackHistory.length, 2);
  assert.equal(
    event(s, 'temperature').feedbackHistory[1].outcome,
    'risk_persisted',
  );

  s = careReducer(s, {
    type: 'feedback',
    id,
    feedback: 'declined',
    requestKey: firstKey,
    now: 16001,
  });
  assert.match(s.notice, /已经结束的干预轮次/);
  assert.equal(event(s, 'temperature').feedbackHistory.length, 2);

  s = deliver(s, id, 16001);
  s = advance(s, 17000, 36000, { temperature: 32 });
  s = sample(s, 36001, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I4');
  assert.equal(event(s, 'temperature').feedbackHistory.length, 3);
  s = sample(s, 36002, { temperature: 32 });
  assert.equal(event(s, 'temperature').feedbackHistory.length, 3);
});
await test('reported improvement never clears risk, but later normal evidence verifies recovery', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  const id = event(s, 'temperature').id;
  s = deliver(s, id, 5000);
  s = careReducer(s, {
    type: 'feedback',
    id,
    feedback: 'improved',
    requestKey: currentOutput(s, event(s, 'temperature'))!.requestKey,
    now: 6000,
  });
  assert.equal(event(s, 'temperature').phase, 'WAITING_RESPONSE');
  s = sample(s, 7000, { temperature: 24 });
  assert.equal(event(s, 'temperature').phase, 'RECOVERING');
  assert.equal(
    event(s, 'temperature').feedbackHistory[0].outcome,
    'verified_recovery',
  );
});
await test('visual-support profile uses a shorter response window without changing risk', () => {
  let s = careReducer(initialState(), {
    type: 'profile',
    profile: 'visual',
    now: 0,
  });
  s = advance(s, 0, 5000, { humidity: 80 });
  const id = event(s, 'humidity').id;
  s = deliver(s, id, 5000);
  s = advance(s, 6000, 19000, { humidity: 80 });
  assert.equal(event(s, 'humidity').intervention, 'I2');
  s = sample(s, 20000, { humidity: 80 });
  assert.equal(event(s, 'humidity').intervention, 'I4');
  assert.equal(event(s, 'humidity').risk, 'L2');
});
await test('offline input retains smoke alarm and separately raises a device event', () => {
  let s = sample(initialState(), 0, { smoke: 'alarm' });
  s = sample(s, 1000, { online: false });
  assert.equal(event(s, 'smoke').risk, 'L4');
  assert.equal(event(s, 'smoke').signal, 'unknown');
  assert.ok(event(s, 'device'));
  assert.equal(activeEvents(s)[0].kind, 'smoke');
});
await test('stale input cannot clear events, including a close command', () => {
  let s = sample(initialState(), 0, { smoke: 'alarm' });
  s = sample(s, 1000);
  const id = event(s, 'smoke').id;
  s = careReducer(s, { type: 'close', id, now: 20000 });
  assert.equal(event(s, 'smoke').signal, 'unknown');
  assert.match(s.notice, /不能结束/);
});
await test('stable normal input enables closure; recurring alarm gets a new event', () => {
  let s = sample(initialState(), 0, { smoke: 'alarm' });
  const id = event(s, 'smoke').id;
  s = advance(s, 1000, 6000, {});
  assert.ok(canClose(event(s, 'smoke'), s, 6000));
  s = careReducer(s, { type: 'close', id, now: 6000 });
  assert.equal(activeEvents(s).length, 0);
  s = sample(s, 7000, { smoke: 'alarm' });
  assert.notEqual(event(s, 'smoke').id, id);
  assert.equal(s.events.length, 2);
});
await test('cannot close active alarm or close by marking it a false alarm', () => {
  let s = sample(initialState(), 0, { gas: 'alarm' });
  const id = event(s, 'gas').id;
  s = careReducer(s, { type: 'false-alarm', id, note: '测试误报', now: 0 });
  s = careReducer(s, { type: 'close', id, now: 0 });
  assert.equal(event(s, 'gas').falseAlarmNote, '测试误报');
  assert.equal(event(s, 'gas').phase, 'ESCALATED');
});
await test('ending one event does not clear concurrent smoke alarm', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32, smoke: 'alarm' });
  const id = event(s, 'temperature').id;
  s = advance(s, 6000, 11000, { smoke: 'alarm' });
  s = careReducer(s, { type: 'close', id, now: 11000 });
  assert.equal(activeEvents(s).length, 1);
  assert.equal(event(s, 'smoke').risk, 'L4');
});
await test('invalid and out-of-range readings are unknown, not normal', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  s = sample(s, 1000, { temperature: NaN, humidity: 150 });
  assert.equal(event(s, 'temperature').signal, 'unknown');
  assert.ok(event(s, 'device'));
});
await test('interruption resets confirmation; recovery uses hysteresis and restarts after loss', () => {
  let s = advance(initialState(), 0, 3000, { temperature: 32 });
  s = sample(s, 4000, { online: false });
  s = sample(s, 5000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I0');
  s = advance(s, 6000, 10000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I1');
  s = sample(s, 11000, { temperature: 27.5 });
  assert.equal(event(s, 'temperature').signal, 'abnormal');
  s = sample(s, 12000, { temperature: 24 });
  s = sample(s, 13000, { online: false });
  s = sample(s, 14000, { temperature: 32 });
  assert.equal(event(s, 'temperature').phase, 'INTERVENING');
});
await test('reducer does not mutate earlier state or accept backwards/invalid time', () => {
  const s = sample(initialState(), 1000, { smoke: 'alarm' });
  const before = JSON.stringify(s);
  careReducer(s, { type: 'acknowledge', id: event(s, 'smoke').id, now: 1000 });
  assert.equal(JSON.stringify(s), before);
  assert.equal(sample(s, 0), s);
  assert.equal(sample(s, NaN), s);
});

await test('requesting help during recovery must not trap the event outside recovery', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  s = sample(s, 1000);
  const id = event(s, 'temperature').id;
  s = careReducer(s, { type: 'feedback', id, feedback: 'help', now: 1000 });
  assert.equal(event(s, 'temperature').intervention, 'I4');
  s = advance(s, 2000, 6000, {});
  assert.ok(canClose(event(s, 'temperature'), s, 6000));
});

await test('late samples cannot count an unobserved gap as continuous abnormal evidence', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  s = sample(s, 30000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I0');
  assert.equal(event(s, 'temperature').abnormalSince, 30000);
  s = advance(s, 31000, 35000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I1');
});

await test('profile changes only affect new events, not active ones', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  s = careReducer(s, { type: 'profile', profile: 'visual', now: 0 });
  s = advance(s, 1000, 6000, { temperature: 32, humidity: 80 });
  assert.equal(event(s, 'temperature').intervention, 'I1');
  assert.equal(event(s, 'humidity').intervention, 'I2');
});

function deliver(
  s: CareState,
  id: string,
  now: number,
  status: 'executed' | 'failed' = 'executed',
) {
  const a = currentOutput(
    s,
    s.events.find((e) => e.id === id)!,
  )!;
  return careReducer(s, { type: 'output-result', actionId: a.id, status, now });
}
const vision = (
  at: number,
  fall = false,
  patch: Partial<VisionEvidence> = {},
): VisionEvidence => ({
  sessionId: 'CAM-test',
  frameId: at + 1,
  capturedAt: at,
  model: 'unit-fixture',
  device: 'cpu',
  persons: [
    {
      id: 1,
      reliable: true,
      fallEvidence: fall ? '测试夹具：站立、快速下降、持续横卧' : null,
    },
  ],
  ...patch,
});
const camera = (s: CareState, v: VisionEvidence | null, now: number) =>
  careReducer(s, {
    type: 'camera',
    vision: v,
    observations: [],
    now,
  });

await test('feedback timer starts at presentation receipt, not at decision time', () => {
  let s = advance(initialState(), 0, 30000, { temperature: 32 });
  const e = event(s, 'temperature');
  assert.equal(e.intervention, 'I1');
  assert.equal(e.phase, 'INTERVENING');
  assert.equal(s.actions.length, 1);
  assert.equal(s.actions[0].status, 'requested');
  s = deliver(s, e.id, 30000);
  assert.equal(event(s, 'temperature').phase, 'WAITING_RESPONSE');
  s = advance(s, 31000, 49000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I1');
  s = sample(s, 50000, { temperature: 32 });
  assert.equal(event(s, 'temperature').intervention, 'I2');
  assert.equal(s.actions[0].status, 'requested');
});

await test('output failure requests web help without raising risk or fabricating a response', () => {
  let s = advance(initialState(), 0, 5000, { humidity: 80 });
  s = deliver(s, event(s, 'humidity').id, 5000, 'failed');
  assert.equal(event(s, 'humidity').intervention, 'I4');
  assert.equal(event(s, 'humidity').risk, 'L2');
  assert.equal(event(s, 'humidity').feedback, 'none');
  assert.equal(s.actions[1].status, 'failed');
  assert.equal(s.actions[0].status, 'requested');
});

await test('repeated presentation receipts cannot execute again or restart feedback window', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  s = deliver(s, event(s, 'temperature').id, 5000);
  const before = JSON.stringify(s);
  const repeated = deliver(s, event(s, 'temperature').id, 5100);
  assert.equal(repeated.actions.length, 1);
  assert.equal(repeated.actions[0].completed_at, 5000);
  assert.equal(repeated.logs.length, s.logs.length);
  assert.equal(JSON.stringify(s), before);
});

await test('closed events cancel pending output requests and ignore late execution callbacks', () => {
  let s = sample(initialState(), 0, { gas: 'alarm' });
  const id = event(s, 'gas').id;
  const actionId = s.actions[0].id;
  s = advance(s, 1000, 6000, {});
  s = careReducer(s, { type: 'close', id, now: 6000 });
  s = careReducer(s, {
    type: 'output-result',
    actionId,
    status: 'executed',
    now: 6100,
  });
  assert.equal(s.actions[0].status, 'unavailable');
  assert.equal(s.events[0].phase, 'CLOSED');
});

await test('unavailable phone and simulated protection never claim external execution or downgrade I4', () => {
  let s = sample(initialState(), 0, { smoke: 'alarm' });
  const id = event(s, 'smoke').id;
  for (const channel of ['protection', 'phone'] as const) {
    s = careReducer(s, { type: 'request-output', id, channel, now: 0 });
    s = careReducer(s, { type: 'request-output', id, channel, now: 0 });
  }
  assert.equal(s.actions.length, 3);
  assert.equal(
    s.actions.find((a) => a.channel === 'phone')!.status,
    'unavailable',
  );
  assert.equal(
    s.actions.find((a) => a.channel === 'protection')!.status,
    'simulated',
  );
  assert.equal(event(s, 'smoke').intervention, 'I4');
  s = careReducer(s, {
    type: 'output-result',
    actionId: s.actions[0].id,
    status: 'executed',
    now: 0,
  });
  assert.equal(s.actions[0].status, 'unavailable');
});

await test('low risk or refusal cannot authorize protection even via a direct command', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  const id = event(s, 'temperature').id;
  s = careReducer(s, { type: 'feedback', id, feedback: 'declined', now: 5000 });
  s = careReducer(s, {
    type: 'request-output',
    id,
    channel: 'protection',
    now: 5000,
  });
  assert.match(s.notice, /仅 L3/);
  assert.equal(
    s.actions.some((a) => a.action_type === 'I3'),
    false,
  );
});

await test('fall candidate bypasses reminders, deduplicates frames, and remains latched after posture recovery', () => {
  let s = camera(initialState(), vision(0, true), 0);
  assert.equal(event(s, 'fall_candidate').risk, 'L3');
  assert.equal(event(s, 'fall_candidate').intervention, 'I4');
  s = camera(s, vision(0, true), 100);
  s = camera(s, vision(1000, true), 1000);
  assert.equal(s.events.length, 1);
  assert.equal(s.actions.length, 1);
  s = camera(s, vision(2000), 2000);
  const id = event(s, 'fall_candidate').id;
  s = careReducer(s, { type: 'acknowledge', id, now: 2000 });
  s = careReducer(s, {
    type: 'feedback',
    id,
    feedback: 'responded',
    now: 2000,
  });
  s = careReducer(s, { type: 'false-alarm', id, note: '仅为标注', now: 2000 });
  s = careReducer(s, { type: 'close', id, now: 2000 });
  assert.equal(event(s, 'fall_candidate').phase, 'ESCALATED');
  assert.equal(canClose(event(s, 'fall_candidate'), s, 2000), false);
  s = camera(s, null, 2100);
  assert.equal(event(s, 'fall_candidate').signal, 'unknown');
  assert.equal(s.overview.code, 'HIGH_RISK');
});

await test('fall closure requires actual review text, held candidates do not immediately reopen', () => {
  let s = camera(initialState(), vision(0, true), 0);
  const id = event(s, 'fall_candidate').id;
  s = careReducer(s, { type: 'review-close', id, note: '  ', now: 0 });
  assert.equal(activeEvents(s).length, 1);
  s = careReducer(s, {
    type: 'review-close',
    id,
    note: '测试：已现场核查并完成处置',
    now: 0,
  });
  assert.equal(activeEvents(s).length, 0);
  assert.ok(s.events[0].reviewNote);
  s = camera(s, vision(1000, true), 1000);
  assert.equal(activeEvents(s).length, 0);
  s = camera(s, vision(2000), 2000);
  s = camera(s, vision(3000, true), 3000);
  assert.equal(activeEvents(s).length, 1);
  assert.notEqual(event(s, 'fall_candidate').id, id);
});

await test('delayed fall is retained as historical unknown; future, unreliable and reordered evidence is rejected', () => {
  let s = camera(initialState(), vision(0, true), 5000);
  assert.equal(event(s, 'fall_candidate').signal, 'unknown');
  assert.match(event(s, 'fall_candidate').evidence, /延迟返回/);
  s = camera(s, vision(1000, true, { sessionId: 'CAM-new' }), 5000);
  assert.equal(s.events.length, 2);
  s = camera(s, vision(0, true, { sessionId: 'CAM-new', frameId: 1 }), 5000);
  assert.equal(s.vision!.frameId, 1001);
  assert.equal(camera(initialState(), vision(10000, true), 0).events.length, 0);
  assert.equal(
    camera(
      initialState(),
      vision(0, true, {
        persons: [{ id: 1, reliable: false, fallEvidence: '不可靠' }],
      }),
      0,
    ).events.length,
    0,
  );
});

await test('smoke stays highest priority with simultaneous falls and remains after manual fall closure', () => {
  let s = sample(initialState(), 0, { smoke: 'alarm' });
  s = camera(s, vision(0, true), 0);
  assert.equal(activeEvents(s)[0].kind, 'smoke');
  assert.equal(s.overview.code, 'EMERGENCY');
  s = careReducer(s, {
    type: 'review-close',
    id: event(s, 'fall_candidate').id,
    note: '测试核查',
    now: 0,
  });
  s = sample(s, 1000, { online: false });
  assert.equal(s.overview.code, 'EMERGENCY');
  assert.equal(event(s, 'smoke').signal, 'unknown');
});

await test('SAFE requires complete current coverage; missing camera/target/configuration is never L0', () => {
  let s = sample(initialState(), 0);
  assert.equal(s.overview.code, 'UNKNOWN');
  s = camera(s, vision(0), 0);
  assert.equal(s.overview.code, 'MONITORING');
  s = careReducer(s, {
    type: 'camera',
    observations: [],
    coverage: {
      targetId: 1,
      regionsArmed: true,
      regionsKnown: true,
      observingDwell: false,
    },
    now: 0,
  });
  assert.equal(s.overview.code, 'SAFE');
  assert.equal(s.overview.risk, 'L0');
  assert.equal(s.overview.observation, 'VALID');
  s = camera(s, vision(1000, false, { persons: [] }), 1000);
  assert.equal(s.overview.code, 'UNKNOWN');
  assert.notEqual(s.overview.risk, 'L0');
  s = careReducer(s, { type: 'tick', now: 5000 });
  assert.equal(s.overview.observation, 'UNKNOWN');
});

await test('overview derives monitoring, support and recovery without conflating intervention and risk', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  assert.equal(s.overview.code, 'MONITORING');
  s = advance(s, 1000, 5000, { temperature: 32 });
  const id = event(s, 'temperature').id;
  assert.equal(s.overview.code, 'MEDIUM_RISK');
  s = careReducer(s, { type: 'feedback', id, feedback: 'help', now: 5000 });
  assert.equal(s.overview.code, 'MEDIUM_RISK');
  s = advance(s, 6000, 11000, {});
  assert.equal(s.overview.code, 'RECOVERING');
  s = careReducer(s, { type: 'close', id, now: 11000 });
  assert.equal(s.overview.code, 'UNKNOWN');
  assert.ok(
    s.transitions.some(
      (t) => t.from === 'MEDIUM_RISK' && t.code === 'RECOVERING',
    ),
  );
});

await test('session report contains decision/evidence/output history without frames or backend tokens', () => {
  let s = sample(initialState(), 0, { gas: 'alarm' });
  s = camera(s, vision(0, true), 0);
  const gasId = event(s, 'gas').id;
  s = deliver(s, gasId, 0);
  s = careReducer(s, {
    type: 'feedback',
    id: gasId,
    feedback: 'help',
    requestKey: currentOutput(s, event(s, 'gas'))!.requestKey,
    now: 0,
  });
  const zones = {
    ...initialZoneState(),
    observation: { privateToken: 'DO_NOT_EXPORT', pixels: 'DO_NOT_EXPORT' },
  };
  // Runtime extras must not be exported, even if a future UI adds them to state.
  const report = buildSessionReport(
    { ...s, extraImage: 'DO_NOT_EXPORT' } as CareState,
    zones as unknown as ReturnType<typeof initialZoneState>,
  );
  assert.equal(report.formatVersion, 5);
  assert.equal(report.events.length, 2);
  assert.equal(
    report.actions.some((a) => a.status === 'executed'),
    true,
  );
  assert.equal(report.boundaries.imagesIncluded, false);
  assert.equal(report.boundaries.rawManualTextIncluded, false);
  assert.match(report.boundaries.feedback, /caregiver-entered/);
  assert.equal(
    report.events.find((item) => item.id === gasId)!.feedbackHistory.length,
    1,
  );
  assert.equal(JSON.stringify(report).includes('DO_NOT_EXPORT'), false);
});

const aiDecision = (
  patch: Partial<AIDecisionPayload> = {},
): AIDecisionPayload => ({
  model: 'fixture-v31',
  runName: 'fixture-run',
  device: 'cpu',
  inferenceMs: 12.5,
  risk: 'L2',
  intervention: 'I2',
  alertMode: 'PAGE_WARNING',
  manualReview: false,
  abstain: false,
  riskConfidence: 0.9,
  interventionConfidence: 0.88,
  alertConfidence: 0.92,
  reviewReasons: [],
  ...patch,
});

await test('early AI output cannot bypass the later capability-based rule floor', () => {
  let s = careReducer(initialState(), {
    type: 'profile',
    profile: 'visual',
    now: 0,
  });
  s = sample(s, 0, { temperature: 32 });
  const id = event(s, 'temperature').id;
  const key = decisionRequestKey(event(s, 'temperature'));
  s = careReducer(s, { type: 'ai-start', id, requestKey: key, now: 1 });
  s = careReducer(s, {
    type: 'ai-result',
    id,
    requestKey: key,
    decision: aiDecision({ risk: 'L2', intervention: 'I1' }),
    now: 2,
  });
  assert.equal(event(s, 'temperature').intervention, 'I2');
  s = advance(s, 1000, 5000, { temperature: 32 });
  assert.deepEqual(event(s, 'temperature').ruleDecision, {
    risk: 'L2',
    intervention: 'I2',
    reason: '档案预设需要图示支持。',
  });
  assert.equal(event(s, 'temperature').intervention, 'I2');
  assert.equal(event(s, 'temperature').risk, 'L2');
});

await test('feedback revisions archive stale AI results and queue only the new context', () => {
  let s = advance(initialState(), 0, 5000, { temperature: 32 });
  const id = event(s, 'temperature').id;
  s = deliver(s, id, 5000);
  const responseKey = currentOutput(s, event(s, 'temperature'))!.requestKey;
  const oldAIKey = decisionRequestKey(event(s, 'temperature'));
  s = careReducer(s, {
    type: 'ai-start',
    id,
    requestKey: oldAIKey,
    now: 5001,
  });
  s = careReducer(s, {
    type: 'feedback',
    id,
    feedback: 'responded',
    requestKey: responseKey,
    now: 5002,
  });
  assert.equal(event(s, 'temperature').context.revision, 1);
  const newAIKey = decisionRequestKey(event(s, 'temperature'));
  assert.notEqual(newAIKey, oldAIKey);
  assert.doesNotMatch(buildDecisionText(event(s, 'temperature').context), /I1/);
  assert.match(
    buildDecisionText(event(s, 'temperature').context),
    /温和提醒.*当前回应和现场证据/,
  );
  s = careReducer(s, {
    type: 'ai-result',
    id,
    requestKey: oldAIKey,
    decision: aiDecision({ risk: 'L4', intervention: 'I4' }),
    now: 5003,
  });
  assert.equal(event(s, 'temperature').aiDecision, null);
  assert.equal(event(s, 'temperature').aiDecisionHistory.length, 1);
  assert.equal(event(s, 'temperature').aiDecisionHistory[0].applied, false);
  assert.equal(event(s, 'temperature').risk, 'L2');
  assert.equal(event(s, 'temperature').intervention, 'I1');

  s = careReducer(s, {
    type: 'ai-start',
    id,
    requestKey: newAIKey,
    now: 5004,
  });
  s = careReducer(s, {
    type: 'ai-result',
    id,
    requestKey: newAIKey,
    decision: aiDecision({ risk: 'L2', intervention: 'I1' }),
    now: 5005,
  });
  assert.equal(event(s, 'temperature').aiDecision?.status, 'completed');
  assert.equal(event(s, 'temperature').aiDecision?.requestKey, newAIKey);
});

await test('all automatic sources carry one normalized AI context without rule-label leakage', () => {
  const s = sample(initialState(), 0, { temperature: 32 });
  const e = event(s, 'temperature');
  assert.equal(e.context.schemaVersion, 2);
  assert.equal(e.context.revision, 0);
  assert.equal(e.context.eventId, e.id);
  assert.equal(e.context.source, 'simulated');
  assert.equal(e.context.scene, '客厅');
  assert.equal(e.aiDecision, null);
  assert.deepEqual(e.ruleDecision, {
    risk: 'L1',
    intervention: 'I0',
    reason: '异常刚出现，先连续确认。',
  });
  const text = buildDecisionText(e.context);
  assert.match(text, /地点：客厅/);
  assert.match(text, /模拟传感器/);
  assert.doesNotMatch(text, /规则基线|L1|I0/);
  assert.equal(decisionRequestKey(e), `${e.id}:2:0:0`);
});

await test('AI may raise an ordinary event but can never lower the rule safety floor', () => {
  let s = sample(initialState(), 0, { temperature: 32 });
  let e = event(s, 'temperature');
  const key = decisionRequestKey(e);
  s = careReducer(s, { type: 'ai-start', id: e.id, requestKey: key, now: 1 });
  s = careReducer(s, {
    type: 'ai-result',
    id: e.id,
    requestKey: key,
    decision: aiDecision({ risk: 'L3', intervention: 'I3' }),
    now: 2,
  });
  e = event(s, 'temperature');
  assert.equal(e.risk, 'L3');
  assert.equal(e.intervention, 'I3');
  assert.equal(e.aiDecision?.status, 'completed');
  assert.equal(e.aiDecision?.applied, true);

  let urgentState = sample(initialState(), 0, { smoke: 'alarm' });
  const urgentEvent = event(urgentState, 'smoke');
  const urgentKey = decisionRequestKey(urgentEvent);
  urgentState = careReducer(urgentState, {
    type: 'ai-start',
    id: urgentEvent.id,
    requestKey: urgentKey,
    now: 1,
  });
  urgentState = careReducer(urgentState, {
    type: 'ai-result',
    id: urgentEvent.id,
    requestKey: urgentKey,
    decision: aiDecision({
      risk: 'L0',
      intervention: 'I0',
      alertMode: 'NONE',
    }),
    now: 2,
  });
  assert.equal(event(urgentState, 'smoke').risk, 'L4');
  assert.equal(event(urgentState, 'smoke').intervention, 'I4');
});

await test('abstention, stale keys and AI failure preserve the deterministic event', () => {
  let s = sample(initialState(), 0, { humidity: 80 });
  const e = event(s, 'humidity');
  const key = decisionRequestKey(e);
  s = careReducer(s, { type: 'ai-start', id: e.id, requestKey: key, now: 1 });
  const stale = careReducer(s, {
    type: 'ai-result',
    id: e.id,
    requestKey: 'stale-key',
    decision: aiDecision({ risk: 'L4', intervention: 'I4' }),
    now: 2,
  });
  assert.equal(event(stale, 'humidity').aiDecision?.status, 'running');
  s = careReducer(stale, {
    type: 'ai-result',
    id: e.id,
    requestKey: key,
    decision: aiDecision({
      risk: 'L4',
      intervention: 'I4',
      abstain: true,
      manualReview: true,
    }),
    now: 3,
  });
  assert.equal(event(s, 'humidity').risk, 'L1');
  assert.equal(event(s, 'humidity').intervention, 'I0');
  assert.equal(event(s, 'humidity').aiDecision?.applied, false);

  let failed = sample(initialState(), 0, { temperature: 32 });
  const failedEvent = event(failed, 'temperature');
  const failedKey = decisionRequestKey(failedEvent);
  failed = careReducer(failed, {
    type: 'ai-start',
    id: failedEvent.id,
    requestKey: failedKey,
    now: 1,
  });
  failed = careReducer(failed, {
    type: 'ai-failed',
    id: failedEvent.id,
    requestKey: failedKey,
    error: '模型缺失',
    now: 2,
  });
  assert.equal(event(failed, 'temperature').risk, 'L1');
  assert.equal(event(failed, 'temperature').aiDecision?.status, 'failed');
  failed = careReducer(failed, {
    type: 'ai-retry',
    id: failedEvent.id,
    now: 3,
  });
  assert.equal(event(failed, 'temperature').aiDecision, null);
});

await test('manual model results create reviewable events while L0/I0 creates no alert', () => {
  const text = '地点：阳台。\n事件1（人工输入）：患者正在倚靠松动栏杆。';
  let s = careReducer(initialState(), {
    type: 'manual-ai-event',
    text,
    decision: aiDecision({ risk: 'L3', intervention: 'I3' }),
    now: 1,
  });
  assert.equal(activeEvents(s).length, 1);
  const e = activeEvents(s)[0];
  assert.equal(e.source, 'manual');
  assert.equal(e.context.scene, '阳台');
  assert.equal(e.ruleDecision, null);
  assert.equal(e.aiDecision?.applied, true);
  assert.equal(canClose(e, s, 1), false);
  s = careReducer(s, {
    type: 'review-close',
    id: e.id,
    note: '已现场核查并移除风险物品',
    now: 2,
  });
  assert.equal(activeEvents(s).length, 0);
  const report = buildSessionReport(s, initialZoneState());
  assert.equal(report.events[0].context.rawText, null);
  assert.equal(JSON.stringify(report).includes('松动栏杆'), true);

  const observation = careReducer(initialState(), {
    type: 'manual-ai-event',
    text: '地点：客厅。事件1：患者坐在沙发上阅读。',
    decision: aiDecision({ risk: 'L0', intervention: 'I0', alertMode: 'NONE' }),
    now: 1,
  });
  assert.equal(observation.events.length, 0);
  assert.match(observation.logs[0].message, /不创建待处理事件/);
});
