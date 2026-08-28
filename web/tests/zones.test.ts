import test from 'node:test';
import assert from 'node:assert/strict';
import {
  freshObservation,
  initialZoneState,
  normalizeRect,
  positionInZone,
  validRect,
  zoneEvidence,
  zoneReducer,
} from '../lib/zones.ts';
import type { CameraObservation, ZoneState } from '../lib/zones.ts';
import {
  activeEvents,
  canClose,
  careReducer,
  initialState,
  INITIAL_INPUT,
} from '../lib/care/engine.ts';
import type { CareState } from '../lib/care/types.ts';
import { cameraAction } from '../lib/care/camera.ts';

type Options = {
  x?: number;
  y?: number;
  id?: number;
  missing?: boolean;
  confidence?: number;
  session?: string;
  width?: number;
};
function observation(at: number, options: Options = {}): CameraObservation {
  return {
    sessionId: options.session ?? 'CAM-test',
    capturedAt: at,
    frame: {
      schema_version: 1,
      source: 'camera',
      model: 'unit-fixture',
      device: 'cpu',
      frame_id: at + 1,
      width: options.width ?? 640,
      height: 480,
      inference_ms: 100,
      persons: options.missing
        ? []
        : [
            {
              id: options.id ?? 1,
              bbox: [0.2, 0.1, 0.8, 0.98],
              confidence: 0.9,
              pose_reliable: true,
              actions: [],
              keypoints: Array.from({ length: 17 }, (_, i) => ({
                x: options.x ?? 0.5,
                y: i >= 15 ? (options.y ?? 0.75) : 0.4,
                confidence: options.confidence ?? 0.9,
              })),
            },
          ],
    },
  };
}
function frame(s: ZoneState, at: number, options: Options = {}, received = at) {
  return zoneReducer(s, {
    type: 'frame',
    observation: observation(at, options),
    now: received,
  });
}
function setup() {
  let s = frame(initialZoneState(), 0);
  s = zoneReducer(s, { type: 'select', personId: 1, now: 0 });
  s = zoneReducer(s, {
    type: 'add',
    name: '灶台前地面',
    rect: [0.25, 0.5, 0.75, 0.95],
    dwellMs: 5000,
    now: 0,
  });
  return zoneReducer(s, { type: 'arm', enabled: true, now: 0 });
}
function advance(
  s: ZoneState,
  start: number,
  end: number,
  options: Options = {},
) {
  for (let at = start; at <= end; at += 1000) s = frame(s, at, options);
  return s;
}
function sync(c: CareState, z: ZoneState, now: number) {
  return careReducer(c, cameraAction(z, now));
}

await test('vision adapter routes falls without requiring selected person or armed regions', () => {
  const o = observation(0);
  o.frame.persons[0].actions = [
    {
      code: 'fall_candidate',
      label: '测试候选',
      evidence: '测试：站立、下降、横卧',
    },
  ];
  let z = zoneReducer(initialZoneState(), {
    type: 'frame',
    observation: o,
    now: 0,
  });
  let c = sync(initialState(), z, 0);
  assert.equal(c.events.length, 1);
  assert.equal(c.events[0].kind, 'fall_candidate');
  assert.equal(c.events[0].source, 'camera');
  assert.equal(c.events[0].intervention, 'I4');
  assert.equal(z.armed, false);
  const payload = JSON.stringify(cameraAction(z, 0));
  assert.equal(payload.includes('keypoints'), false);
  assert.equal(payload.includes('bbox'), false);
  z = zoneReducer(z, { type: 'frame', observation: null, now: 1000 });
  c = sync(c, z, 1000);
  assert.equal(c.events[0].signal, 'unknown');
  assert.equal(c.events[0].phase, 'ESCALATED');
});

await test('selected-person dwell and another visible person fall remain separate in the shared pipeline', () => {
  let z = setup();
  let c = initialState();
  for (let at = 1000; at <= 6000; at += 1000) {
    z = frame(z, at);
    c = sync(c, z, at);
  }
  const o = observation(7000);
  o.frame.persons.push({
    ...o.frame.persons[0],
    id: 2,
    actions: [
      { code: 'fall_candidate', label: '测试候选', evidence: 'P2 测试候选' },
    ],
  });
  z = zoneReducer(z, { type: 'frame', observation: o, now: 7000 });
  c = sync(c, z, 7000);
  assert.equal(c.events.length, 2);
  assert.equal(activeEvents(c)[0].kind, 'fall_candidate');
  assert.match(
    c.events.find((e) => e.kind === 'fall_candidate')!.target!,
    /P2/,
  );
  assert.match(c.events.find((e) => e.kind === 'zone_dwell')!.target!, /P1/);
  assert.equal(c.overview.code, 'HIGH_RISK');
});

await test('rectangles normalize either drag direction and reject invalid or tiny regions', () => {
  assert.deepEqual(normalizeRect([0.8, 0.9], [0.2, 0.5]), [0.2, 0.5, 0.8, 0.9]);
  assert.ok(validRect([0.2, 0.5, 0.8, 0.9]));
  assert.ok(!validRect([0.2, 0.5, 0.21, 0.9]));
  assert.ok(!validRect([NaN, 0, 1, 1]));
});
await test('boundary hysteresis preserves an inside point but requires evidence to enter', () => {
  const rect: [number, number, number, number] = [0.25, 0.5, 0.75, 0.95];
  assert.equal(positionInZone([0.25, 0.7], rect, null), 'unknown');
  assert.equal(positionInZone([0.245, 0.7], rect, 'inside'), 'inside');
  assert.equal(positionInZone([0.2, 0.7], rect, 'inside'), 'outside');
});
await test('monitoring requires a selected current person and calibrated region', () => {
  let s = frame(initialZoneState(), 0);
  s = zoneReducer(s, { type: 'arm', enabled: true, now: 0 });
  assert.equal(s.armed, false);
  s = zoneReducer(s, { type: 'select', personId: 99, now: 0 });
  assert.equal(s.selected, null);
  assert.ok(setup().armed);
});
await test('entry uses confirmation and dwell starts at first new frame, not arm time', () => {
  let s = setup();
  s = frame(s, 1000);
  assert.equal(s.statuses[0].position, 'unknown');
  s = frame(s, 2000);
  assert.equal(s.statuses[0].position, 'inside');
  assert.equal(s.statuses[0].dwellMs, 1000);
  s = advance(s, 3000, 5000);
  assert.equal(zoneEvidence(s, 5000)[0].triggered, false);
  s = frame(s, 6000);
  assert.equal(zoneEvidence(s, 6000)[0].triggered, true);
});
await test('wall clock ticks never fabricate dwell and stale input resets continuity', () => {
  let s = advance(setup(), 1000, 2000);
  s = zoneReducer(s, { type: 'tick', now: 4000 });
  assert.equal(s.statuses[0].dwellMs, 1000);
  s = zoneReducer(s, { type: 'tick', now: 6000 });
  assert.equal(s.statuses[0].position, 'unknown');
  assert.equal(s.statuses[0].dwellMs, 0);
  s = frame(s, 7000);
  assert.equal(s.statuses[0].position, 'unknown');
});
await test('capture times drive dwell; delayed response time does not count toward it', () => {
  let s = setup();
  s = frame(s, 1000, {}, 3000);
  s = frame(s, 2000, {}, 4000);
  assert.equal(s.statuses[0].dwellMs, 1000);
  s = frame(s, 3000, {}, 7000);
  assert.equal(s.statuses[0].position, 'unknown');
  assert.equal(zoneEvidence(s, 7000)[0].triggered, false);
  assert.ok(!freshObservation(observation(8000), 7000));
});
await test('missing target and low confidence ankles never count as leaving', () => {
  let s = advance(setup(), 1000, 6000);
  s = frame(s, 7000, { id: 2 });
  assert.equal(s.selected?.personId, 1);
  assert.equal(zoneEvidence(s, 7000)[0].signal, 'unknown');
  s = frame(s, 8000, { confidence: 0.1 });
  assert.equal(zoneEvidence(s, 8000)[0].signal, 'unknown');
  assert.equal(s.statuses[0].dwellMs, 0);
});
await test('only confirmed outside observations report departure', () => {
  let s = advance(setup(), 1000, 6000);
  s = frame(s, 7000, { x: 0.1 });
  assert.equal(zoneEvidence(s, 7000)[0].signal, 'unknown');
  s = frame(s, 8000, { x: 0.1 });
  assert.equal(zoneEvidence(s, 8000)[0].signal, 'normal');
  assert.match(s.logs[0].message, /离开区域/);
});
await test('new sessions and changed frame dimensions invalidate calibration and target', () => {
  for (const options of [{ session: 'CAM-new' }, { width: 320 }]) {
    const s = frame(advance(setup(), 1000, 2000), 3000, options);
    assert.equal(s.armed, false);
    assert.equal(s.calibrated, null);
    assert.equal(s.selected, null);
    assert.equal(s.zones.length, 1);
  }
});
await test('edits are blocked while armed; pause and deletion preserve event keys only as unknown', () => {
  let s = advance(setup(), 1000, 6000);
  const id = s.zones[0].id;
  s = zoneReducer(s, { type: 'delete', id, now: 6000 });
  assert.equal(s.zones.length, 1);
  s = zoneReducer(s, { type: 'arm', enabled: false, now: 6000 });
  assert.equal(zoneEvidence(s, 6000)[0].signal, 'unknown');
  s = zoneReducer(s, { type: 'delete', id, now: 6000 });
  assert.equal(s.zones.length, 0);
});
await test('selection changes reset dwell and are scoped to the new manually selected person', () => {
  let s = advance(setup(), 1000, 6000);
  const key = zoneEvidence(s, 6000)[0].key;
  s = frame(s, 7000, { id: 2 });
  s = zoneReducer(s, { type: 'select', personId: 2, now: 7000 });
  assert.notEqual(zoneEvidence(s, 7000)[0].key, key);
  assert.equal(zoneEvidence(s, 7000)[0].signal, 'unknown');
});
await test('duplicate/out-of-order frames and backwards time cannot change prior state', () => {
  const s = advance(setup(), 1000, 2000);
  const json = JSON.stringify(s);
  assert.equal(frame(s, 2000), s);
  assert.equal(frame(s, 1000), s);
  frame(s, 3000, { x: 0.1 });
  assert.equal(JSON.stringify(s), json);
});
await test('complete camera chain creates one L2 event with correct source and profile', () => {
  let z = setup();
  let c = careReducer(initialState(), {
    type: 'profile',
    profile: 'visual',
    now: 0,
  });
  for (let at = 1000; at <= 10000; at += 1000) {
    z = frame(z, at);
    c = sync(c, z, at);
  }
  assert.equal(c.events.length, 1);
  assert.equal(c.events[0].source, 'camera');
  assert.equal(c.events[0].risk, 'L2');
  assert.equal(c.events[0].intervention, 'I2');
});
await test('sustained camera evidence escalates support without raising risk', () => {
  let z = setup();
  let c = initialState();
  for (let at = 1000; at <= 47000; at += 1000) {
    z = frame(z, at);
    c = sync(c, z, at);
    for (const a of c.actions.filter(
      (a) => a.status === 'requested' && a.channel === 'web',
    ))
      c = careReducer(c, {
        type: 'output-result',
        actionId: a.id,
        status: 'executed',
        now: at,
      });
  }
  assert.equal(c.events[0].intervention, 'I4');
  assert.equal(c.events[0].risk, 'L2');
});
await test('camera event recovers from same-target exit even if environment input is offline', () => {
  let z = setup();
  let c = initialState();
  for (let at = 1000; at <= 6000; at += 1000) {
    z = frame(z, at);
    c = sync(c, z, at);
  }
  const id = c.events[0].id;
  c = careReducer(c, {
    type: 'sample',
    input: { ...INITIAL_INPUT, online: false },
    now: 6000,
  });
  for (let at = 7000; at <= 13000; at += 1000) {
    z = frame(z, at, { x: 0.1 });
    c = sync(c, z, at);
  }
  const e = c.events.find((e) => e.id === id)!;
  assert.ok(canClose(e, c, 13000));
  c = careReducer(c, { type: 'close', id, now: 13000 });
  assert.equal(c.events.find((e) => e.id === id)!.phase, 'CLOSED');
  assert.ok(activeEvents(c).some((e) => e.kind === 'device'));
});
await test('stopping camera or selecting another person cannot clear the old event', () => {
  let z = advance(setup(), 1000, 6000);
  let c = sync(initialState(), z, 6000);
  const id = c.events[0].id;
  z = frame(z, 7000, { id: 2 });
  z = zoneReducer(z, { type: 'select', personId: 2, now: 7000 });
  c = sync(c, z, 7000);
  assert.equal(c.events[0].signal, 'unknown');
  assert.ok(!canClose(c.events[0], c, 7000));
  z = zoneReducer(z, { type: 'frame', observation: null, now: 8000 });
  c = sync(c, z, 8000);
  c = careReducer(c, { type: 'close', id, now: 14000 });
  assert.notEqual(c.events[0].phase, 'CLOSED');
});
await test('active camera event stays abnormal if observation resumes inside below new dwell threshold', () => {
  let z = advance(setup(), 1000, 6000);
  let c = sync(initialState(), z, 6000);
  z = frame(z, 7000, { confidence: 0.1 });
  c = sync(c, z, 7000);
  assert.equal(c.events[0].signal, 'unknown');
  z = advance(z, 8000, 9000);
  assert.equal(zoneEvidence(z, 9000)[0].triggered, false);
  c = sync(c, z, 9000);
  assert.equal(c.events[0].signal, 'abnormal');
  assert.equal(c.events[0].phase, 'INTERVENING');
});
await test('manual closure requires actual notes and unavailable camera evidence, not simulated alarms', () => {
  const z = advance(setup(), 1000, 6000);
  let c = sync(initialState(), z, 6000);
  const id = c.events[0].id;
  c = careReducer(c, { type: 'review-close', id, note: '已查看', now: 6000 });
  assert.notEqual(c.events[0].phase, 'CLOSED');
  c = careReducer(c, { type: 'camera', observations: [], now: 7000 });
  c = careReducer(c, { type: 'review-close', id, note: ' ', now: 7000 });
  assert.notEqual(c.events[0].phase, 'CLOSED');
  c = careReducer(c, {
    type: 'review-close',
    id,
    note: '照护者已到现场核查并处理。',
    now: 7000,
  });
  assert.equal(c.events[0].phase, 'CLOSED');
  assert.ok(c.events[0].reviewNote);
  c = careReducer(c, {
    type: 'sample',
    input: { ...INITIAL_INPUT, smoke: 'alarm' },
    now: 8000,
  });
  const smoke = activeEvents(c).find((e) => e.kind === 'smoke')!;
  c = careReducer(c, {
    type: 'review-close',
    id: smoke.id,
    note: '测试',
    now: 8000,
  });
  assert.equal(
    activeEvents(c).find((e) => e.kind === 'smoke')!.phase,
    'ESCALATED',
  );
});
await test('smoke retains emergency priority when a camera dwell event occurs concurrently', () => {
  const z = advance(setup(), 1000, 6000);
  let c = sync(initialState(), z, 6000);
  c = careReducer(c, {
    type: 'sample',
    input: { ...INITIAL_INPUT, smoke: 'alarm' },
    now: 6000,
  });
  assert.equal(activeEvents(c)[0].kind, 'smoke');
  assert.equal(activeEvents(c).length, 2);
});
await test('invalid capture timestamps never create camera events', () => {
  const z = advance(setup(), 1000, 6000);
  const observations = zoneEvidence(z, 6000).map((o) => ({
    ...o,
    capturedAt: NaN,
  }));
  const c = careReducer(initialState(), {
    type: 'camera',
    observations,
    now: 6000,
  });
  assert.equal(c.events.length, 0);
});

await test('recovery requires five seconds of captured outside evidence, not delayed delivery', () => {
  let z = advance(setup(), 1000, 6000);
  let c = sync(initialState(), z, 6000);
  z = frame(z, 7000, { x: 0.1 });
  c = sync(c, z, 7000);
  z = frame(z, 8000, { x: 0.1 });
  c = sync(c, z, 8000);
  z = frame(z, 9000, { x: 0.1 }, 11000);
  c = sync(c, z, 11000);
  z = frame(z, 10000, { x: 0.1 }, 12000);
  c = sync(c, z, 12000);
  c = careReducer(c, { type: 'tick', now: 13000 });
  assert.equal(c.events[0].signal, 'normal');
  assert.ok(!canClose(c.events[0], c, 13000));
  z = frame(z, 13000, { x: 0.1 });
  c = sync(c, z, 13000);
  assert.ok(canClose(c.events[0], c, 13000));
});

await test('acknowledging a camera event stops escalation without clearing the region risk', () => {
  let z = advance(setup(), 1000, 6000);
  let c = sync(initialState(), z, 6000);
  c = careReducer(c, { type: 'acknowledge', id: c.events[0].id, now: 6000 });
  for (let at = 7000; at <= 40000; at += 1000) {
    z = frame(z, at);
    c = sync(c, z, at);
  }
  assert.equal(c.events[0].intervention, 'I1');
  assert.equal(c.events[0].signal, 'abnormal');
  assert.ok(!canClose(c.events[0], c, 40000));
});
