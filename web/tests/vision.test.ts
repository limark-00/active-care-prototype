import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraError, drawFrame, parseVisionFrame } from '../lib/vision.ts';
import {
  createFrameGate,
  displayRect,
  displayX,
  startFramePump,
} from '../lib/vision-runtime.ts';
import { positionInZone } from '../lib/zones.ts';

const fixture = () => ({
  schema_version: 1,
  source: 'camera',
  model: 'fixture',
  device: 'cpu',
  frame_id: 1,
  width: 640,
  height: 480,
  inference_ms: 100,
  persons: [
    {
      id: 1,
      bbox: [0.1, 0.1, 0.8, 0.9],
      confidence: 0.9,
      pose_reliable: true,
      keypoints: Array.from({ length: 17 }, () => ({
        x: 0.4,
        y: 0.5,
        confidence: 0.8,
      })),
      actions: [{ code: 'hands_up', label: '双手举起', evidence: '测试规则' }],
    },
  ],
});
await test('camera contract accepts actual structure and zero detections without inventing people', () => {
  assert.equal(
    parseVisionFrame(fixture(), 1).persons[0].actions[0].code,
    'hands_up',
  );
  assert.equal(
    parseVisionFrame({ ...fixture(), persons: [] }, 1).persons.length,
    0,
  );
});
await test('rejects stale frame ids, simulated sources, and invalid coordinates', () => {
  assert.throws(() => parseVisionFrame(fixture(), 2));
  assert.throws(() =>
    parseVisionFrame({ ...fixture(), source: 'simulated' }, 1),
  );
  const value = fixture();
  value.persons[0].keypoints[0].x = NaN;
  assert.throws(() => parseVisionFrame(value, 1));
});
await test('rejects incompatible actions, duplicate identities and malformed dimensions', () => {
  const value = fixture();
  value.persons[0].actions[0].code = 'diagnosed_fall';
  assert.throws(() => parseVisionFrame(value, 1));
  const duplicate = fixture();
  duplicate.persons.push(duplicate.persons[0]);
  assert.throws(() => parseVisionFrame(duplicate, 1));
  assert.throws(() => parseVisionFrame({ ...fixture(), width: 1 }, 1));
});
await test('permission denial and frame timeout produce explicit actionable errors', () => {
  assert.match(
    cameraError(new DOMException('denied', 'NotAllowedError')),
    /权限/,
  );
  assert.match(
    cameraError(new DOMException('timeout', 'AbortError')),
    /状态未知/,
  );
});

await test('flipping is reversible and preserves region inclusion in either direction', () => {
  const region: [number, number, number, number] = [0.1, 0.3, 0.4, 0.9];
  const flipped = displayRect(region, true);
  assert.deepEqual(flipped, [0.6, 0.3, 0.9, 0.9]);
  assert.ok(
    displayRect(flipped, true).every((v, i) => Math.abs(v - region[i]) < 1e-10),
  );
  assert.deepEqual(displayRect(region, false), region);
  assert.equal(displayX(0.25, true), 0.75);
  // Pointer and numeric input both convert displayed coordinates back to source.
  const sourceX = displayX(0.8, true);
  assert.equal(positionInZone([sourceX, 0.5], region, null), 'inside');
  assert.equal(positionInZone([0.8, 0.5], flipped, null), 'inside');
});

await test('canvas flips source pixels and geometry while keeping text readable and evidence immutable', () => {
  const f = parseVisionFrame(fixture(), 1);
  const before = JSON.stringify(f);
  const calls: { method: string; args: unknown[] }[] = [];
  const record =
    (method: string) =>
    (...args: unknown[]) =>
      calls.push({ method, args });
  const context = {
    clearRect: record('clear'),
    save: record('save'),
    restore: record('restore'),
    translate: record('translate'),
    scale: record('scale'),
    drawImage: record('draw'),
    strokeRect: record('box'),
    fillRect: record('label'),
    fillText: record('text'),
    beginPath: record('begin'),
    moveTo: record('move'),
    lineTo: record('line'),
    stroke: record('stroke'),
    arc: record('point'),
    fill: record('fill'),
    measureText: () => ({ width: 70 }),
  };
  const canvas = { width: 640, height: 480, getContext: () => context };
  drawFrame(
    canvas as unknown as HTMLCanvasElement,
    {} as HTMLCanvasElement,
    f,
    true,
    true,
  );
  assert.deepEqual(calls.find((c) => c.method === 'scale')!.args, [-1, 1]);
  assert.ok(
    calls.findIndex((c) => c.method === 'restore') <
      calls.findIndex((c) => c.method === 'text'),
  );
  assert.ok(
    Math.abs(Number(calls.find((c) => c.method === 'box')!.args[0]) - 128) <
      1e-10,
  );
  assert.equal(calls.find((c) => c.method === 'point')!.args[0], 384);
  assert.equal(JSON.stringify(f), before);
});

await test('latest-frame gate skips work while busy instead of processing a backlog', () => {
  const gate = createFrameGate();
  assert.equal(gate.begin(0, 0), true);
  assert.equal(gate.begin(0.1, 100), false);
  gate.finish();
  assert.equal(gate.begin(0.1, 150), false, 'skipped frame is never replayed');
  assert.equal(gate.begin(0.2, 200), true);
  gate.finish();
  assert.equal(gate.begin(0.21, 210), false, 'rate cap prevents CPU flooding');
  gate.stop();
  gate.finish();
  assert.equal(
    gate.begin(0.4, 400),
    false,
    'late completion cannot revive stopped capture',
  );
});

function fakeVideo() {
  let sequence = 0;
  const callbacks = new Map<number, VideoFrameRequestCallback>();
  const video = {
    currentTime: 0,
    readyState: 2,
    requestVideoFrameCallback(callback: VideoFrameRequestCallback) {
      const id = ++sequence;
      callbacks.set(id, callback);
      return id;
    },
    cancelVideoFrameCallback(id: number) {
      callbacks.delete(id);
    },
  };
  return {
    video,
    pending: () => callbacks.size,
    frame(mediaTime: number) {
      video.currentTime = mediaTime;
      const next = callbacks.entries().next().value;
      if (!next) return;
      callbacks.delete(next[0]);
      next[1](mediaTime * 1000, { mediaTime } as VideoFrameCallbackMetadata);
    },
  };
}
const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

await test('native frame callbacks stay serial and cancellation removes future callbacks', async () => {
  const source = fakeVideo();
  let at = 0,
    count = 0;
  let complete: () => void = () => {};
  const errors: unknown[] = [];
  const stop = startFramePump(
    source.video,
    () => {
      count += 1;
      return new Promise<void>((resolve) => {
        complete = resolve;
      });
    },
    (error) => errors.push(error),
    () => at,
  );
  try {
    source.frame(0);
    at = 100;
    source.frame(0.1);
    assert.equal(count, 1);
    complete();
    await settle();
    at = 200;
    source.frame(0.2);
    assert.equal(count, 2);
    stop();
    complete();
    await settle();
    at = 300;
    source.frame(0.3);
    assert.equal(count, 2);
    assert.equal(source.pending(), 0);
    assert.equal(errors.length, 0);
  } finally {
    stop();
  }
});

await test('unsupported-browser fallback checks media time rather than reanalyzing frozen video', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const video = { currentTime: 0, readyState: 2 };
  let at = 0,
    count = 0;
  const stop = startFramePump(
    video,
    async () => {
      count += 1;
    },
    () => {},
    () => at,
  );
  try {
    context.mock.timers.tick(16);
    await settle();
    at = 100;
    context.mock.timers.tick(100);
    await settle();
    assert.equal(count, 1);
    video.currentTime = 0.2;
    at = 200;
    context.mock.timers.tick(16);
    await settle();
    assert.equal(count, 2);
    stop();
    video.currentTime = 0.4;
    at = 400;
    context.mock.timers.tick(100);
    assert.equal(count, 2);
  } finally {
    stop();
  }
});

await test('stalled video stops independently of a slow in-flight inference', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const source = fakeVideo();
  let at = 0;
  let complete: () => void = () => {};
  const errors: unknown[] = [];
  const stop = startFramePump(
    source.video,
    () =>
      new Promise<void>((resolve) => {
        complete = resolve;
      }),
    (e) => errors.push(e),
    () => at,
  );
  try {
    source.frame(0);
    at = 3100;
    context.mock.timers.tick(250);
    assert.equal(errors.length, 1);
    assert.equal(source.pending(), 0);
    complete();
    await settle();
    context.mock.timers.tick(4000);
    assert.equal(errors.length, 1);
  } finally {
    stop();
  }
});

await test('inference failure stops the pump once and does not keep requesting frames', async () => {
  const source = fakeVideo();
  const errors: unknown[] = [];
  const stop = startFramePump(
    source.video,
    async () => {
      throw new Error('test failure');
    },
    (e) => errors.push(e),
  );
  try {
    source.frame(0);
    await settle();
    assert.equal(errors.length, 1);
    assert.equal(source.pending(), 0);
  } finally {
    stop();
  }
});

await test('silent compositor callbacks fall back only when camera media time still advances', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const source = fakeVideo();
  let at = 0,
    count = 0;
  const errors: unknown[] = [];
  const stop = startFramePump(
    source.video,
    async () => {
      count += 1;
    },
    (e) => errors.push(e),
    () => at,
  );
  try {
    at = 600;
    source.video.currentTime = 0.6;
    context.mock.timers.tick(250);
    await settle();
    assert.equal(source.pending(), 0);
    assert.equal(count, 1);
    at = 700;
    source.video.currentTime = 0.7;
    context.mock.timers.tick(16);
    await settle();
    assert.equal(count, 2);
    assert.equal(errors.length, 0);
  } finally {
    stop();
  }
});
