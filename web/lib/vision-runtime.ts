/** Scheduling/geometry only: all care evidence remains in original camera coordinates. */
export const VISION_TIMING = {
  maxAnalysisFps: 15,
  stallMs: 3000,
  fallbackMs: 16,
  watchdogMs: 250,
} as const;
export const displayX = (x: number, flipped: boolean) => (flipped ? 1 - x : x);
export function displayRect(
  rect: readonly [number, number, number, number],
  flipped: boolean,
): [number, number, number, number] {
  return flipped ? [1 - rect[2], rect[1], 1 - rect[0], rect[3]] : [...rect];
}

/** No queue: skip frames while busy, and wait for a NEW frame after completion. */
export function createFrameGate(
  intervalMs = 1000 / VISION_TIMING.maxAnalysisFps,
) {
  let busy = false,
    stopped = false,
    lastMediaTime = -1,
    startedAt = -Infinity;
  return {
    begin(mediaTime: number, at: number) {
      if (
        stopped ||
        !Number.isFinite(mediaTime) ||
        !Number.isFinite(at) ||
        mediaTime <= lastMediaTime
      )
        return false;
      lastMediaTime = mediaTime;
      if (busy || at - startedAt < intervalMs) return false;
      busy = true;
      startedAt = at;
      return true;
    },
    finish() {
      busy = false;
    },
    stop() {
      stopped = true;
    },
  };
}

type VideoSource = Pick<HTMLVideoElement, 'currentTime' | 'readyState'> &
  Partial<
    Pick<
      HTMLVideoElement,
      'requestVideoFrameCallback' | 'cancelVideoFrameCallback'
    >
  >;

/** Feature-detect rVFC; the fallback checks media time, never fabricates new frames. */
export function startFramePump(
  video: VideoSource,
  onFrame: () => Promise<void>,
  onError: (error: unknown) => void,
  clock: () => number = () => performance.now(),
) {
  const gate = createFrameGate();
  let stopped = false,
    callbackId: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let lastMediaTime = -1,
    lastAdvanceAt = clock();
  let native =
    typeof video.requestVideoFrameCallback === 'function' &&
    typeof video.cancelVideoFrameCallback === 'function';
  const halt = () => {
    stopped = true;
    gate.stop();
    if (callbackId !== null) video.cancelVideoFrameCallback?.(callbackId);
    if (timer !== null) clearTimeout(timer);
    clearInterval(watchdog);
  };
  const watchdog = setInterval(() => {
    // Some renderers throttle compositor callbacks for an occluded video. If
    // media time still advances, switch to the duplicate-checked fallback.
    if (
      !stopped &&
      native &&
      clock() - lastAdvanceAt > 500 &&
      video.readyState >= 2 &&
      video.currentTime > lastMediaTime
    ) {
      native = false;
      if (callbackId !== null) video.cancelVideoFrameCallback?.(callbackId);
      callbackId = null;
      pulse(video.currentTime);
    }
    if (!stopped && clock() - lastAdvanceAt > VISION_TIMING.stallMs) {
      halt();
      onError(new Error('摄像头连续 3 秒未产生新画面，已停止识别。'));
    }
  }, VISION_TIMING.watchdogMs);
  function schedule() {
    if (stopped) return;
    if (native)
      callbackId = video.requestVideoFrameCallback!((_at, metadata) => {
        if (native) pulse(metadata.mediaTime);
      });
    else
      timer = setTimeout(
        () => pulse(video.currentTime),
        VISION_TIMING.fallbackMs,
      );
  }
  function pulse(mediaTime: number) {
    if (stopped) return;
    schedule();
    const at = clock();
    if (video.readyState < 2 || !Number.isFinite(mediaTime)) return;
    if (mediaTime > lastMediaTime) {
      lastMediaTime = mediaTime;
      lastAdvanceAt = at;
    }
    if (!gate.begin(mediaTime, at)) return;
    void (async () => {
      try {
        await onFrame();
      } catch (error) {
        if (!stopped) {
          halt();
          onError(error);
        }
      } finally {
        gate.finish();
      }
    })();
  }
  schedule();
  return halt;
}
