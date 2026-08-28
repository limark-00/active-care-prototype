'use client';
import { useCallback, useRef, useState } from 'react';
import type { Dispatch } from 'react';
import type { CareAction } from '@/lib/care/types';
import { initialZoneState, zoneReducer } from '@/lib/zones';
import { cameraAction } from '@/lib/care/camera';
import type { CameraObservation, ZoneAction } from '@/lib/zones';

export function useZoneMonitor(
  dispatch: Dispatch<CareAction>,
  now: () => number,
) {
  const [state, setState] = useState(initialZoneState);
  const current = useRef(state);
  const send = useCallback(
    (action: ZoneAction) => {
      const next = zoneReducer(current.current, action);
      if (next === current.current) return;
      current.current = next;
      setState(next);
      dispatch(cameraAction(next, action.now));
    },
    [dispatch],
  );
  const onObservation = useCallback(
    (observation: CameraObservation | null) =>
      send({ type: 'frame', observation, now: now() }),
    [send, now],
  );
  const tick = useCallback(
    (at: number) => send({ type: 'tick', now: at }),
    [send],
  );
  return { state, send, onObservation, tick };
}
