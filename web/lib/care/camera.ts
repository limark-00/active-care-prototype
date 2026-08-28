import { zoneEvidence } from '../zones.ts';
import type { ZoneState } from '../zones.ts';
import type { CareAction } from './types.ts';

/** One frame snapshot supplies both sources, without images or authorization tokens. */
export function cameraAction(zones: ZoneState, now: number): CareAction {
  const o = zones.observation;
  return {
    type: 'camera',
    observations: zoneEvidence(zones, now),
    now,
    vision: o
      ? {
          sessionId: o.sessionId,
          frameId: o.frame.frame_id,
          capturedAt: o.capturedAt,
          model: o.frame.model,
          device: o.frame.device,
          persons: o.frame.persons.map((p) => ({
            id: p.id,
            reliable: p.pose_reliable,
            fallEvidence:
              p.actions.find((a) => a.code === 'fall_candidate')?.evidence ??
              null,
          })),
        }
      : null,
    coverage: {
      targetId: zones.selected?.personId ?? null,
      regionsArmed: zones.armed,
      regionsKnown:
        zones.statuses.length > 0 &&
        zones.statuses.every((s) => s.position !== 'unknown'),
      observingDwell: zones.statuses.some((s) => s.position === 'inside'),
    },
  };
}
