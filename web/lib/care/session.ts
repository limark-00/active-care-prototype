import { POLICY } from './engine.ts';
import { ZONE_POLICY } from '../zones.ts';
import type { ZoneState } from '../zones.ts';
import type { CareState } from './types.ts';

/** Explicit allowlist: raw frames and backend authorization tokens never export. */
export function buildSessionReport(state: CareState, zones: ZoneState) {
  return {
    formatVersion: 3,
    prototypeVersion: '0.4',
    exportedAt: state.lastNow,
    warning:
      '工业设计演示原型；含模拟环境和摄像头规则候选，不是临床记录。时间为本会话时钟。',
    boundaries: {
      environment: 'simulated',
      vision: 'local-yolo-pose-and-rules',
      speech: 'not-connected',
      protection: 'simulation-only',
      phone: 'not-connected',
      storage: 'session-memory',
      imagesIncluded: false,
    },
    policy: { ...POLICY, zones: ZONE_POLICY },
    profile: state.profile,
    snapshot: state.snapshot,
    overview: state.overview,
    transitions: state.transitions,
    events: state.events,
    actions: state.actions,
    logs: state.logs,
    vision: state.vision,
    coverage: state.coverage,
    regions: {
      zones: zones.zones,
      selected: zones.selected,
      armed: zones.armed,
      evidence: state.camera,
      logs: zones.logs,
    },
    retention: {
      stateTransitions: 100,
      decisionLogs: 250,
      regionLogs: 80,
      eventsAndActions: '本会话全部；刷新清空',
    },
  };
}
