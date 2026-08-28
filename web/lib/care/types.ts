import type { ZoneEvidence } from '../zones.ts';
/** UI-independent contracts; camera evidence stays separate from simulated inputs. */
export type AlarmState = 'normal' | 'alarm' | 'fault' | 'offline';
export type EventKind =
  | 'temperature'
  | 'humidity'
  | 'smoke'
  | 'gas'
  | 'device'
  | 'zone_dwell'
  | 'fall_candidate';
export type RiskLevel = 'L1' | 'L2' | 'L3' | 'L4';
export type Intervention = 'I0' | 'I1' | 'I2' | 'I3' | 'I4';
export type Phase =
  | 'CONFIRMING'
  | 'INTERVENING'
  | 'WAITING_RESPONSE'
  | 'ESCALATED'
  | 'RECOVERING'
  | 'CLOSED';
export type Signal = 'abnormal' | 'normal' | 'unknown';
export type Profile = 'voice' | 'visual';

/** No image, keypoints, or private backend session token enters the care engine. */
export interface VisionEvidence {
  sessionId: string;
  frameId: number;
  capturedAt: number;
  model: string;
  device: string;
  persons: { id: number; reliable: boolean; fallEvidence: string | null }[];
}
export interface Coverage {
  targetId: number | null;
  regionsArmed: boolean;
  regionsKnown: boolean;
  observingDwell: boolean;
}
export type OverviewCode =
  | 'UNKNOWN'
  | 'SAFE'
  | 'MONITORING'
  | 'LOW_RISK'
  | 'MEDIUM_RISK'
  | 'HIGH_RISK'
  | 'EMERGENCY'
  | 'RECOVERING';
export interface Overview {
  code: OverviewCode;
  risk: RiskLevel | 'L0' | 'UNKNOWN';
  observation: 'VALID' | 'DEGRADED' | 'UNKNOWN';
  reason: string;
}
export interface StateTransition extends Overview {
  at: number;
  from: OverviewCode;
}
export type OutputChannel = 'web' | 'protection' | 'phone';
export type OutputStatus =
  | 'requested'
  | 'executed'
  | 'simulated'
  | 'failed'
  | 'unavailable';
export interface ActionRecord {
  id: string;
  eventId: string;
  requestKey: string;
  action_type: Intervention;
  channel: OutputChannel;
  requested_at: number;
  completed_at: number | null;
  status: OutputStatus;
  result: string;
}

export interface EnvironmentInput {
  temperature: number;
  humidity: number;
  smoke: AlarmState;
  gas: AlarmState;
  doorOpen: boolean;
  online: boolean;
}
export interface Snapshot extends EnvironmentInput {
  source: 'simulated';
  sampledAt: number;
}
export interface CareEvent {
  id: string;
  kind: EventKind;
  title: string;
  risk: RiskLevel;
  intervention: Intervention;
  phase: Phase;
  signal: Signal;
  evidence: string;
  source: 'simulated' | 'camera';
  cameraKey?: string;
  target?: string;
  reviewNote?: string;
  profile: Profile;
  createdAt: number;
  abnormalSince: number | null;
  updatedAt: number;
  interventionAt: number | null;
  normalSince: number | null;
  acknowledgedAt: number | null;
  feedback: 'none' | 'responded' | 'declined' | 'help';
  closedAt: number | null;
  falseAlarmNote: string | null;
}
export interface LogEntry {
  id: number;
  at: number;
  eventId: string | null;
  message: string;
}
export interface CareState {
  snapshot: Snapshot | null;
  camera: ZoneEvidence[];
  vision: VisionEvidence | null;
  visionCursor: {
    sessionId: string;
    frameId: number;
    capturedAt: number;
  } | null;
  coverage: Coverage;
  actions: ActionRecord[];
  overview: Overview;
  transitions: StateTransition[];
  profile: Profile;
  events: CareEvent[];
  logs: LogEntry[];
  sequence: number;
  lastNow: number;
  notice: string;
}
export type CareAction =
  | {
      type: 'camera';
      observations: ZoneEvidence[];
      vision?: VisionEvidence | null;
      coverage?: Coverage;
      now: number;
    }
  | {
      type: 'output-result';
      actionId: string;
      status: 'executed' | 'failed';
      now: number;
    }
  | {
      type: 'request-output';
      id: string;
      channel: 'protection' | 'phone';
      now: number;
    }
  | { type: 'review-close'; id: string; note: string; now: number }
  | { type: 'sample'; input: EnvironmentInput; now: number }
  | { type: 'tick'; now: number }
  | { type: 'profile'; profile: Profile; now: number }
  | { type: 'acknowledge'; id: string; now: number }
  | {
      type: 'feedback';
      id: string;
      feedback: 'responded' | 'declined' | 'help';
      now: number;
    }
  | { type: 'false-alarm'; id: string; note: string; now: number }
  | { type: 'close'; id: string; now: number };
