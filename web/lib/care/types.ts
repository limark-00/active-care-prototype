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
  | 'fall_candidate'
  | 'manual_text';
export type RiskLevel = 'L1' | 'L2' | 'L3' | 'L4';
export type ModelRiskLevel = 'L0' | RiskLevel;
export type Intervention = 'I0' | 'I1' | 'I2' | 'I3' | 'I4';
export type AlertMode = 'NONE' | 'PAGE_WARNING' | 'URGENT_HELP';
export type Phase =
  | 'CONFIRMING'
  | 'INTERVENING'
  | 'WAITING_RESPONSE'
  | 'ESCALATED'
  | 'RECOVERING'
  | 'CLOSED';
export type Signal = 'abnormal' | 'normal' | 'unknown';
export type Profile = 'voice' | 'visual';
export type PatientFeedback =
  | 'responded'
  | 'improved'
  | 'no_response'
  | 'risk_persisted'
  | 'declined'
  | 'help';
export type FeedbackSource = 'caregiver_report' | 'system_timeout';
export type FeedbackOutcome =
  | 'pending_verification'
  | 'verified_recovery'
  | 'risk_persisted'
  | 'intervention_escalated'
  | 'recorded';
export type CareScene =
  | '厨房'
  | '卫生间'
  | '卧室'
  | '客厅'
  | '入户门区域'
  | '阳台';
export type EventSource = 'simulated' | 'camera' | 'manual';

/** One normalized envelope is used by camera, simulated and manual inputs. */
export interface UnifiedEventContext {
  schemaVersion: 2;
  revision: number;
  eventId: string;
  scene: string;
  source: EventSource;
  kind: EventKind;
  title: string;
  evidence: string;
  observedAt: number;
  profile: Profile;
  response:
    | 'not_requested'
    | 'responded'
    | 'improved'
    | 'no_response'
    | 'risk_persisted'
    | 'declined'
    | 'requested_help'
    | 'unknown';
  previousIntervention: Intervention | 'none';
  previousOutcome:
    | 'not_available'
    | 'pending'
    | 'effective_reported'
    | 'ineffective'
    | 'declined'
    | 'help_requested';
  rawText: string | null;
}
export interface RuleDecision {
  risk: RiskLevel;
  intervention: Intervention;
  reason: string;
}
export interface AIDecisionPayload {
  model: string;
  runName: string;
  device: string;
  inferenceMs: number;
  risk: ModelRiskLevel;
  intervention: Intervention;
  alertMode: AlertMode;
  manualReview: boolean;
  abstain: boolean;
  riskConfidence: number;
  interventionConfidence: number;
  alertConfidence: number;
  reviewReasons: string[];
}
export interface AIDecisionRecord {
  status: 'running' | 'completed' | 'failed';
  requestKey: string;
  requestedAt: number;
  completedAt: number | null;
  decision: AIDecisionPayload | null;
  applied: boolean;
  finalRisk: RiskLevel | null;
  finalIntervention: Intervention | null;
  error: string | null;
}

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
  risk: ModelRiskLevel | 'UNKNOWN';
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
export interface FeedbackRecord {
  id: string;
  eventId: string;
  requestKey: string;
  at: number;
  source: FeedbackSource;
  feedback: PatientFeedback;
  signal: Signal;
  interventionBefore: Intervention;
  interventionAfter: Intervention;
  outcome: FeedbackOutcome;
  reason: string;
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
  source: EventSource;
  context: UnifiedEventContext;
  ruleDecision: RuleDecision | null;
  aiDecision: AIDecisionRecord | null;
  aiDecisionHistory: AIDecisionRecord[];
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
  feedback: 'none' | PatientFeedback;
  feedbackAt: number | null;
  feedbackRequestKey: string | null;
  feedbackHistory: FeedbackRecord[];
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
  scene: CareScene;
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
  | { type: 'scene'; scene: CareScene; now: number }
  | { type: 'ai-start'; id: string; requestKey: string; now: number }
  | {
      type: 'ai-result';
      id: string;
      requestKey: string;
      decision: AIDecisionPayload;
      now: number;
    }
  | {
      type: 'ai-failed';
      id: string;
      requestKey: string;
      error: string;
      now: number;
    }
  | { type: 'ai-retry'; id: string; now: number }
  | {
      type: 'manual-ai-event';
      text: string;
      decision: AIDecisionPayload;
      now: number;
    }
  | { type: 'acknowledge'; id: string; now: number }
  | {
      type: 'feedback';
      id: string;
      feedback: PatientFeedback;
      requestKey?: string;
      now: number;
    }
  | { type: 'false-alarm'; id: string; note: string; now: number }
  | { type: 'close'; id: string; now: number };
