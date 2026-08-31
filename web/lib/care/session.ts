import { POLICY } from './engine.ts';
import { ZONE_POLICY } from '../zones.ts';
import type { ZoneState } from '../zones.ts';
import type { AIDecisionRecord, CareState } from './types.ts';

function exportedDecision(
  decision: AIDecisionRecord | null,
): AIDecisionRecord | null {
  if (!decision) return decision;
  return {
    ...decision,
    decision: decision.decision
      ? {
          ...decision.decision,
          reviewReasons: [...decision.decision.reviewReasons],
        }
      : null,
  };
}

/** Explicit allowlist: raw frames and backend authorization tokens never export. */
export function buildSessionReport(state: CareState, zones: ZoneState) {
  return {
    formatVersion: 5,
    prototypeVersion: '0.6',
    exportedAt: state.lastNow,
    warning:
      '工业设计演示原型；含模拟环境和摄像头规则候选，不是临床记录。时间为本会话时钟。',
    boundaries: {
      environment: 'simulated',
      vision: 'local-yolo-pose-and-rules',
      speech: 'not-connected',
      protection: 'simulation-only',
      phone: 'not-connected',
      decisionModel: 'local-v31-with-rule-floor',
      feedback:
        'caregiver-entered-enums-and-observed-signal-only; no speech-comprehension detection',
      storage: 'session-memory',
      imagesIncluded: false,
      rawManualTextIncluded: false,
    },
    policy: { ...POLICY, zones: ZONE_POLICY },
    profile: state.profile,
    scene: state.scene,
    snapshot: state.snapshot,
    overview: state.overview,
    transitions: state.transitions,
    events: state.events.map((event) => ({
      id: event.id,
      kind: event.kind,
      title: event.title,
      risk: event.risk,
      intervention: event.intervention,
      phase: event.phase,
      signal: event.signal,
      evidence: event.evidence,
      source: event.source,
      context: {
        schemaVersion: event.context.schemaVersion,
        revision: event.context.revision,
        eventId: event.context.eventId,
        scene: event.context.scene,
        source: event.context.source,
        kind: event.context.kind,
        title: event.context.title,
        evidence: event.context.evidence,
        observedAt: event.context.observedAt,
        profile: event.context.profile,
        response: event.context.response,
        previousIntervention: event.context.previousIntervention,
        previousOutcome: event.context.previousOutcome,
        rawText: null,
      },
      ruleDecision: event.ruleDecision ? { ...event.ruleDecision } : null,
      aiDecision: exportedDecision(event.aiDecision),
      aiDecisionHistory: (event.aiDecisionHistory ?? []).map((decision) =>
        exportedDecision(decision),
      ),
      cameraKey: event.cameraKey,
      target: event.target,
      reviewNote: event.reviewNote,
      profile: event.profile,
      createdAt: event.createdAt,
      abnormalSince: event.abnormalSince,
      updatedAt: event.updatedAt,
      interventionAt: event.interventionAt,
      normalSince: event.normalSince,
      acknowledgedAt: event.acknowledgedAt,
      feedback: event.feedback,
      feedbackAt: event.feedbackAt,
      feedbackRequestKey: event.feedbackRequestKey,
      feedbackHistory: (event.feedbackHistory ?? []).map((record) => ({
        id: record.id,
        eventId: record.eventId,
        requestKey: record.requestKey,
        at: record.at,
        source: record.source,
        feedback: record.feedback,
        signal: record.signal,
        interventionBefore: record.interventionBefore,
        interventionAfter: record.interventionAfter,
        outcome: record.outcome,
        reason: record.reason,
      })),
      closedAt: event.closedAt,
      falseAlarmNote: event.falseAlarmNote,
    })),
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
      feedbackHistoryPerEvent: 30,
      aiDecisionHistoryPerEvent: 12,
    },
  };
}
