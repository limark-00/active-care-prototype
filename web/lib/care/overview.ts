import type { CareState, Overview, OverviewCode } from './types.ts';

export const OVERVIEW_LABELS: Record<OverviewCode, string> = {
  UNKNOWN: '观测未知',
  SAFE: '当前未见目标异常',
  MONITORING: '持续观察',
  LOW_RISK: '关注中',
  MEDIUM_RISK: '需要支持',
  HIGH_RISK: '高度关注',
  EMERGENCY: '紧急事件待处理',
  RECOVERING: '恢复观察',
};
export const EMPTY_COVERAGE = {
  targetId: null,
  regionsArmed: false,
  regionsKnown: false,
  observingDwell: false,
};
export const INITIAL_OVERVIEW: Overview = {
  code: 'UNKNOWN',
  risk: 'UNKNOWN',
  observation: 'UNKNOWN',
  reason: '尚无有效观测。',
};

/** Derived labels, not a second mutable patient diagnosis/state machine. */
export function deriveOverview(s: CareState, now: number): Overview {
  const env = s.snapshot;
  const envValid =
    !!env &&
    env.online &&
    now >= env.sampledAt &&
    now - env.sampledAt <= 3500 &&
    Number.isFinite(env.temperature) &&
    env.temperature >= -40 &&
    env.temperature <= 80 &&
    Number.isFinite(env.humidity) &&
    env.humidity >= 0 &&
    env.humidity <= 100 &&
    [env.smoke, env.gas].every((v) => v === 'normal' || v === 'alarm');
  const v = s.vision;
  const visionValid = !!v && now >= v.capturedAt && now - v.capturedAt <= 3500;
  const c = s.coverage;
  const targetKnown =
    visionValid && v.persons.some((p) => p.id === c.targetId && p.reliable);
  const observation =
    envValid && targetKnown && c.regionsArmed && c.regionsKnown
      ? 'VALID'
      : envValid || visionValid
        ? 'DEGRADED'
        : 'UNKNOWN';
  const events = s.events.filter((e) => e.phase !== 'CLOSED');
  const unresolved = events.filter((e) => e.phase !== 'RECOVERING');
  const priority = [...unresolved].sort(
    (a, b) => Number(b.risk.slice(1)) - Number(a.risk.slice(1)),
  )[0];
  if (priority) {
    const codes = {
      L1: 'LOW_RISK',
      L2: 'MEDIUM_RISK',
      L3: 'HIGH_RISK',
      L4: 'EMERGENCY',
    } as const;
    return {
      code:
        priority.phase === 'CONFIRMING' ? 'MONITORING' : codes[priority.risk],
      risk: priority.risk,
      observation,
      reason: `${priority.title}；${priority.signal === 'unknown' ? '观测已丢失，已有风险记录仍保留' : '等待证据更新与人工处理'}。`,
    };
  }
  if (events.length)
    return {
      code: 'RECOVERING',
      risk: [...events].sort(
        (a, b) => Number(b.risk.slice(1)) - Number(a.risk.slice(1)),
      )[0].risk,
      observation,
      reason: '事件进入恢复观察，尚未结束；风险等级保留为本次事件记录。',
    };
  if (!envValid || !visionValid || (c.targetId !== null && !targetKnown))
    return {
      code: 'UNKNOWN',
      risk: 'UNKNOWN',
      observation,
      reason: '环境或视觉观测不足；没有事件不等于安全。',
    };
  if (observation !== 'VALID' || c.observingDwell)
    return {
      code: 'MONITORING',
      risk: 'UNKNOWN',
      observation,
      reason: c.observingDwell
        ? '区域内持续观察，尚未达到停留阈值。'
        : '请选定观察对象、校准并开启区域；当前监测范围尚不完整。',
    };
  return {
    code: 'SAFE',
    risk: 'L0',
    observation,
    reason: '仅指当前有效观测范围内未触发目标异常，不是人员安全或健康结论。',
  };
}
