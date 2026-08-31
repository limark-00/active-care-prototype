import type {
  ActionRecord,
  CareEvent,
  CareState,
  OutputChannel,
} from './types.ts';

export const OUTPUT_LABELS = {
  requested: '等待呈现',
  executed: '已在网页呈现',
  simulated: '仅模拟',
  failed: '呈现失败',
  unavailable: '未接通',
};
export const CHANNEL_LABELS: Record<OutputChannel, string> = {
  web: '网页',
  protection: '保护动作接口',
  phone: '手机通知接口',
};
export const outputKey = (e: CareEvent) =>
  `${e.id}:${e.intervention}:${e.interventionAt}`;
export function currentOutput(
  s: CareState,
  e: CareEvent,
): ActionRecord | undefined {
  return (s.actions ?? []).find(
    (a) => a.channel === 'web' && a.requestKey === outputKey(e),
  );
}
export function responseWindowStarted(
  s: CareState,
  e: CareEvent,
): number | null {
  const a = currentOutput(s, e);
  return a?.status === 'executed' ? a.completed_at : null;
}
/** Unsupported adapters deliberately never report successful external execution. */
export function adapterResult(channel: 'protection' | 'phone') {
  return channel === 'protection'
    ? {
        status: 'simulated' as const,
        result:
          '已记录 I3 保护动作建议；没有执行器，不关火、不锁门、不限制行动。',
      }
    : {
        status: 'unavailable' as const,
        result: '手机通知适配器未配置；未发送消息，未联系家属或外部人员。',
      };
}
export function interventionText(e: CareEvent): string[] {
  if (e.kind === 'fall_candidate')
    return [
      '画面出现疑似跌倒候选，请照护者现场核查。',
      '后续站立、摄像头断开或点击接手都不会自动解除此事件。',
    ];
  if (e.intervention === 'I4')
    return [
      '请照护者接手并核查当前事件。',
      '这是网页内待处理请求，未联系外部人员。',
    ];
  if (e.intervention === 'I3')
    return [
      '模型建议进入保护级处置，请照护者立即核查并移除当前危险源。',
      '本原型没有设备执行器，不会自动关火、断电、锁门或限制人员行动。',
    ];
  if (e.intervention === 'I2')
    return [
      '先查看本事件的对象、区域或异常输入。',
      '由照护者确认现场情况，再向观察对象提供清楚、简短的帮助。',
      '查看后续证据并记录反馈；图示步骤不构成安全路径。',
    ];
  return ['请留意当前异常。可以温和地提醒观察对象，并确认是否需要帮助。'];
}
