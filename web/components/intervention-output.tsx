'use client';
import { useEffect } from 'react';
import type { Dispatch } from 'react';
import { Button } from '@/components/ui/button';
import {
  CHANNEL_LABELS,
  currentOutput,
  interventionText,
  OUTPUT_LABELS,
} from '@/lib/care/outputs';
import type { CareAction, CareEvent, CareState } from '@/lib/care/types';

/** A receipt confirms visible DOM presentation, never human comprehension. */
export default function InterventionOutput({
  event,
  state,
  dispatch,
  now,
}: {
  event: CareEvent;
  state: CareState;
  dispatch: Dispatch<CareAction>;
  now: () => number;
}) {
  const action = currentOutput(state, event);
  const actionId = action?.id;
  const pending = action?.status === 'requested' && event.phase !== 'CLOSED';
  useEffect(() => {
    if (!pending || !actionId) return;
    const delivered = () => {
      if (document.visibilityState === 'visible')
        dispatch({
          type: 'output-result',
          actionId,
          status: 'executed',
          now: now(),
        });
    };
    delivered();
    document.addEventListener('visibilitychange', delivered);
    return () => document.removeEventListener('visibilitychange', delivered);
  }, [actionId, pending, dispatch, now]);
  const history = (state.actions ?? []).filter((a) => a.eventId === event.id);
  const closed = event.phase === 'CLOSED';
  return (
    <div className="intervention-output">
      {!closed && event.intervention !== 'I0' && (
        <>
          <div className="output-heading">
            <strong>
              {event.intervention} ·{' '}
              {event.intervention === 'I4'
                ? '网页照护者请求'
                : event.intervention === 'I3'
                  ? '保护建议（仅模拟）'
                  : '当前提示'}
            </strong>
            <span>
              {action ? OUTPUT_LABELS[action.status] : '等待决策记录'}
            </span>
          </div>
          <ol
            className={
              event.intervention === 'I2' ? 'guidance-steps' : 'prompt-steps'
            }
          >
            {interventionText(event).map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ol>
          <small>
            当前仅文字 / 步骤图示。未播放语音；已显示不等于患者已理解。
          </small>
        </>
      )}
      {history.length > 0 && (
        <details className="output-history">
          <summary>干预执行记录 · {history.length} 条</summary>
          <ol>
            {history.map((a) => (
              <li key={a.id}>
                <strong>
                  {a.action_type} · {CHANNEL_LABELS[a.channel]} ·{' '}
                  {OUTPUT_LABELS[a.status]}
                </strong>
                <span>
                  {new Date(a.requested_at).toLocaleTimeString('zh-CN', {
                    hour12: false,
                  })}{' '}
                  · {a.id}
                </span>
                <p>{a.result}</p>
              </li>
            ))}
          </ol>
        </details>
      )}
      {!closed && (
        <details className="adapter-controls">
          <summary>扩展接口演示（无外部动作）</summary>
          <p>I3 只生成保护建议。手机接口只返回未接通；不会发送任何消息。</p>
          <div className="event-actions">
            <Button
              size="sm"
              variant="outline"
              disabled={
                !(event.risk === 'L3' || event.risk === 'L4') ||
                history.some((a) => a.channel === 'protection')
              }
              onClick={() =>
                dispatch({
                  type: 'request-output',
                  id: event.id,
                  channel: 'protection',
                  now: now(),
                })
              }
            >
              模拟 I3 保护请求
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={history.some((a) => a.channel === 'phone')}
              onClick={() =>
                dispatch({
                  type: 'request-output',
                  id: event.id,
                  channel: 'phone',
                  now: now(),
                })
              }
            >
              检查手机通知接口
            </Button>
          </div>
        </details>
      )}
    </div>
  );
}
