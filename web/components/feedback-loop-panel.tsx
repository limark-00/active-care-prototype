'use client';

import type { Dispatch } from 'react';
import {
  CheckCircle2,
  Clock3,
  History,
  MessageCircleReply,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  FEEDBACK_LABELS,
  FEEDBACK_OUTCOME_LABELS,
  POLICY,
  PROFILE_LABELS,
  responseWindowMs,
} from '@/lib/care/engine';
import {
  currentOutput,
  outputKey,
  responseWindowStarted,
} from '@/lib/care/outputs';
import type {
  CareAction,
  CareEvent,
  CareState,
  PatientFeedback,
} from '@/lib/care/types';

const BUTTONS: { feedback: PatientFeedback; label: string }[] = [
  { feedback: 'improved', label: '已按提醒行动' },
  { feedback: 'responded', label: '已回应，正在处理' },
  { feedback: 'risk_persisted', label: '复核后风险仍在' },
  { feedback: 'declined', label: '拒绝本次提醒' },
  { feedback: 'help', label: '主动请求帮助' },
];

function seconds(ms: number) {
  return Math.max(0, Math.ceil(ms / 1000));
}

export default function FeedbackLoopPanel({
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
  const output = currentOutput(state, event);
  const currentKey = outputKey(event);
  const startedAt = responseWindowStarted(state, event);
  const windowMs = responseWindowMs(event.profile);
  const hasCurrentFeedback =
    !!output && event.feedbackRequestKey === output.requestKey;
  const pending = (event.feedbackHistory ?? []).find(
    (record) =>
      record.requestKey === output?.requestKey &&
      record.outcome === 'pending_verification',
  );
  const responseRemaining =
    startedAt === null ? null : seconds(windowMs - (state.lastNow - startedAt));
  const followupRemaining = pending
    ? seconds(POLICY.followupMs - (state.lastNow - pending.at))
    : null;
  const waiting =
    event.intervention !== 'I0' &&
    event.intervention !== 'I4' &&
    event.acknowledgedAt === null &&
    startedAt !== null &&
    !hasCurrentFeedback;
  const progress = waiting
    ? Math.min(
        100,
        Math.max(
          0,
          ((windowMs - (responseRemaining ?? 0) * 1000) / windowMs) * 100,
        ),
      )
    : pending
      ? Math.min(
          100,
          Math.max(
            0,
            ((POLICY.followupMs - (followupRemaining ?? 0) * 1000) /
              POLICY.followupMs) *
              100,
          ),
        )
      : 0;
  const feedbackEnabled = output?.status === 'executed';
  const history = event.feedbackHistory ?? [];

  let status = '尚未进入回应窗口';
  let detail = '观察阶段不会假定患者已经收到提醒。';
  if (event.acknowledgedAt !== null) {
    status = '照护者已接手';
    detail = '自动未回应升级已暂停，但风险证据仍会继续更新。';
  } else if (event.intervention === 'I4') {
    status = '等待照护者处理';
    detail = '当前已是网页求助等级，不再因计时继续提高风险。';
  } else if (pending) {
    status = `已记录反馈 · ${followupRemaining ?? 0} 秒后复核`;
    detail = '回应不等于危险解除；系统继续等待摄像头或环境证据。';
  } else if (waiting) {
    status = `等待患者回应 · ${responseRemaining ?? 0} 秒`;
    detail = `倒计时从网页实际呈现开始；超时只增加支持，不提高 ${event.risk} 风险等级。`;
  } else if (output?.status === 'requested') {
    status = '提示等待呈现';
    detail = '网页尚未给出呈现回执，因此响应倒计时没有开始。';
  } else if (hasCurrentFeedback) {
    status = FEEDBACK_LABELS[event.feedback as PatientFeedback];
    detail = '本轮反馈已经记录；新的干预轮次会重新开始独立计时。';
  }

  return (
    <section className="feedback-loop" aria-label={`${event.id} 患者反馈闭环`}>
      <div className="feedback-loop-heading">
        <div>
          <span className="eyebrow">FEEDBACK LOOP</span>
          <strong>患者反馈与再决策</strong>
        </div>
        <span className="feedback-loop-status">
          {waiting || pending ? (
            <Clock3 size={14} />
          ) : (
            <CheckCircle2 size={14} />
          )}
          {status}
        </span>
      </div>
      {(waiting || pending) && (
        <progress
          className="feedback-progress"
          max={100}
          value={Math.round(progress)}
          aria-label={waiting ? '患者回应等待进度' : '反馈后证据复核进度'}
        />
      )}
      <p>{detail}</p>
      <div className="feedback-reason-grid">
        <span>
          <small>风险依据</small>
          <strong>
            {event.risk} ·{' '}
            {event.signal === 'abnormal'
              ? '异常仍在'
              : event.signal === 'normal'
                ? '证据恢复'
                : '观测未知'}
          </strong>
        </span>
        <span>
          <small>能力快照</small>
          <strong>{PROFILE_LABELS[event.profile]}</strong>
        </span>
        <span>
          <small>当前支持</small>
          <strong>
            {event.intervention} · 第{' '}
            {Math.max(
              1,
              (state.actions ?? []).filter(
                (action) =>
                  action.eventId === event.id && action.channel === 'web',
              ).length,
            )}{' '}
            轮
          </strong>
        </span>
      </div>
      <div className="feedback-loop-buttons">
        {BUTTONS.map(({ feedback, label }) => {
          const followup = feedback === 'risk_persisted';
          const help = feedback === 'help';
          const disabled = help
            ? event.feedback === 'help'
            : followup
              ? !pending
              : !feedbackEnabled || hasCurrentFeedback;
          return (
            <Button
              key={feedback}
              size="sm"
              variant={help ? 'destructive' : 'secondary'}
              disabled={disabled}
              onClick={() =>
                dispatch({
                  type: 'feedback',
                  id: event.id,
                  feedback,
                  requestKey: currentKey,
                  now: now(),
                })
              }
            >
              <MessageCircleReply size={13} />
              {label}
            </Button>
          );
        })}
      </div>
      <small className="feedback-boundary">
        反馈由演示者代填，不代表系统识别到患者理解；行为改善仍需现场证据验证。拒绝提醒不会授权限制行动。
      </small>
      {history.length > 0 && (
        <details className="feedback-history">
          <summary>
            <History size={14} /> 反馈与再决策轨迹 · {history.length} 条
          </summary>
          <ol>
            {history.map((record) => (
              <li key={record.id}>
                <time>
                  {new Date(record.at).toLocaleTimeString('zh-CN', {
                    hour12: false,
                  })}
                </time>
                <div>
                  <strong>{FEEDBACK_LABELS[record.feedback]}</strong>
                  <span>
                    {record.interventionBefore} → {record.interventionAfter} ·{' '}
                    {FEEDBACK_OUTCOME_LABELS[record.outcome]}
                  </span>
                  <p>{record.reason}</p>
                </div>
              </li>
            ))}
          </ol>
        </details>
      )}
      {(event.aiDecisionHistory ?? []).length > 0 && (
        <small className="feedback-ai-history">
          已存档 {(event.aiDecisionHistory ?? []).length} 次旧上下文 AI
          判断；反馈更新后只融合最新请求。
        </small>
      )}
    </section>
  );
}
