'use client';

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import type { Dispatch } from 'react';
import {
  Activity,
  ArrowDown,
  Bell,
  Check,
  ChevronRight,
  Clock3,
  Download,
  Droplets,
  Flame,
  LayoutDashboard,
  ListChecks,
  Settings2,
  Shield,
  Thermometer,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import VisionPanel from '@/components/vision-panel';
import FrameworkPanel from '@/components/framework-panel';
import InterventionOutput from '@/components/intervention-output';
import { buildSessionReport } from '@/lib/care/session';
import { useZoneMonitor } from '@/components/use-zone-monitor';
import type { ZoneState } from '@/lib/zones';
import {
  activeEvents,
  canClose,
  careReducer,
  initialState,
  INITIAL_INPUT,
  inputIsAvailable,
  INTERVENTION_LABELS,
  PHASE_LABELS,
  POLICY,
} from '@/lib/care/engine';
import type {
  AlarmState,
  CareAction,
  CareEvent,
  CareState,
  EnvironmentInput,
  Intervention,
  Profile,
} from '@/lib/care/types';

const ALARM_LABELS: Record<AlarmState, string> = {
  normal: '正常',
  alarm: '报警',
  fault: '故障',
  offline: '离线',
};
const STEPS: Intervention[] = ['I0', 'I1', 'I2', 'I3', 'I4'];
function timeLabel(at: number | null) {
  return at === null
    ? '尚无采样'
    : new Date(at).toLocaleTimeString('zh-CN', { hour12: false });
}
function exportSession(state: CareState, zones: ZoneState) {
  const blob = new Blob(
    [JSON.stringify(buildSessionReport(state, zones), null, 2)],
    { type: 'application/json' },
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'active-care-demo-session.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function EventCard({
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
  const [note, setNote] = useState('');
  const [review, setReview] = useState('');
  const closed = event.phase === 'CLOSED';
  const ready = canClose(event, state, state.lastNow);
  const recoveryTime =
    event.source === 'camera'
      ? ((state.camera ?? []).find((o) => o.key === event.cameraKey)
          ?.capturedAt ?? 0)
      : state.lastNow;
  const remaining =
    event.normalSince === null
      ? null
      : Math.max(
          0,
          Math.ceil(
            (POLICY.recoveryMs - (recoveryTime - event.normalSince)) / 1000,
          ),
        );
  return (
    <article
      className={`event-card ${(event.risk === 'L4' || event.risk === 'L3') && !closed ? 'event-urgent' : ''}`}
    >
      <div className="event-title">
        <div>
          <span className="event-id">
            {event.id} · {timeLabel(event.createdAt)}
          </span>
          <h3>{event.title}</h3>
        </div>
        <Badge
          variant={
            closed
              ? 'secondary'
              : event.risk === 'L4'
                ? 'destructive'
                : 'outline'
          }
        >
          {closed ? '已结束' : event.risk}
        </Badge>
      </div>
      <div className="event-tags">
        <span>{PHASE_LABELS[event.phase]}</span>
        <span>
          {event.intervention} · {INTERVENTION_LABELS[event.intervention]}
        </span>
        <span
          className={
            event.source === 'camera' ? 'camera-source-label' : 'source-label'
          }
        >
          {event.source === 'camera' ? '摄像头证据' : '模拟输入'}
        </span>
        {event.target && <span>{event.target}</span>}
        {event.signal === 'unknown' && !closed && (
          <strong>观测未知 · 保留警报</strong>
        )}
      </div>
      <p className="event-evidence">{event.evidence}</p>
      <InterventionOutput
        event={event}
        state={state}
        dispatch={dispatch}
        now={now}
      />
      <div className="event-status">
        <span>
          {event.acknowledgedAt !== null ? '✓ 已确认接手' : '待人工接手'}
        </span>
        <span>
          反馈：
          {
            {
              none: '未收到',
              responded: '人工代填 · 已回应',
              declined: '人工代填 · 拒绝提醒',
              help: '人工代填 · 请求帮助',
            }[event.feedback]
          }
        </span>
        <span>
          预设：{event.profile === 'voice' ? '可理解简短提醒' : '需要图示支持'}
        </span>
      </div>
      {event.falseAlarmNote && (
        <p className="annotation">
          误报标注：{event.falseAlarmNote}（不覆盖异常输入）
        </p>
      )}
      {!closed && (
        <>
          <div className="event-actions">
            <Button
              size="sm"
              variant="outline"
              disabled={event.acknowledgedAt !== null}
              onClick={() =>
                dispatch({ type: 'acknowledge', id: event.id, now: now() })
              }
            >
              <Check size={14} />
              {event.acknowledgedAt !== null ? '已接手' : '确认接手'}
            </Button>
            <Button
              size="sm"
              disabled={!ready}
              onClick={() =>
                dispatch({ type: 'close', id: event.id, now: now() })
              }
            >
              结束事件
            </Button>
            <span>
              {event.kind === 'fall_candidate'
                ? '须填写实际核查说明，不能仅凭姿态恢复结束'
                : event.signal === 'unknown'
                  ? '等待有效输入'
                  : remaining === null
                    ? event.source === 'camera'
                      ? '须确认同一对象在区域外'
                      : '须先恢复模拟输入'
                    : remaining > 0
                      ? `稳定观察还需 ${remaining} 秒`
                      : '输入稳定，可人工结束'}
            </span>
          </div>
          {event.source === 'camera' &&
            (event.signal === 'unknown' || event.kind === 'fall_candidate') && (
              <details className="manual-review">
                <summary>
                  {event.kind === 'fall_candidate'
                    ? '疑似跌倒后的人工核查'
                    : '视觉证据中断后的人工核查'}
                </summary>
                <p>
                  仅在你已实际核查对象情况后使用。此操作记录人工处置，不代表系统识别到风险解除。
                </p>
                <div>
                  <Input
                    aria-label={`${event.id} 人工核查说明`}
                    maxLength={200}
                    value={review}
                    onChange={(e) => setReview(e.target.value)}
                    placeholder="填写实际核查情况与处置结果"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!review.trim()}
                    onClick={() =>
                      dispatch({
                        type: 'review-close',
                        id: event.id,
                        note: review,
                        now: now(),
                      })
                    }
                  >
                    记录核查并结束
                  </Button>
                </div>
              </details>
            )}
          <details className="feedback-details">
            <summary>
              记录反馈 / 标记误报 <ChevronRight size={13} />
            </summary>
            <p>由演示者代填；不代表系统识别到了患者回应。</p>
            <div className="feedback-buttons">
              {(
                [
                  ['responded', '已回应'],
                  ['declined', '拒绝提醒'],
                  ['help', '请求帮助'],
                ] as const
              ).map(([feedback, label]) => (
                <Button
                  key={feedback}
                  size="sm"
                  variant="secondary"
                  onClick={() =>
                    dispatch({
                      type: 'feedback',
                      id: event.id,
                      feedback,
                      now: now(),
                    })
                  }
                >
                  {label}
                </Button>
              ))}
            </div>
            <div className="false-alarm-form">
              <Input
                aria-label={`${event.id} 误报原因`}
                placeholder="填写误报原因，不会直接解除报警"
                maxLength={200}
                value={note}
                onChange={(e) => setNote(e.target.value)}
              />
              <Button
                size="sm"
                variant="outline"
                disabled={!note.trim()}
                onClick={() => {
                  dispatch({
                    type: 'false-alarm',
                    id: event.id,
                    note,
                    now: now(),
                  });
                  setNote('');
                }}
              >
                保存标注
              </Button>
            </div>
          </details>
        </>
      )}
      {closed && (
        <p className="closed-at">
          人工结束于 {timeLabel(event.closedAt)}
          {event.reviewNote &&
            ` · 人工核查说明：${event.reviewNote}（非系统安全结论）`}
        </p>
      )}
    </article>
  );
}

export default function CareDashboard() {
  const [state, dispatch] = useReducer(careReducer, undefined, initialState);
  const [input, setInput] = useState<EnvironmentInput>({ ...INITIAL_INPUT });
  const [running, setRunning] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const inputRef = useRef(input);
  const runningRef = useRef(true);
  const clockRef = useRef<{ epoch: number; monotonic: number } | null>(null);
  const now = useCallback(() => {
    if (!clockRef.current)
      clockRef.current = { epoch: Date.now(), monotonic: performance.now() };
    return Math.round(
      clockRef.current.epoch + performance.now() - clockRef.current.monotonic,
    );
  }, []);
  const zones = useZoneMonitor(dispatch, now);
  const tickZones = zones.tick;
  useEffect(() => {
    const sample = () => {
      const at = now();
      tickZones(at);
      dispatch(
        runningRef.current
          ? { type: 'sample', input: { ...inputRef.current }, now: at }
          : { type: 'tick', now: at },
      );
    };
    sample();
    const timer = setInterval(sample, 1000);
    return () => clearInterval(timer);
  }, [now, tickZones]);
  function changeInput(patch: Partial<EnvironmentInput>) {
    const next = { ...inputRef.current, ...patch };
    inputRef.current = next;
    setInput(next);
    if (runningRef.current)
      dispatch({ type: 'sample', input: next, now: now() });
  }
  function toggleRunning() {
    const next = !runningRef.current;
    runningRef.current = next;
    setRunning(next);
    dispatch(
      next
        ? { type: 'sample', input: { ...inputRef.current }, now: now() }
        : { type: 'tick', now: now() },
    );
  }
  const active = activeEvents(state);
  const priority = active[0];
  const available = inputIsAvailable(state.snapshot, state.lastNow);
  const smoke =
    available && state.snapshot ? ALARM_LABELS[state.snapshot.smoke] : '未知';
  const visibleEvents = showHistory ? state.events : active;
  const intervention = priority?.intervention ?? 'I0';
  const metrics = [
    {
      label: '室内温度',
      value:
        available && state.snapshot
          ? state.snapshot.temperature.toFixed(1)
          : '—',
      unit: '°C',
      icon: Thermometer,
      foot: '触发范围 18–28°C',
    },
    {
      label: '相对湿度',
      value:
        available && state.snapshot ? String(state.snapshot.humidity) : '—',
      unit: '%',
      icon: Droplets,
      foot: '触发范围 30–70%',
    },
    {
      label: '烟雾报警器',
      value: smoke,
      unit: '',
      icon: Flame,
      foot: `燃气：${available && state.snapshot ? ALARM_LABELS[state.snapshot.gas] : '未知'}`,
    },
    {
      label: '待处理事件',
      value: String(active.length).padStart(2, '0'),
      unit: '',
      icon: Activity,
      foot: priority
        ? `最高风险 ${priority.risk} · 保留至处理`
        : '无已触发事件，非安全结论',
    },
  ];
  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        跳转到内容
      </a>
      <aside className="sidebar">
        <div className="brand">
          <Shield size={28} />
          <div>
            主动伴护<small>ACTIVE CARE</small>
          </div>
        </div>
        <div className="sidebar-caption">工作空间</div>
        <nav aria-label="页面导航">
          <a className="nav-item active" href="#overview">
            <LayoutDashboard size={18} />
            伴护总览
          </a>
          <a className="nav-item" href="#events">
            <Bell size={18} />
            事件中心
          </a>
          <a className="nav-item" href="#simulator">
            <Settings2 size={18} />
            输入模拟
          </a>
          <a className="nav-item" href="#log">
            <ListChecks size={18} />
            决策记录
          </a>
          <a className="nav-item" href="#framework">
            <Activity size={18} />
            系统框架
          </a>
        </nav>
        <div className="sidebar-bottom">
          <span className="dot" />
          原型开发 · 第 04 部分<small>统一事件与完整流程框架</small>
        </div>
      </aside>
      <main className="workspace" id="main-content">
        <header className="topbar">
          <span>
            工作空间 <span className="crumb">/</span> 伴护总览
          </span>
          <Badge className="source-badge">环境模拟 · 视觉需手动开启</Badge>
        </header>
        <section className="page-heading" id="overview">
          <div>
            <div className="eyebrow">CARE OVERVIEW</div>
            <h1>每一次关注，都有依据。</h1>
            <p>从环境变化到事件处理，查看当前伴护状态。</p>
          </div>
          <div className="session-label">
            <span className="dot" />
            会话内演示<small>刷新页面将重置数据</small>
          </div>
        </section>
        <section className="metric-grid" aria-label="模拟环境状态">
          {metrics.map(({ label, value, unit, icon: Icon, foot }) => (
            <article
              key={label}
              className={`metric-card ${label === '烟雾报警器' && smoke === '报警' ? 'metric-alert' : ''}`}
            >
              <div className="metric-top">
                <span>{label}</span>
                <Icon size={19} />
              </div>
              <div className="metric-value">
                {value}
                <small>{unit}</small>
              </div>
              <div className="metric-foot">
                <span
                  className={
                    label === '待处理事件'
                      ? 'camera-source-label'
                      : 'source-label'
                  }
                >
                  {label === '待处理事件' ? '环境 + 摄像头' : '模拟数据'}
                </span>
                <span>{foot}</span>
              </div>
            </article>
          ))}
        </section>
        <div className="main-grid">
          <VisionPanel
            zoneState={zones.state}
            onZoneAction={zones.send}
            onObservation={zones.onObservation}
            pendingFalls={
              active.filter((e) => e.kind === 'fall_candidate').length
            }
            now={now}
          />
          <section className="panel decision-panel">
            <div className="panel-heading">
              <div>
                <h2>当前干预 · 统一事件</h2>
                <p>风险描述严重程度，干预描述帮助方式</p>
              </div>
              <Shield size={19} />
            </div>
            <div
              className={`decision-callout ${priority?.risk === 'L4' ? 'urgent' : ''}`}
              aria-live="polite"
            >
              <span className="eyebrow">
                {priority
                  ? `${priority.risk} · ${priority.intervention}`
                  : 'I0 · OBSERVE'}
              </span>
              <h3>
                {priority
                  ? INTERVENTION_LABELS[priority.intervention]
                  : '保持观察'}
              </h3>
              <p>
                {priority
                  ? `${priority.title}。${priority.signal === 'unknown' ? '观测丢失，保留已有事件。' : priority.phase === 'RECOVERING' ? '输入已恢复，等待稳定观察与人工结束。' : priority.intervention === 'I4' ? '网页请求协助；未发送手机消息或联系外部人员。' : priority.intervention === 'I2' ? '请查看触发区域或异常输入，并向观察对象提供帮助。' : priority.intervention === 'I1' ? '请关注当前异常，用简短、温和的方式提醒。' : '正在积累连续观测，暂不主动打扰。'}`
                  : available
                    ? '当前没有已触发事件。请查看视觉与区域监测状态，不能据此判断人员安全。'
                    : '等待有效模拟输入，不能判断环境正常。'}
              </p>
            </div>
            <ol className="intervention-track" aria-label="干预等级">
              {STEPS.map((step) => (
                <li
                  key={step}
                  className={`${step === intervention ? 'current' : ''} ${step === 'I3' ? 'unavailable' : ''}`}
                  aria-current={step === intervention ? 'step' : undefined}
                >
                  <span>{step}</span>
                  <small>
                    {
                      {
                        I0: '观察',
                        I1: '提醒',
                        I2: '引导',
                        I3: '未接入',
                        I4: '求助',
                      }[step]
                    }
                  </small>
                </li>
              ))}
            </ol>
            <p className="decision-note">
              I1 / I2 记录网页呈现回执后开始等待反馈。I3 仅模拟，I4 不要求先执行
              I3。
            </p>
          </section>
        </div>
        <FrameworkPanel state={state} />
        <section className="panel simulator-panel" id="simulator">
          <div className="panel-heading">
            <div>
              <h2>环境输入模拟台</h2>
              <p>演示者调节数值；不是传感器测量，也不是 AI 识别结果</p>
            </div>
            <Button size="sm" variant="outline" onClick={toggleRunning}>
              {running ? '暂停采样' : '恢复采样'}
            </Button>
          </div>
          <div className="sampling-line">
            <span className={`dot ${!available ? 'offline-dot' : ''}`} />
            {!state.snapshot
              ? '等待首次采样'
              : running
                ? '每秒模拟采样'
                : '已暂停采样'}
            <span>最近采样 {timeLabel(state.snapshot?.sampledAt ?? null)}</span>
            <strong>
              {available ? (running ? '输入可用' : '即将过期') : '输入未知'}
            </strong>
          </div>
          <div className="simulator-grid">
            <div className="range-control">
              <div>
                <span id="temperature-label">温度</span>
                <output>{input.temperature.toFixed(1)} °C</output>
              </div>
              <Slider
                aria-labelledby="temperature-label"
                value={[input.temperature]}
                min={5}
                max={45}
                step={0.5}
                onValueChange={(v) =>
                  changeInput({ temperature: Array.isArray(v) ? v[0] : v })
                }
              />
              <small>演示范围 18–28°C · 恢复范围 19–27°C</small>
            </div>
            <div className="range-control">
              <div>
                <span id="humidity-label">湿度</span>
                <output>{input.humidity} %</output>
              </div>
              <Slider
                aria-labelledby="humidity-label"
                value={[input.humidity]}
                min={10}
                max={95}
                step={1}
                onValueChange={(v) =>
                  changeInput({ humidity: Array.isArray(v) ? v[0] : v })
                }
              />
              <small>演示范围 30–70% · 恢复范围 35–65%</small>
            </div>
            {(['smoke', 'gas'] as const).map((kind) => (
              <div className="select-control" key={kind}>
                <label htmlFor={`${kind}-input`}>
                  {kind === 'smoke' ? '烟雾报警器' : '燃气报警器'}
                </label>
                <NativeSelect
                  id={`${kind}-input`}
                  className="full-select"
                  value={input[kind]}
                  onChange={(e) =>
                    changeInput({ [kind]: e.target.value as AlarmState })
                  }
                >
                  {Object.entries(ALARM_LABELS).map(([value, label]) => (
                    <NativeSelectOption key={value} value={value}>
                      {label}
                    </NativeSelectOption>
                  ))}
                </NativeSelect>
              </div>
            ))}
          </div>
          <div className="simulator-bottom">
            <label className="door-control" htmlFor="door-input">
              <Switch
                id="door-input"
                checked={input.doorOpen}
                onCheckedChange={(value) => changeInput({ doorOpen: value })}
              />
              门磁模拟：
              {available && state.snapshot
                ? state.snapshot.doorOpen
                  ? '门已打开'
                  : '门已关闭'
                : '状态未知'}
              <small>仅显示，不推断外出</small>
            </label>
            <div className="preset-buttons">
              <Button
                size="sm"
                variant="outline"
                disabled={!running}
                onClick={() => changeInput({ temperature: 32 })}
              >
                演示温度异常
              </Button>
              <Button
                size="sm"
                variant="destructive"
                disabled={!running}
                onClick={() => changeInput({ smoke: 'alarm' })}
              >
                演示烟雾报警
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => changeInput({ ...INITIAL_INPUT })}
              >
                恢复输入默认值
              </Button>
            </div>
          </div>
          <p className="control-note">
            恢复输入不会关闭事件。暂停后修改数值将在恢复采样时生效；超过 3.5
            秒未采样即视为未知。
          </p>
        </section>
        <section className="profile-strip">
          <div>
            <span className="eyebrow">MINIMUM NECESSARY SUPPORT</span>
            <h2>相同风险，不同支持。</h2>
            <p>能力预设由演示者选择，只影响新事件，不用于诊断。</p>
          </div>
          <div>
            <label htmlFor="profile">支持方式预设</label>
            <NativeSelect
              id="profile"
              className="profile-select"
              value={state.profile}
              onChange={(e) =>
                dispatch({
                  type: 'profile',
                  profile: e.target.value as Profile,
                  now: now(),
                })
              }
            >
              <NativeSelectOption value="voice">
                可理解简短提醒 · 从 I1 开始
              </NativeSelectOption>
              <NativeSelectOption value="visual">
                需要图示支持 · 从 I2 开始
              </NativeSelectOption>
            </NativeSelect>
            <small>
              普通异常持续 5 秒确认；提示呈现后 20
              秒未回应且证据仍有效，再增加支持。
            </small>
          </div>
        </section>
        <div className="records-grid">
          <section className="panel" id="events">
            <div className="panel-heading">
              <div>
                <h2>
                  事件中心 <span className="count">{active.length}</span>
                </h2>
                <p>确认接手 ≠ 风险解除 · 误报标注 ≠ 关闭</p>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowHistory((v) => !v)}
              >
                {showHistory ? '只看待处理' : '包含已结束'}
              </Button>
            </div>
            <div className="event-list">
              {visibleEvents.length ? (
                visibleEvents.map((event) => (
                  <EventCard
                    key={event.id}
                    event={event}
                    state={state}
                    dispatch={dispatch}
                    now={now}
                  />
                ))
              ) : (
                <div className="empty-events">
                  <Bell size={25} />
                  <h3>暂时没有{showHistory ? '' : '待处理'}事件</h3>
                  <p>在模拟台触发异常，或开启区域监测，查看事件处理过程。</p>
                  <a href="#simulator">
                    前往模拟台 <ArrowDown size={13} />
                  </a>
                </div>
              )}
            </div>
            <output className="action-notice">{state.notice}</output>
          </section>
          <section className="panel" id="log">
            <div className="panel-heading">
              <div>
                <h2>决策记录</h2>
                <p>记录状态变化，不重复刷屏</p>
              </div>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label="导出演示记录 JSON"
                disabled={!state.snapshot}
                onClick={() => exportSession(state, zones.state)}
              >
                <Download size={17} />
              </Button>
            </div>
            {state.logs.length ? (
              <ol className="log-list">
                {state.logs.map((entry) => (
                  <li key={entry.id}>
                    <span className="log-dot" />
                    <div>
                      <span className="log-time">
                        {timeLabel(entry.at)}{' '}
                        {entry.eventId && `· ${entry.eventId}`}
                      </span>
                      <p>{entry.message}</p>
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="empty-log">
                <Clock3 size={24} />
                <p>事件发生后，决策依据会留在这里。</p>
              </div>
            )}
            <div className="log-footnote">
              仅保存本次页面会话，最多展示 250 条记录。
            </div>
          </section>
        </div>
        <footer className="page-footer">
          <Shield size={14} />
          工业设计演示原型，不替代照护者或独立烟雾报警器。阈值仅为演示设置。刷新后数据重置。
        </footer>
      </main>
    </div>
  );
}
