'use client';
import { Badge } from '@/components/ui/badge';
import {
  EMPTY_COVERAGE,
  INITIAL_OVERVIEW,
  OVERVIEW_LABELS,
} from '@/lib/care/overview';
import type { CareState } from '@/lib/care/types';

export default function FrameworkPanel({ state }: { state: CareState }) {
  const overview = state.overview ?? INITIAL_OVERVIEW;
  const coverage = state.coverage ?? EMPTY_COVERAGE;
  const transitions = state.transitions ?? [];
  const vision = state.vision;
  const fresh = !!vision && state.lastNow - vision.capturedAt <= 3500;
  const requests = state.events.filter(
    (e) => e.phase !== 'CLOSED' && e.intervention === 'I4',
  );
  const pending = requests.filter((e) => e.acknowledgedAt === null).length;
  const modules = [
    {
      name: '01 · 视觉感知',
      status: fresh ? '实时证据' : '未知 / 未开启',
      text: fresh
        ? `${vision.model} · ${vision.device.toUpperCase()} · 当前 ${vision.persons.length} 人`
        : '需手动开启摄像头；人物检测与 17 个关键点来自本机 YOLO。',
      href: '#vision',
    },
    {
      name: '02 · 场景与行为',
      status: coverage.regionsArmed ? '区域监测已开启' : '等待配置',
      text: '人工选人、区域进出与停留、疑似跌倒候选；不判断身份或真实米数。',
      href: '#vision',
    },
    {
      name: '03 · 环境输入',
      status: '模拟数据',
      text: '温湿度、烟雾、燃气与门磁。真实传感器尚未接入。',
      href: '#simulator',
    },
    {
      name: '04 · 风险与决策',
      status: '规则引擎已接入',
      text: '按证据选择 L 风险；再结合支持预设、反馈与执行结果选择 I 干预。',
      href: '#events',
    },
    {
      name: '05 · 网页求助',
      status: `${pending} 条待接手`,
      text: `${requests.length - pending} 条已接手但未结束。接手、反馈、误报、结束分别记录。`,
      href: '#events',
    },
    {
      name: '06 · 扩展与记录',
      status: '接口有边界',
      text: 'I3 返回模拟，手机返回未接通。状态、事件、回执可导出 JSON。',
      href: '#log',
    },
  ];
  return (
    <section className="panel framework-panel" id="framework">
      <div className="panel-heading">
        <div>
          <h2>伴护状态与系统框架</h2>
          <p>AI 感知 → 证据规则 → 个体支持 → 执行回执 → 反馈与恢复</p>
        </div>
        <Badge variant="outline">框架 v0.4</Badge>
      </div>
      <div
        className={`overview-state ${overview.code === 'EMERGENCY' || overview.code === 'HIGH_RISK' ? 'overview-attention' : ''}`}
      >
        <div>
          <span className="eyebrow">
            {overview.code} · {overview.risk}
          </span>
          <h3>{OVERVIEW_LABELS[overview.code]}</h3>
          <p>{overview.reason}</p>
        </div>
        <div className="coverage-label">
          <strong>
            观测质量 ·{' '}
            {
              { VALID: '当前范围有效', DEGRADED: '部分可用', UNKNOWN: '未知' }[
                overview.observation
              ]
            }
          </strong>
          <small>环境始终为模拟；视觉与区域需单独启用。</small>
        </div>
      </div>
      <div className="module-grid">
        {modules.map((m) => (
          <a className="module-card" href={m.href} key={m.name}>
            <div>
              <strong>{m.name}</strong>
              <span>{m.status}</span>
            </div>
            <p>{m.text}</p>
          </a>
        ))}
      </div>
      <details className="state-history">
        <summary>状态转移记录 · {transitions.length} 条</summary>
        {transitions.length ? (
          <ol>
            {transitions.slice(0, 20).map((t, i) => (
              <li key={`${t.at}-${i}`}>
                <time>
                  {new Date(t.at).toLocaleTimeString('zh-CN', {
                    hour12: false,
                  })}
                </time>
                <strong>
                  {t.from} → {t.code}
                </strong>
                <p>{t.reason}</p>
                <small>
                  风险 {t.risk} · 观测 {t.observation}
                </small>
              </li>
            ))}
          </ol>
        ) : (
          <p>等待有效输入。数据缺失不会被显示为 SAFE。</p>
        )}
        <small>
          显示最近 20 条，导出最多 100
          条。状态由事件与观测质量派生，不是医学判断。
        </small>
      </details>
    </section>
  );
}
