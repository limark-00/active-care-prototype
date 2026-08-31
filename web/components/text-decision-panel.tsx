'use client';

import { useRef, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  LoaderCircle,
  Play,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  requestTextDecision,
  type DecisionResult,
  type Prediction,
} from '@/lib/decision';

const EXAMPLES = {
  中等风险: `地点：厨房。请根据下面的现场记录进行判断。
事件1（摄像头）：患者反复伸手操作与当前锅具不对应的灶台旋钮。
事件2（人工输入）：盛有热汤的锅具被放在主要通行路线边缘。
理解与求助能力：理解语音不稳定，需要图示和分步骤引导。
当前回应：已发出的提醒没有获得可确认的回应。
之前的处理：温和提醒已经展示，效果还要结合现在的回应判断。
请给出下一步干预、网页警示，以及是否转人工核查。`,
  明确危险: `地点：客厅。请根据下面的现场记录进行判断。
事件1（摄像头）：倾斜电视柜正在继续位移且患者位于其前方。
事件2（传感器）：破损插座出现火花且附近设备仍保持供电。
理解与求助能力：能够理解简短语音，也能够主动按键求助。
当前回应：尚未发出提醒，因此没有回应记录。
之前的处理：之前还没有进行干预。
请给出下一步干预、网页警示，以及是否转人工核查。`,
  紧急求助: `地点：卧室。请根据下面的现场记录进行判断。
事件1（摄像头）：卧室出现浓烟且患者在床边没有可确认回应。
理解与求助能力：当前没有足够资料判断理解和主动求助能力。
当前回应：当前观测不足，无法判断患者是否回应。
之前的处理：之前还没有进行干预。
请给出下一步干预、网页警示，以及是否转人工核查。`,
};

const INTERVENTION_TEXT = {
  I0: '观察',
  I1: '温和提醒',
  I2: '主动引导',
  I3: '保护 / 限制',
  I4: '请求帮助',
};
const ALERT_TEXT = {
  NONE: '无网页警示',
  PAGE_WARNING: '网页警告',
  URGENT_HELP: '紧急网页求助',
};

function confidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function PredictionCard({
  title,
  value,
  detail,
  prediction,
}: {
  title: string;
  value: string;
  detail: string;
  prediction: Prediction<unknown>;
}) {
  return (
    <article className="model-output-card">
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
      <div className="confidence-line">
        <i style={{ width: confidence(prediction.confidence) }} />
      </div>
      <em>置信度 {confidence(prediction.confidence)}</em>
    </article>
  );
}

export default function TextDecisionPanel({
  onDecision,
}: {
  onDecision?: (text: string, result: DecisionResult) => void;
}) {
  const [text, setText] = useState(EXAMPLES.明确危险);
  const [result, setResult] = useState<DecisionResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [integrationNote, setIntegrationNote] = useState('');
  const requestRef = useRef<AbortController | null>(null);

  async function run() {
    const value = text.trim();
    if (value.length < 10 || loading) return;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);
    setError('');
    setIntegrationNote('');
    try {
      const decision = await requestTextDecision(value, controller.signal);
      setResult(decision);
      if (
        decision.guarded_output.risk_level === 'L0' &&
        decision.guarded_output.intervention_level === 'I0'
      ) {
        setIntegrationNote('AI判断为 L0 / I0，本次仅保留页面结果，不创建待处理事件。');
      } else {
        onDecision?.(value, decision);
        setIntegrationNote('已按统一人工事件格式加入事件中心。');
      }
    } catch (reason) {
      if (!controller.signal.aborted)
        setError(reason instanceof Error ? reason.message : '文字决策失败。');
    } finally {
      if (requestRef.current === controller) setLoading(false);
    }
  }

  const output = result?.model_output;
  const guarded = result?.guarded_output;
  return (
    <section className="panel text-decision-panel" id="ai-decision">
      <div className="panel-heading">
        <div>
          <h2>AI 文字事件决策实验</h2>
          <p>真实加载 V3.1 · MacBERT + Event Transformer，不读取标签字段</p>
        </div>
        <Badge variant="outline">本机模型</Badge>
      </div>
      <div className="text-decision-grid">
        <div className="decision-input-column">
          <div className="example-row" aria-label="文字决策示例">
            {Object.entries(EXAMPLES).map(([label, value]) => (
              <Button
                type="button"
                size="sm"
                variant="outline"
                key={label}
                disabled={loading}
                onClick={() => {
                  setText(value);
                  setResult(null);
                  setError('');
                  setIntegrationNote('');
                }}
              >
                {label}
              </Button>
            ))}
          </div>
          <label htmlFor="decision-text">场景、事件、能力、回应与历史干预</label>
          <textarea
            id="decision-text"
            maxLength={6000}
            value={text}
            disabled={loading}
            onChange={(event) => setText(event.target.value)}
            spellCheck={false}
          />
          <div className="decision-submit-row">
            <small>{text.length} / 6000 字符 · 每条事件建议单独一行</small>
            <Button onClick={run} disabled={loading || text.trim().length < 10}>
              {loading ? (
                <LoaderCircle className="model-spinner" size={15} />
              ) : (
                <Play size={14} />
              )}
              {loading ? '模型推理中' : '运行真实模型'}
            </Button>
          </div>
          <p className="privacy-note">
            文字只发送到 127.0.0.1 本机服务；事件中心仅在本次页面会话保留摘要。
            请勿输入真实姓名、联系方式或病历。
          </p>
        </div>
        <div className="decision-result-column" aria-live="polite">
          {!result && !error && (
            <div className="model-empty">
              <BrainCircuit size={30} />
              <h3>等待文字事件</h3>
              <p>首次运行需要把 MacBERT 和增量权重载入内存，后续请求会更快。</p>
            </div>
          )}
          {error && (
            <div className="model-error">
              <AlertTriangle size={20} />
              <div>
                <strong>模型未返回结果</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
          {result && output && guarded && (
            <>
              <div className="model-meta">
                <span>{result.run_name}</span>
                <span>{result.device.toUpperCase()}</span>
                <span>{result.inference_ms.toFixed(1)} ms</span>
                <span>{result.segment_count} 个文字段</span>
              </div>
              <div className="model-output-grid">
                <PredictionCard
                  title="风险等级"
                  value={String(output.risk_level.label)}
                  detail="事件语义辅助任务"
                  prediction={output.risk_level}
                />
                <PredictionCard
                  title="干预等级"
                  value={String(output.intervention_level.label)}
                  detail={INTERVENTION_TEXT[output.intervention_level.label]}
                  prediction={output.intervention_level}
                />
                <PredictionCard
                  title="警示方式"
                  value={String(output.alert_mode.label)}
                  detail={ALERT_TEXT[output.alert_mode.label]}
                  prediction={output.alert_mode}
                />
                <PredictionCard
                  title="人工核查"
                  value={output.manual_review.label ? '需要' : '暂不需要'}
                  detail={output.abstain.label ? '模型拒绝自动定案' : '模型已给出分类'}
                  prediction={output.manual_review}
                />
              </div>
              <div
                className={`guarded-result ${result.guardrail_applied ? 'guarded-attention' : ''}`}
              >
                {result.guardrail_applied ? (
                  <AlertTriangle size={18} />
                ) : (
                  <CheckCircle2 size={18} />
                )}
                <div>
                  <strong>
                    最终演示输出：{guarded.risk_level} ·{' '}
                    {guarded.intervention_level} · {guarded.alert_mode}
                  </strong>
                  <p>
                    {result.review_reasons.length
                      ? result.review_reasons.join('；')
                      : '多任务输出满足当前一致性检查，未追加安全校验。'}
                  </p>
                </div>
              </div>
              {integrationNote && (
                <p className="model-integration-note">{integrationNote}</p>
              )}
              <details className="model-probabilities">
                <summary>查看各类别概率与模型边界</summary>
                {Object.entries({
                  风险: output.risk_level.probabilities,
                  干预: output.intervention_level.probabilities,
                  警示: output.alert_mode.probabilities,
                }).map(([name, values]) => (
                  <div key={name}>
                    <strong>{name}</strong>
                    <span>
                      {Object.entries(values)
                        .map(([label, value]) => `${label} ${confidence(value)}`)
                        .join(' · ')}
                    </span>
                  </div>
                ))}
              </details>
              <p className="model-limitations">{result.limitations}</p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
