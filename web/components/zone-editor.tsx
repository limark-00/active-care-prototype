'use client';
import { useRef, useState } from 'react';
import type { PointerEvent, ReactNode } from 'react';
import { Crosshair, Pencil, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import { freshObservation, normalizeRect, validRect } from '@/lib/zones';
import type { Rect, ZoneAction, ZoneState } from '@/lib/zones';
import type { VisionFrame } from '@/lib/vision';
import { displayRect, displayX } from '@/lib/vision-runtime';

export default function ZoneEditor({
  state,
  onAction,
  now,
  frame,
  flipped,
  showObservation,
  children,
}: {
  state: ZoneState;
  onAction: (action: ZoneAction) => void;
  now: () => number;
  frame: VisionFrame | null;
  flipped: boolean;
  showObservation: boolean;
  children: ReactNode;
}) {
  const [editing, setEditing] = useState(false);
  const [rect, setRect] = useState<Rect>([0.25, 0.45, 0.75, 0.95]);
  const [name, setName] = useState('关注区域');
  const [seconds, setSeconds] = useState(10);
  const drag = useRef<{ id: number; start: [number, number] } | null>(null);
  const fresh = freshObservation(state.observation, state.lastNow);
  const selected = state.selected;
  const people = fresh ? state.observation!.frame.persons : [];
  const selectedPerson = people.find(
    (p) =>
      p.id === selected?.personId &&
      state.observation?.sessionId === selected?.sessionId,
  );
  const activeDraft = editing && !state.armed && !!frame && fresh;
  function point(e: PointerEvent<SVGSVGElement>): [number, number] {
    const box = e.currentTarget.getBoundingClientRect();
    return [
      displayX(
        Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)),
        flipped,
      ),
      Math.max(0, Math.min(1, (e.clientY - box.top) / box.height)),
    ];
  }
  function draw(e: PointerEvent<SVGSVGElement>) {
    if (!activeDraft || drag.current?.id !== e.pointerId) return;
    setRect(normalizeRect(drag.current.start, point(e)));
  }
  function save() {
    onAction({ type: 'add', rect, name, dwellMs: seconds * 1000, now: now() });
    if (
      validRect(rect) &&
      name.trim() &&
      seconds >= 5 &&
      seconds <= 120 &&
      state.zones.length < 3 &&
      fresh
    )
      setEditing(false);
  }
  const w = frame?.width ?? 640,
    h = frame?.height ?? 480;
  const draft = displayRect(rect, flipped);
  const foot =
    selectedPerson &&
    selectedPerson.keypoints[15].confidence >= 0.45 &&
    selectedPerson.keypoints[16].confidence >= 0.45
      ? [
          (selectedPerson.keypoints[15].x + selectedPerson.keypoints[16].x) / 2,
          (selectedPerson.keypoints[15].y + selectedPerson.keypoints[16].y) / 2,
        ]
      : null;
  return (
    <div className="zone-workspace">
      <div className="vision-stage" style={{ aspectRatio: `${w}/${h}` }}>
        {children}
        {frame && (
          <svg
            className={`zone-overlay ${activeDraft ? 'zone-draw-mode' : ''}`}
            viewBox={`0 0 ${w} ${h}`}
            aria-label={
              activeDraft
                ? '用指针拖动画出地面区域；也可在下方输入边界百分比。'
                : '人工划定的地面区域和所选对象脚踝位置'
            }
            onPointerDown={(e) => {
              if (!activeDraft) return;
              e.preventDefault();
              drag.current = { id: e.pointerId, start: point(e) };
              e.currentTarget.setPointerCapture(e.pointerId);
              setRect(normalizeRect(point(e), point(e)));
            }}
            onPointerMove={draw}
            onPointerUp={(e) => {
              draw(e);
              if (drag.current?.id === e.pointerId) {
                drag.current = null;
                e.currentTarget.releasePointerCapture(e.pointerId);
              }
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
          >
            {state.zones.map((zone) => {
              const row = state.statuses.find((r) => r.zoneId === zone.id);
              const bounds = displayRect(zone.rect, flipped);
              return (
                <g
                  key={zone.id}
                  className={
                    showObservation && state.armed && row?.position === 'inside'
                      ? 'zone-inside'
                      : 'zone-normal'
                  }
                >
                  <rect
                    x={bounds[0] * w}
                    y={bounds[1] * h}
                    width={(zone.rect[2] - zone.rect[0]) * w}
                    height={(zone.rect[3] - zone.rect[1]) * h}
                  />
                  <text
                    x={bounds[0] * w + 5}
                    y={Math.max(15, zone.rect[1] * h + 16)}
                  >
                    {zone.id} · {zone.name}
                  </text>
                </g>
              );
            })}
            {activeDraft && rect.every(Number.isFinite) && (
              <rect
                className="zone-draft"
                x={draft[0] * w}
                y={draft[1] * h}
                width={Math.max(0, rect[2] - rect[0]) * w}
                height={Math.max(0, rect[3] - rect[1]) * h}
              />
            )}
            {foot && showObservation && (
              <g className="zone-foot">
                <circle
                  cx={displayX(foot[0], flipped) * w}
                  cy={foot[1] * h}
                  r={6}
                />
                <text
                  x={displayX(foot[0], flipped) * w + 10}
                  y={foot[1] * h - 8}
                >
                  P{selected!.personId} 观察点
                </text>
              </g>
            )}
          </svg>
        )}
      </div>
      <div className="zone-controls">
        <div className="zone-heading">
          <h3>
            <Crosshair size={16} />
            区域与观察对象
          </h3>
          <span>{state.armed ? '监测已开启' : '监测暂停'}</span>
        </div>
        <div className="zone-target">
          <label htmlFor="zone-target">观察对象（人工选择）</label>
          <NativeSelect
            id="zone-target"
            className="full-select"
            value={selected?.personId ?? ''}
            disabled={!fresh}
            onChange={(e) =>
              onAction({
                type: 'select',
                personId: e.target.value ? Number(e.target.value) : null,
                now: now(),
              })
            }
          >
            <NativeSelectOption value="">请选择，不自动绑定</NativeSelectOption>
            {selected && !selectedPerson && (
              <NativeSelectOption value={selected.personId} disabled>
                P{selected.personId}（对象丢失 / 过期）
              </NativeSelectOption>
            )}
            {people.map((p) => (
              <NativeSelectOption value={p.id} key={p.id}>
                人物 P{p.id}
              </NativeSelectOption>
            ))}
          </NativeSelect>
          <Button
            size="sm"
            variant={state.armed ? 'outline' : 'default'}
            onClick={() => {
              setEditing(false);
              onAction({ type: 'arm', enabled: !state.armed, now: now() });
            }}
          >
            {state.armed ? '暂停区域监测' : '开始区域监测'}
          </Button>
        </div>
        <p className="zone-hint">
          用脚踝中点判断地面位置。请圈出灶台前、门口等地面范围；没有真实距离标定，不报告米数。
        </p>
        <div className="zone-tool-buttons">
          <Button
            size="sm"
            variant="outline"
            disabled={state.armed || !fresh || state.zones.length >= 3}
            onClick={() => setEditing((v) => !v)}
          >
            <Pencil size={13} />
            {editing ? '取消划区' : '新增地面区域'}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            disabled={state.armed || !fresh || !state.zones.length}
            onClick={() => onAction({ type: 'calibrate', now: now() })}
          >
            确认当前机位与区域
          </Button>
        </div>
        {editing && !state.armed && (
          <fieldset className="zone-form">
            <legend>拖动上方画面框选，或填写边界百分比</legend>
            <div className="zone-name-fields">
              <label htmlFor="zone-name">
                区域名称
                <Input
                  id="zone-name"
                  value={name}
                  maxLength={24}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="如：灶台前地面"
                />
              </label>
              <label htmlFor="zone-seconds">
                停留阈值（秒）
                <Input
                  id="zone-seconds"
                  type="number"
                  min={5}
                  max={120}
                  value={Number.isFinite(seconds) ? seconds : ''}
                  onChange={(e) =>
                    setSeconds(
                      e.target.value === '' ? NaN : Number(e.target.value),
                    )
                  }
                />
              </label>
            </div>
            <div className="zone-coordinates">
              {(['左边界 %', '上边界 %', '右边界 %', '下边界 %'] as const).map(
                (label, i) => (
                  <label key={label} htmlFor={`zone-coordinate-${i}`}>
                    {label}
                    <Input
                      id={`zone-coordinate-${i}`}
                      type="number"
                      min={0}
                      max={100}
                      step={1}
                      value={
                        Number.isFinite(draft[i])
                          ? Math.round(draft[i] * 100)
                          : ''
                      }
                      onChange={(e) =>
                        setRect((old) => {
                          const next = displayRect(old, flipped);
                          next[i] =
                            e.target.value === ''
                              ? NaN
                              : Number(e.target.value) / 100;
                          return displayRect(next, flipped);
                        })
                      }
                    />
                  </label>
                ),
              )}
            </div>
            <Button
              size="sm"
              disabled={
                !fresh ||
                !validRect(rect) ||
                !name.trim() ||
                !Number.isFinite(seconds) ||
                seconds < 5 ||
                seconds > 120
              }
              onClick={save}
            >
              保存区域
            </Button>
            <small>
              矩形至少占画面宽、高各 6%；最多 3 个区域。保存后再开始监测。
            </small>
          </fieldset>
        )}
        <output className="zone-notice">{state.notice}</output>
        {!state.calibrated && state.zones.length > 0 && (
          <p className="zone-calibration-warning">
            机位未确认：摄像头重启或尺寸改变后，请核对区域位置并重新确认。
          </p>
        )}
        <div className="zone-list">
          {state.zones.map((zone) => {
            const row = state.statuses.find((r) => r.zoneId === zone.id);
            const position = state.armed && fresh ? row?.position : 'unknown';
            return (
              <article key={zone.id}>
                <div>
                  <strong>
                    {zone.id} · {zone.name}
                  </strong>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    disabled={state.armed}
                    aria-label={`删除${zone.name}`}
                    onClick={() =>
                      onAction({ type: 'delete', id: zone.id, now: now() })
                    }
                  >
                    <Trash2 size={13} />
                  </Button>
                </div>
                <div className="zone-row-status">
                  <span>
                    {position === 'inside'
                      ? '区域内'
                      : position === 'outside'
                        ? '区域外'
                        : '未知 / 暂停'}
                  </span>
                  <b>
                    {position === 'inside'
                      ? `${((row?.dwellMs ?? 0) / 1000).toFixed(1)}s`
                      : '—'}{' '}
                    <small>/ {zone.dwellMs / 1000}s</small>
                  </b>
                </div>
                <progress
                  max={zone.dwellMs}
                  value={
                    position === 'inside'
                      ? Math.min(row?.dwellMs ?? 0, zone.dwellMs)
                      : 0
                  }
                  aria-label={`${zone.name}连续观测停留时间`}
                />
                <p>
                  {position === 'inside' && (row?.dwellMs ?? 0) >= zone.dwellMs
                    ? '已达阈值：请查看事件中心。'
                    : row?.reason ||
                      '只累计相邻有效观测，不用等待响应的时间凑计时。'}
                </p>
              </article>
            );
          })}
        </div>
        {!state.zones.length && (
          <p className="zone-hint">
            尚未划定区域。开启摄像头后，新增区域并选择要观察的人。
          </p>
        )}
        <details className="zone-history">
          <summary>区域进入 / 离开记录（{state.logs.length}）</summary>
          {state.logs.length ? (
            <ol>
              {state.logs.map((log) => (
                <li key={log.id}>
                  <time>
                    {new Date(log.at).toLocaleTimeString('zh-CN', {
                      hour12: false,
                    })}
                  </time>
                  {log.message}
                </li>
              ))}
            </ol>
          ) : (
            <p>暂无区域记录。</p>
          )}
        </details>
        <p className="zone-hint">
          移动电脑后请暂停监测并重新划区。目标换号时不会自动换人；暂停、换人、删区域都不会清除旧警报。
        </p>
      </div>
    </div>
  );
}
