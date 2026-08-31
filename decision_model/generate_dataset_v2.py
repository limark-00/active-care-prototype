#!/usr/bin/env python3
"""Generate V2 of the synthetic Chinese multi-event decision dataset.

V2 is intentionally synthetic. It is designed to test a supervised learning and
UI pipeline, not to establish medical, fire-safety, or care policy validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .generate_dataset import (
    CAPABILITIES,
    INTERVENTION_ORDER,
    PREVIOUS as V1_PREVIOUS,
    RESPONSES,
    RISK_ORDER,
    SCENES as V1_SCENES,
)

SEED = 20260830
POLICY_VERSION = "synthetic-care-policy-v2"
DATASET_NAME = "active-care-synthetic-text-decisions-v2"
ALERT_MODES = ("NONE", "PAGE_WARNING", "URGENT_HELP")
SOURCE_LABELS = {
    "camera": "摄像头事件",
    "sensor": "模拟传感器事件",
    "manual": "人工输入事件",
    "system": "系统状态事件",
}
PREVIOUS = {
    **V1_PREVIOUS,
    "I1_presented": ("上一次I1温和提醒已经呈现，效果需要结合当前回应判断", "pending"),
    "I2_presented": ("上一次I2图示引导已经呈现，效果需要结合当前回应判断", "pending"),
}
CONTEXT_RESPONSES = {
    "none": ("not_requested", "unknown", "requested_help"),
    "I0_effective": ("not_requested", "unknown", "requested_help"),
    "I1_effective": ("responded", "changed_behavior"),
    "I1_ineffective": ("no_response", "declined", "unknown", "requested_help"),
    "I1_presented": ("responded", "changed_behavior", "no_response", "declined", "requested_help", "unknown"),
    "I2_effective": ("responded", "changed_behavior"),
    "I2_ineffective": ("no_response", "declined", "unknown", "requested_help"),
    "I2_presented": ("responded", "changed_behavior", "no_response", "declined", "requested_help", "unknown"),
    "I2_failed": ("unknown", "requested_help"),
    "I3_unavailable": ("unknown", "requested_help"),
    "I4_pending": ("no_response", "unknown", "responded"),
}


@dataclass(frozen=True)
class EventV2:
    id: str
    risk: str
    minimum_intervention: str
    source: str
    channel: str
    tags: tuple[str, ...]
    phrases: tuple[str, ...]
    ood_only: bool = False


def _rewrite(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    value = text
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def phrase_variants(text: str, scene_name: str) -> tuple[str, ...]:
    second = _rewrite(
        text,
        (("患者", "使用者"), ("持续", "连续"), ("出现", "发生"), ("没有", "未"), ("处于", "目前为")),
    )
    third = _rewrite(
        text,
        (("患者", "照护对象"), ("短时间内", "在较短时间里"), ("多次", "反复"), ("保持", "维持"), ("正常", "符合日常状态")),
    )
    fourth = _rewrite(
        text,
        (("患者", "当前人员"), ("正在", "正处于"), ("接近", "靠近"), ("离开", "走出"), ("输入", "信号")),
    )
    values = (
        text,
        f"{scene_name}的记录显示，{second}",
        f"监测人员描述：{third}",
        f"系统汇总到一项情况：{fourth}",
        f"本次场景信息表明，{_rewrite(second, (('系统', '监测模块'), ('区域', '范围')))}",
    )
    if len(set(values)) != 5:
        raise ValueError(f"phrases are not unique: {text}")
    return values


def V(
    id: str,
    risk: str,
    intervention: str,
    text: str,
    source: str,
    channel: str,
    *tags: str,
    scene_name: str,
    ood_only: bool = False,
) -> EventV2:
    return EventV2(
        id=id,
        risk=risk,
        minimum_intervention=intervention,
        source=source,
        channel=channel,
        tags=tuple(tags),
        phrases=phrase_variants(text, scene_name),
        ood_only=ood_only,
    )


EXTRA_SPECS: dict[str, list[tuple]] = {
    "kitchen": [
        ("k19", "L0", "I0", "电热水壶已经断电并放回安全位置", "sensor", "appliance", ("normal",)),
        ("k20", "L1", "I0", "冰箱门短暂保持打开，当前先继续观察", "sensor", "fridge", ("observe",)),
        ("k21", "L1", "I1", "患者准备餐食时多次停下并重新确认步骤", "camera", "task", ("repetition",)),
        ("k22", "L2", "I1", "烤箱预热时间超过当前演示设定", "sensor", "oven", ("dwell",)),
        ("k23", "L2", "I2", "患者的手持续靠近已标记的刀具区域", "camera", "sharp_tool", ("hazard",)),
        ("k24", "L3", "I3", "小型加热设备温度异常且仍保持通电", "sensor", "appliance", ("protection",)),
        ("k25", "L3", "I4", "患者在水槽附近突然失去平衡并保持低位", "camera", "posture", ("fall",)),
        ("k26", "L0", "I0", "患者完成备餐并将常用工具归位", "camera", "task", ("normal", "ood")),
        ("k27", "L1", "I1", "冷藏设备的门长时间未完全关闭", "sensor", "fridge", ("environment", "ood")),
        ("k28", "L2", "I2", "患者反复拿起并放下锋利厨具", "camera", "sharp_tool", ("hazard", "ood")),
        ("k29", "L3", "I3", "烤箱持续高温运行且现场无人照看", "sensor", "oven", ("protection", "ood")),
        ("k30", "L4", "I4", "一氧化碳报警输入被触发", "sensor", "air_alarm", ("urgent", "ood")),
    ],
    "bathroom": [
        ("b19", "L0", "I0", "患者完成洗手并关闭水龙头", "camera", "task", ("normal",)),
        ("b20", "L1", "I0", "患者扶住洗手台短暂停顿后继续移动", "camera", "movement", ("observe",)),
        ("b21", "L1", "I1", "卫生间湿度持续偏离演示范围", "sensor", "humidity", ("environment",)),
        ("b22", "L2", "I1", "浴缸水位持续升高但仍低于紧急阈值", "sensor", "water_level", ("dwell",)),
        ("b23", "L2", "I2", "患者尝试在没有支撑的位置转身", "camera", "posture", ("instability",)),
        ("b24", "L3", "I3", "通电电器被放置在靠近积水的位置", "manual", "electric", ("protection",)),
        ("b25", "L3", "I4", "患者在淋浴区域出现突然下降并持续低位", "camera", "posture", ("fall",)),
        ("b26", "L0", "I0", "患者取用毛巾后保持稳定站立", "camera", "task", ("normal", "ood")),
        ("b27", "L1", "I1", "排风设备停止工作且室内湿度继续升高", "sensor", "humidity", ("environment", "ood")),
        ("b28", "L2", "I2", "浴缸水位接近预设上限且患者仍在加水", "sensor", "water_level", ("hazard", "ood")),
        ("b29", "L3", "I3", "患者将通电设备移动到水源附近", "camera", "electric", ("protection", "ood")),
        ("b30", "L4", "I4", "卫生间紧急呼叫按钮被人工触发", "manual", "help", ("urgent", "ood")),
    ],
    "bedroom": [
        ("d19", "L0", "I0", "患者起床后正常整理床铺", "camera", "task", ("normal",)),
        ("d20", "L1", "I0", "患者坐在床边寻找个人物品", "camera", "activity", ("observe",)),
        ("d21", "L1", "I1", "卧室湿度持续低于演示舒适范围", "sensor", "humidity", ("environment",)),
        ("d22", "L2", "I1", "患者在非日常时段反复开关床头灯", "sensor", "light", ("repetition",)),
        ("d23", "L2", "I2", "患者离床后走向与日常路线不同的区域", "camera", "movement", ("hazard",)),
        ("d24", "L3", "I3", "电热毯长时间运行且温度持续升高", "sensor", "heater", ("protection",)),
        ("d25", "L3", "I4", "患者在床和墙之间保持异常低位姿态", "camera", "posture", ("fall",)),
        ("d26", "L0", "I0", "患者拉开窗帘后返回床边安全区域", "camera", "task", ("normal", "ood")),
        ("d27", "L1", "I1", "加湿设备缺水并停止运行", "sensor", "humidity", ("environment", "ood")),
        ("d28", "L2", "I2", "患者夜间起身后持续触碰陌生物品", "camera", "activity", ("hazard", "ood")),
        ("d29", "L3", "I3", "患者借助不稳定物品试图靠近高处窗户", "camera", "opening", ("protection", "ood")),
        ("d30", "L4", "I4", "卧室一氧化碳报警输入被触发", "sensor", "air_alarm", ("urgent", "ood")),
    ],
    "living_room": [
        ("l19", "L0", "I0", "患者在座椅上正常阅读并能稳定起身", "camera", "activity", ("normal",)),
        ("l20", "L1", "I0", "患者在家具之间短暂停顿并查看周围", "camera", "movement", ("observe",)),
        ("l21", "L1", "I1", "客厅主灯长时间保持开启且已进入休息时段", "sensor", "light", ("environment",)),
        ("l22", "L2", "I1", "患者在同一储物柜前反复停留", "camera", "dwell", ("repetition",)),
        ("l23", "L2", "I2", "患者重复打开药盒但没有服药确认记录", "manual", "medicine", ("hazard",)),
        ("l24", "L3", "I3", "大型家具出现倾斜且患者正在靠近", "manual", "furniture", ("protection",)),
        ("l25", "L3", "I4", "患者从座椅旁突然下降并持续没有明显位移", "camera", "posture", ("fall",)),
        ("l26", "L0", "I0", "患者完成阅读后将物品放回原处", "camera", "task", ("normal", "ood")),
        ("l27", "L1", "I1", "落地灯持续闪烁并影响当前照明", "sensor", "light", ("environment", "ood")),
        ("l28", "L2", "I2", "患者短时间内多次取出同一药盒", "manual", "medicine", ("hazard", "ood")),
        ("l29", "L3", "I3", "患者正在拉动已经不稳定的高层置物架", "camera", "furniture", ("protection", "ood")),
        ("l30", "L4", "I4", "客厅一氧化碳报警输入被触发", "sensor", "air_alarm", ("urgent", "ood")),
    ],
    "entrance": [
        ("e19", "L0", "I0", "患者在门内整理随身物品后返回室内", "camera", "task", ("normal",)),
        ("e20", "L1", "I0", "患者在门边停下并查看门外声音来源", "camera", "activity", ("observe",)),
        ("e21", "L1", "I1", "门口照明不足并影响边界识别", "system", "light", ("unknown",)),
        ("e22", "L2", "I1", "患者在非预定时段开始整理外出物品", "camera", "task", ("hazard",)),
        ("e23", "L2", "I2", "患者反复尝试识别门外陌生人员", "camera", "visitor", ("hazard",)),
        ("e24", "L3", "I3", "门锁持续异常动作且无法保持预设关闭状态", "sensor", "door_lock", ("protection",)),
        ("e25", "L3", "I4", "患者在门外台阶区域出现倒地候选姿态", "camera", "posture", ("fall",)),
        ("e26", "L0", "I0", "患者正常领取物品后关闭入户门", "camera", "task", ("normal", "ood")),
        ("e27", "L1", "I1", "公共走廊照明失效，门外区域难以看清", "manual", "light", ("environment", "ood")),
        ("e28", "L2", "I2", "患者在夜间反复准备鞋帽并靠近门口", "camera", "task", ("hazard", "ood")),
        ("e29", "L3", "I3", "检测到未经授权的持续开锁尝试", "sensor", "door_lock", ("protection", "ood")),
        ("e30", "L4", "I4", "门口紧急求助按钮被触发", "manual", "help", ("urgent", "ood")),
    ],
    "balcony": [
        ("a19", "L0", "I0", "患者正常晾晒衣物并停留在安全区域", "camera", "task", ("normal",)),
        ("a20", "L1", "I0", "患者短暂移动花盆但没有接近边界", "camera", "activity", ("observe",)),
        ("a21", "L1", "I1", "阳台风速提示高于日常观察范围", "sensor", "weather", ("environment",)),
        ("a22", "L2", "I1", "浇水后形成的积水范围持续扩大", "manual", "floor", ("environment",)),
        ("a23", "L2", "I2", "患者在强风状态下继续靠近晾晒区域边缘", "camera", "boundary", ("hazard",)),
        ("a24", "L3", "I3", "折叠梯没有完全展开且患者正在攀爬", "camera", "ladder", ("protection",)),
        ("a25", "L3", "I4", "患者在阳台门槛外侧出现倒地候选姿态", "camera", "posture", ("fall",)),
        ("a26", "L0", "I0", "患者收回衣物后正常返回室内", "camera", "task", ("normal", "ood")),
        ("a27", "L1", "I1", "室外风力继续增大并影响阳台物品稳定", "sensor", "weather", ("environment", "ood")),
        ("a28", "L2", "I2", "花盆倾倒后患者在湿滑区域继续移动", "camera", "floor", ("hazard", "ood")),
        ("a29", "L3", "I3", "患者站在未固定的梯具上接近阳台边缘", "camera", "ladder", ("protection", "ood")),
        ("a30", "L4", "I4", "人工确认阳台栏杆出现严重结构异常", "manual", "boundary", ("urgent", "ood")),
    ],
}


def build_scenes() -> dict[str, dict]:
    scenes: dict[str, dict] = {}
    for scene_id, old in V1_SCENES.items():
        name = old["name"]
        events = [
            V(e.id, e.risk, e.minimum_intervention, e.text, e.source, e.channel, *e.tags, scene_name=name)
            for e in old["events"]
        ]
        for spec in EXTRA_SPECS[scene_id]:
            event_id, risk, intervention, text, source, channel, tags = spec
            events.append(
                V(
                    event_id,
                    risk,
                    intervention,
                    text,
                    source,
                    channel,
                    *tags,
                    scene_name=name,
                    ood_only="ood" in tags,
                )
            )
        if len(events) != 30 or sum(event.ood_only for event in events) != 5:
            raise AssertionError(f"invalid event catalog for {scene_id}")
        scenes[scene_id] = {"name": name, "events": events}
    return scenes


SCENES = build_scenes()


def previous_level(key: str) -> str:
    return "none" if key == "none" else key.split("_", 1)[0]


def decide(
    events: Iterable[EventV2],
    capability: str,
    response: str,
    previous: str,
    *,
    force_review: bool = False,
    conflict: bool = False,
) -> dict:
    events = list(events)
    risk = max((event.risk for event in events), key=RISK_ORDER.__getitem__)
    intervention = max(
        (event.minimum_intervention for event in events),
        key=INTERVENTION_ORDER.__getitem__,
    )
    reasons = [
        f"{event.id}:{event.risk}/{event.minimum_intervention}"
        for event in sorted(
            events,
            key=lambda e: (RISK_ORDER[e.risk], INTERVENTION_ORDER[e.minimum_intervention]),
            reverse=True,
        )[:3]
    ]
    previous_i = previous_level(previous)
    previous_result = PREVIOUS[previous][1]
    if risk == "L4":
        intervention = "I4"
        reasons.append("L4事件直接进入紧急求助分支")
    elif capability in {"visual_support", "limited"} and intervention == "I1":
        intervention = "I2"
        reasons.append("能力档案需要图示或更强支持")
    if response == "requested_help":
        intervention = "I4"
        reasons.append("患者或现场人员主动求助")
    elif response == "no_response":
        if risk == "L3" or previous_i in {"I2", "I3", "I4"}:
            intervention = "I4"
            reasons.append("较高风险或多次支持后仍无回应")
        elif previous_i == "I1" and INTERVENTION_ORDER[intervention] < 2:
            intervention = "I2"
            reasons.append("I1无回应后增加支持方式")
    if previous_result in {"failed", "unavailable"}:
        if risk == "L3" or previous_i in {"I2", "I3"}:
            intervention = "I4"
            reasons.append("既有输出失败且事件仍需处理")
        elif intervention == "I1":
            intervention = "I2"
            reasons.append("既有输出不可用，切换支持通道")
    if previous_i == "I4" and previous_result == "pending":
        intervention = "I4"
        reasons.append("已有求助仍待人工接手")

    manual_review = (
        force_review
        or capability == "unknown"
        or response == "unknown"
        or any("unknown" in event.tags for event in events)
    )
    if intervention == "I4" or risk == "L4":
        alert_mode = "URGENT_HELP"
    elif (
        intervention in {"I2", "I3"}
        or risk in {"L2", "L3"}
        or conflict
        or (manual_review and risk != "L0")
    ):
        alert_mode = "PAGE_WARNING"
    else:
        alert_mode = "NONE"
    abstain = conflict and intervention != "I4" and risk != "L4"
    if abstain:
        reasons.append("输入存在冲突，保留安全下限并请求人工核查")
    return {
        "derived_risk_level": risk,
        "intervention_level": intervention,
        "alert_mode": alert_mode,
        "manual_review": manual_review,
        "abstain": abstain,
        "reason_codes": reasons,
    }


def compatible_context(rng: random.Random) -> tuple[str, str, str]:
    previous = rng.choice(list(PREVIOUS))
    response = rng.choice(CONTEXT_RESPONSES[previous])
    capability = rng.choice(list(CAPABILITIES))
    return capability, response, previous


def render_input(group: dict, variant: int) -> str:
    events = list(group["events"])
    if variant == 1:
        events.reverse()
    elif variant == 2 and len(events) > 2:
        events = events[1:] + events[:1]
    elif variant == 3:
        events = sorted(events, key=lambda e: e.source)
    elif variant == 4:
        events = sorted(events, key=lambda e: e.risk, reverse=True)
    headings = (
        f"当前场景：{group['scene_name']}。系统收到以下文字事件。",
        f"地点为{group['scene_name']}，请综合判断本次记录。",
        f"{group['scene_name']}场景的多源事件汇总如下。",
        f"需要处理一组来自{group['scene_name']}的文字词条。",
        f"本次决策对象位于{group['scene_name']}，现有信息如下。",
    )
    lines = [headings[variant]]
    for index, event in enumerate(events, 1):
        lines.append(
            f"词条{index}（{event.risk}，{SOURCE_LABELS[event.source]}）：{event.phrases[variant]}。"
        )
    if group.get("conflict_note"):
        lines.append(group["conflict_note"])
    if group.get("hard_note"):
        lines.append(group["hard_note"])
    lines.extend(
        (
            f"患者能力：{CAPABILITIES[group['capability']]}。",
            f"患者响应：{RESPONSES[group['response']]}。",
            f"前次干预：{PREVIOUS[group['previous']][0]}。",
            "请判断下一步干预、警示方式，以及是否需要人工核查或拒绝自动判断。",
        )
    )
    return "\n".join(lines)


def signature(events: list[EventV2], capability: str, response: str, previous: str, note: str = "") -> tuple:
    return tuple(sorted(event.id for event in events)), capability, response, previous, note


def select_events(
    rng: random.Random,
    pool: list[EventV2],
    anchor: EventV2,
    target: str,
    minimum_count: int,
    maximum_count: int,
) -> list[EventV2]:
    wanted = rng.randint(minimum_count, maximum_count)
    selected = [anchor]
    channels = {anchor.channel}
    candidates = [
        event
        for event in pool
        if event.id != anchor.id
        and INTERVENTION_ORDER[event.minimum_intervention] <= INTERVENTION_ORDER[target]
        and (("normal" in event.tags) == ("normal" in anchor.tags))
    ]
    rng.shuffle(candidates)
    for event in candidates:
        if len(selected) >= wanted:
            break
        if event.channel in channels:
            continue
        selected.append(event)
        channels.add(event.channel)
    return selected


def make_group(
    *,
    group_id: str,
    scene_id: str,
    events: list[EventV2],
    capability: str,
    response: str,
    previous: str,
    case_type: str,
    pair_id: str | None = None,
    conflict_note: str = "",
    hard_note: str = "",
) -> dict:
    labels = decide(
        events,
        capability,
        response,
        previous,
        force_review=case_type == "conflict_unknown",
        conflict=bool(conflict_note),
    )
    sig = signature(events, capability, response, previous, conflict_note + hard_note)
    return {
        "case_group_id": group_id,
        "counterfactual_pair_id": pair_id,
        "case_type": case_type,
        "scene_id": scene_id,
        "scene_name": SCENES[scene_id]["name"],
        "events": events,
        "capability": capability,
        "response": response,
        "previous": previous,
        "conflict_note": conflict_note,
        "hard_note": hard_note,
        "signature": sig,
        "labels": labels,
    }


def counterfactual_groups(scene_id: str, rng: random.Random, used: set[tuple]) -> list[dict]:
    pool = [event for event in SCENES[scene_id]["events"] if not event.ood_only]
    definitions = (
        ("help-i0-i4", "I0", ("independent", "not_requested", "none"), ("independent", "requested_help", "none"), "I0", "I4", "response"),
        ("ability-i1-i2", "I1", ("independent", "not_requested", "none"), ("visual_support", "not_requested", "none"), "I1", "I2", "capability"),
        ("response-i1-i2", "I1", ("independent", "responded", "I1_presented"), ("independent", "no_response", "I1_presented"), "I1", "I2", "response"),
        ("help-i2-i4", "I2", ("independent", "responded", "I2_presented"), ("independent", "requested_help", "I2_presented"), "I2", "I4", "response"),
        ("help-i3-i4", "I3", ("independent", "not_requested", "none"), ("independent", "requested_help", "none"), "I3", "I4", "response"),
    )
    groups: list[dict] = []
    for family, anchor_level, context_a, context_b, expected_a, expected_b, changed in definitions:
        eligible = [
            event for event in pool
            if INTERVENTION_ORDER[event.minimum_intervention] <= INTERVENTION_ORDER[anchor_level]
        ]
        event_sets = []
        for size in range(1, 4):
            for candidate in combinations(eligible, size):
                if max(
                    (event.minimum_intervention for event in candidate),
                    key=INTERVENTION_ORDER.__getitem__,
                ) != anchor_level:
                    continue
                if len({"normal" in event.tags for event in candidate}) != 1:
                    continue
                event_sets.append(list(candidate))
        rng.shuffle(event_sets)
        accepted = []
        for events in event_sets:
            sig_a = signature(events, *context_a)
            sig_b = signature(events, *context_b)
            if sig_a in used or sig_b in used:
                continue
            accepted.append((events, sig_a, sig_b))
            if len(accepted) == 20:
                break
        if len(accepted) != 20:
            raise RuntimeError(f"insufficient counterfactual combinations for {scene_id}/{family}")
        for index, (events, sig_a, sig_b) in enumerate(accepted):
            pair_id = f"{scene_id}-cf-{family}-{index + 1:02d}"
            a = make_group(
                group_id=pair_id + "-a",
                scene_id=scene_id,
                events=events,
                capability=context_a[0],
                response=context_a[1],
                previous=context_a[2],
                case_type="counterfactual",
                pair_id=pair_id,
            )
            b = make_group(
                group_id=pair_id + "-b",
                scene_id=scene_id,
                events=events,
                capability=context_b[0],
                response=context_b[1],
                previous=context_b[2],
                case_type="counterfactual",
                pair_id=pair_id,
            )
            assert a["labels"]["intervention_level"] == expected_a
            assert b["labels"]["intervention_level"] == expected_b
            a["changed_field"] = changed
            b["changed_field"] = changed
            used.update((sig_a, sig_b))
            groups.extend((a, b))
    return groups


CONFLICT_NOTES = (
    "冲突信息：另一个较低可信度来源称其中一项事件可能已经解除，但时间和对象尚未核对。",
    "冲突信息：补充记录给出了相反状态，两个来源的采样时间仍待确认。",
    "冲突信息：摄像头描述与人工记录不一致，暂时不能认定事件已经解除。",
    "冲突信息：传感器状态与现场文字说明不一致，需要核对设备和对象。",
    "冲突信息：同一时段出现互相矛盾的记录，目前不能只采用较新的说法。",
    "冲突信息：一个来源报告异常仍在持续，另一个来源报告已经恢复。",
    "冲突信息：事件对象可能被错误关联，当前记录需要人工确认。",
    "冲突信息：两条记录的场景位置不完全一致，不能自动合并为已解除。",
    "冲突信息：补充数据的可信度较低，尚不足以推翻已有事件。",
    "冲突信息：输入顺序与时间戳存在疑点，需要保留现有安全下限。",
)


NON_COUNTERFACTUAL_ALLOCATION = {
    "regular": {"I0": 140, "I1": 120, "I2": 100, "I3": 140, "I4": 100},
    "conflict_unknown": {f"I{i}": 20 for i in range(5)},
    "hard": {f"I{i}": 20 for i in range(5)},
}


def generated_groups(
    scene_id: str,
    target: str,
    case_type: str,
    count: int,
    rng: random.Random,
    used: set[tuple],
    prefix: str,
    *,
    pool: list[EventV2] | None = None,
    required_anchor: EventV2 | None = None,
) -> list[dict]:
    pool = pool or [event for event in SCENES[scene_id]["events"] if not event.ood_only]
    anchors = [required_anchor] if required_anchor else [
        event for event in pool if event.minimum_intervention == target
    ]
    groups: list[dict] = []
    attempts = 0
    while len(groups) < count:
        attempts += 1
        if attempts > 300_000:
            raise RuntimeError(f"cannot generate {scene_id}/{target}/{case_type}")
        anchor = rng.choice(anchors)
        minimum, maximum = (4, 6) if case_type == "hard" else (1, 5)
        events = select_events(rng, pool, anchor, target, minimum, maximum)
        if case_type == "conflict_unknown":
            capability, response = "unknown", "unknown"
            previous = rng.choice(("none", "I0_effective", "I1_presented", "I2_presented"))
            conflict_note = rng.choice(CONFLICT_NOTES)
            hard_note = ""
        else:
            capability, response, previous = compatible_context(rng)
            conflict_note = ""
            hard_note = (
                (
                    "补充说明：多项低风险记录发生时间接近，需要综合检查而不是只依赖单一词条。"
                    if target == "I0"
                    else "补充说明：多项事件发生时间接近，不能只依据最后一条记录覆盖其他仍未解除的证据。"
                )
                if case_type == "hard"
                else ""
            )
        group = make_group(
            group_id=f"{scene_id}-{prefix}-{target.lower()}-{len(groups) + 1:04d}",
            scene_id=scene_id,
            events=events,
            capability=capability,
            response=response,
            previous=previous,
            case_type=case_type,
            conflict_note=conflict_note,
            hard_note=hard_note,
        )
        if group["labels"]["intervention_level"] != target or group["signature"] in used:
            continue
        used.add(group["signature"])
        groups.append(group)
    return groups


def hashed_order(values: list[str], salt: str) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{SEED}:{salt}:{value}".encode()).hexdigest())


def assign_primary_splits(groups: list[dict]) -> None:
    for scene_id in SCENES:
        pair_groups = [g for g in groups if g["scene_id"] == scene_id and g["case_type"] == "counterfactual"]
        pair_ids = sorted({g["counterfactual_pair_id"] for g in pair_groups})
        families: dict[str, list[str]] = {}
        for pair_id in pair_ids:
            family = pair_id.rsplit("-", 1)[0]
            families.setdefault(family, []).append(pair_id)
        for family, ids in families.items():
            ordered = hashed_order(ids, family)
            mapping = {value: "train" if i < 14 else "validation" if i < 17 else "test" for i, value in enumerate(ordered)}
            for group in pair_groups:
                if group["counterfactual_pair_id"] in mapping:
                    group["split"] = mapping[group["counterfactual_pair_id"]]
        for case_type, allocation in NON_COUNTERFACTUAL_ALLOCATION.items():
            for target, count in allocation.items():
                subset = [
                    g for g in groups
                    if g["scene_id"] == scene_id
                    and g["case_type"] == case_type
                    and g["labels"]["intervention_level"] == target
                ]
                assert len(subset) == count
                ordered_ids = hashed_order([g["case_group_id"] for g in subset], f"{scene_id}:{case_type}:{target}")
                train_end = count * 70 // 100
                validation_end = train_end + count * 15 // 100
                mapping = {
                    value: "train" if i < train_end else "validation" if i < validation_end else "test"
                    for i, value in enumerate(ordered_ids)
                }
                for group in subset:
                    group["split"] = mapping[group["case_group_id"]]
    if any("split" not in group for group in groups):
        raise AssertionError("unassigned primary split")


def serialize(group: dict, variant: int, split: str) -> dict:
    sig_text = json.dumps(group["signature"], ensure_ascii=False, separators=(",", ":"))
    row = {
        "sample_id": f"{group['case_group_id']}-v{variant + 1}",
        "case_group_id": group["case_group_id"],
        "counterfactual_pair_id": group.get("counterfactual_pair_id"),
        "changed_field": group.get("changed_field"),
        "variant_id": variant + 1,
        "split": split,
        "case_type": group["case_type"],
        "synthetic": True,
        "policy_version": POLICY_VERSION,
        "composition_hash": hashlib.sha256(sig_text.encode()).hexdigest()[:20],
        "scene_id": group["scene_id"],
        "scene_name": group["scene_name"],
        "input_text": render_input(group, variant),
        "event_ids": [event.id for event in group["events"]],
        "event_levels": [event.risk for event in group["events"]],
        "capability_code": group["capability"],
        "response_code": group["response"],
        "previous_code": group["previous"],
        **group["labels"],
    }
    return row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def count_by(rows: list[dict], field: str) -> dict:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    used: set[tuple] = set()
    primary_groups: list[dict] = []
    for scene_id in SCENES:
        primary_groups.extend(counterfactual_groups(scene_id, rng, used))
        for case_type, allocation in NON_COUNTERFACTUAL_ALLOCATION.items():
            for target, count in allocation.items():
                primary_groups.extend(
                    generated_groups(scene_id, target, case_type, count, rng, used, case_type)
                )
    assign_primary_splits(primary_groups)
    primary_rows = {"train": [], "validation": [], "test": []}
    for group in primary_groups:
        for variant in range(5):
            primary_rows[group["split"]].append(serialize(group, variant, group["split"]))
    for split, rows in primary_rows.items():
        write_jsonl(output / f"{split}.jsonl", rows)

    ood_rows: list[dict] = []
    for scene_id in SCENES:
        pool = SCENES[scene_id]["events"]
        for target in INTERVENTION_ORDER:
            anchor = next(
                event for event in pool
                if event.ood_only and event.minimum_intervention == target
            )
            groups = generated_groups(
                scene_id,
                target,
                "ood_event",
                8,
                rng,
                used,
                "ood",
                pool=pool,
                required_anchor=anchor,
            )
            for group in groups:
                for variant in range(5):
                    ood_rows.append(serialize(group, variant, "ood_test"))
    write_jsonl(output / "ood_test.jsonl", ood_rows)

    natural_distribution = {"I0": 80, "I1": 50, "I2": 40, "I3": 20, "I4": 10}
    natural_rows: list[dict] = []
    for scene_id in SCENES:
        for target, count in natural_distribution.items():
            groups = generated_groups(
                scene_id,
                target,
                "naturalistic",
                count,
                rng,
                used,
                "natural",
            )
            for index, group in enumerate(groups):
                natural_rows.append(serialize(group, index % 5, "natural_test"))
    write_jsonl(output / "natural_test.jsonl", natural_rows)

    catalog = {
        "dataset": DATASET_NAME,
        "version": 2,
        "synthetic": True,
        "policy_version": POLICY_VERSION,
        "event_phrase_method": "每个事件包含一个人工编写基础描述，并通过同义替换、主体替换和场景化报告句式生成五种文字表达。",
        "scenes": [
            {
                "id": scene_id,
                "name": data["name"],
                "seen_event_count": sum(not event.ood_only for event in data["events"]),
                "ood_event_count": sum(event.ood_only for event in data["events"]),
                "events": [asdict(event) for event in data["events"]],
            }
            for scene_id, data in SCENES.items()
        ],
        "capabilities": CAPABILITIES,
        "responses": RESPONSES,
        "previous_interventions": {
            key: {"text": value[0], "result": value[1]} for key, value in PREVIOUS.items()
        },
        "labels": {
            "derived_risk_level": "由输入事件等级确定的审计字段，不建议作为主要学习目标。",
            "intervention_level": list(INTERVENTION_ORDER),
            "alert_mode": list(ALERT_MODES),
            "manual_review": [False, True],
            "abstain": [False, True],
        },
    }
    (output / "event_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    main_rows = [row for rows in primary_rows.values() for row in rows]
    summary = {
        "dataset": DATASET_NAME,
        "version": 2,
        "seed": SEED,
        "policy_version": POLICY_VERSION,
        "synthetic": True,
        "scene_count": len(SCENES),
        "event_definition_count": sum(len(data["events"]) for data in SCENES.values()),
        "seen_event_count": sum(sum(not e.ood_only for e in data["events"]) for data in SCENES.values()),
        "ood_event_count": sum(sum(e.ood_only for e in data["events"]) for data in SCENES.values()),
        "primary_case_group_count": len(primary_groups),
        "primary_sample_count": len(main_rows),
        "samples_per_split": {split: len(rows) for split, rows in primary_rows.items()},
        "primary_samples_per_scene": count_by(main_rows, "scene_id"),
        "primary_samples_per_intervention": count_by(main_rows, "intervention_level"),
        "primary_samples_per_alert_mode": count_by(main_rows, "alert_mode"),
        "primary_groups_per_case_type": dict(sorted(Counter(g["case_type"] for g in primary_groups).items())),
        "ood_test_sample_count": len(ood_rows),
        "natural_test_sample_count": len(natural_rows),
        "natural_test_interventions": count_by(natural_rows, "intervention_level"),
        "all_sample_count": len(main_rows) + len(ood_rows) + len(natural_rows),
        "warning": "全部标签来自可审计的合成策略，只适合软件与模型流程实验，不代表真实照护、医学、消防或紧急处置标准。",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v2",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
