#!/usr/bin/env python3
"""Generate a deterministic synthetic Chinese decision dataset.

The generated labels encode prototype policy, not medical or emergency guidance.
Only ``input_text`` is intended as model input. Metadata is retained for audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SEED = 20260829
POLICY_VERSION = "synthetic-care-policy-v1"
RISK_ORDER = {f"L{i}": i for i in range(5)}
INTERVENTION_ORDER = {f"I{i}": i for i in range(5)}
SOURCE_LABELS = {
    "camera": "摄像头事件",
    "sensor": "模拟传感器事件",
    "manual": "人工输入事件",
    "system": "系统状态事件",
}


@dataclass(frozen=True)
class Event:
    id: str
    risk: str
    minimum_intervention: str
    text: str
    source: str
    channel: str
    tags: tuple[str, ...] = ()


def E(id: str, risk: str, intervention: str, text: str, source: str, channel: str, *tags: str) -> Event:
    return Event(id, risk, intervention, text, source, channel, tuple(tags))


SCENES: dict[str, dict] = {
    "kitchen": {
        "name": "厨房",
        "events": [
            E("k01", "L0", "I0", "患者按日常步骤准备简单餐食，动作平稳", "camera", "activity", "normal"),
            E("k02", "L0", "I0", "灶具处于关闭状态，患者已经离开灶台", "sensor", "stove", "normal"),
            E("k03", "L0", "I0", "烟雾与燃气输入均为正常状态", "sensor", "air_alarm", "normal"),
            E("k04", "L1", "I0", "患者靠近灶台观察，但尚未进行危险操作", "camera", "activity", "observe"),
            E("k05", "L1", "I0", "患者在厨房内反复改变行走方向", "camera", "movement", "observe"),
            E("k06", "L1", "I1", "厨房温度持续高于演示舒适范围", "sensor", "temperature", "environment"),
            E("k07", "L1", "I1", "厨房环境输入中断，当前状态无法确认", "system", "device", "unknown"),
            E("k08", "L2", "I1", "患者在开启的灶台附近持续停留", "camera", "stove", "dwell"),
            E("k09", "L2", "I1", "患者短时间内重复操作同一个灶具开关", "camera", "activity", "repetition"),
            E("k10", "L2", "I2", "患者伸手接近高温锅具区域", "camera", "reach", "hazard"),
            E("k11", "L2", "I2", "患者准备离开厨房，但灶具仍显示开启", "sensor", "stove", "hazard"),
            E("k12", "L3", "I3", "锅具持续加热且附近没有人员照看", "sensor", "cooking", "protection"),
            E("k13", "L3", "I3", "患者衣物或身体长时间靠近明火区域", "camera", "reach", "protection"),
            E("k14", "L3", "I4", "患者在厨房出现快速下降并保持低位姿态", "camera", "posture", "fall"),
            E("k15", "L3", "I2", "厨房出口被临时物品遮挡，通行空间不足", "manual", "exit", "hazard"),
            E("k16", "L4", "I4", "独立烟雾报警输入被触发", "sensor", "air_alarm", "urgent", "smoke"),
            E("k17", "L4", "I4", "独立燃气报警输入被触发", "sensor", "air_alarm", "urgent", "gas"),
            E("k18", "L4", "I4", "人工确认灶台附近出现明火失控迹象", "manual", "fire", "urgent", "fire"),
        ],
    },
    "bathroom": {
        "name": "卫生间",
        "events": [
            E("b01", "L0", "I0", "患者正常进入卫生间并保持稳定行走", "camera", "activity", "normal"),
            E("b02", "L0", "I0", "患者完成洗漱后正常离开卫生间", "camera", "movement", "normal"),
            E("b03", "L0", "I0", "地面干燥且通行区域没有明显障碍", "manual", "floor", "normal"),
            E("b04", "L1", "I0", "患者在洗手台附近停留时间略长", "camera", "dwell", "observe"),
            E("b05", "L1", "I0", "患者进入卫生间后移动速度比平时慢", "camera", "movement", "observe"),
            E("b06", "L1", "I1", "卫生间照明不足，人体关键点识别质量下降", "system", "device", "unknown"),
            E("b07", "L1", "I1", "卫生间地面被人工标记为潮湿", "manual", "floor", "environment"),
            E("b08", "L2", "I1", "患者在卫生间内停留超过预设观察时间", "camera", "dwell", "dwell"),
            E("b09", "L2", "I1", "热水温度高于原型演示范围", "sensor", "water", "environment"),
            E("b10", "L2", "I2", "患者起身时身体出现持续摇晃", "camera", "posture", "instability"),
            E("b11", "L2", "I2", "患者连续多次开关水龙头", "camera", "activity", "repetition"),
            E("b12", "L3", "I3", "持续出水已经接近地面积水阈值", "sensor", "water", "protection"),
            E("b13", "L3", "I2", "患者踩入已标记的湿滑区域", "camera", "floor", "hazard"),
            E("b14", "L3", "I4", "患者出现快速下降并保持倒地候选姿态", "camera", "posture", "fall"),
            E("b15", "L3", "I4", "卫生间门处于锁闭状态且存在待核查异常事件", "sensor", "door", "hazard"),
            E("b16", "L4", "I4", "人工确认患者发生严重烫伤风险", "manual", "water", "urgent"),
            E("b17", "L4", "I4", "卫生间电气区域触发烟雾报警输入", "sensor", "air_alarm", "urgent", "smoke"),
            E("b18", "L1", "I1", "卫生间门磁或水温输入处于故障状态", "system", "device", "unknown"),
        ],
    },
    "bedroom": {
        "name": "卧室",
        "events": [
            E("d01", "L0", "I0", "患者在床上正常休息，姿态变化平稳", "camera", "activity", "normal"),
            E("d02", "L0", "I0", "患者醒来后坐在床边并保持稳定", "camera", "posture", "normal"),
            E("d03", "L0", "I0", "卧室门与窗均处于预设正常状态", "sensor", "opening", "normal"),
            E("d04", "L1", "I0", "患者夜间短暂坐起后继续观察周围", "camera", "activity", "observe"),
            E("d05", "L1", "I0", "患者多次调整睡姿，尚未离开床区", "camera", "posture", "observe"),
            E("d06", "L1", "I1", "卧室光线不足导致姿态观测质量下降", "system", "device", "unknown"),
            E("d07", "L1", "I1", "卧室温度持续偏离演示舒适范围", "sensor", "temperature", "environment"),
            E("d08", "L2", "I1", "患者在夜间离开床区并停留在床边", "camera", "movement", "dwell"),
            E("d09", "L2", "I1", "患者离床后长时间没有返回休息区域", "camera", "dwell", "dwell"),
            E("d10", "L2", "I2", "患者在卧室内重复往返且路径不稳定", "camera", "movement", "repetition"),
            E("d11", "L2", "I2", "患者尝试从低位姿态起身但动作不稳定", "camera", "posture", "instability"),
            E("d12", "L3", "I3", "患者持续接近已标记的窗边限制区域", "camera", "opening", "protection"),
            E("d13", "L3", "I4", "患者从床边快速下降并保持低位姿态", "camera", "posture", "fall"),
            E("d14", "L3", "I4", "患者位于床外地面且持续没有明显位移", "camera", "floor", "fall"),
            E("d15", "L3", "I2", "床边通道被物品遮挡，起身路径不完整", "manual", "exit", "hazard"),
            E("d16", "L4", "I4", "卧室烟雾报警输入被触发", "sensor", "air_alarm", "urgent", "smoke"),
            E("d17", "L4", "I4", "人工确认患者出现需要立即协助的急性异常", "manual", "health", "urgent"),
            E("d18", "L1", "I1", "床区摄像头或门磁输入处于离线状态", "system", "device", "unknown"),
        ],
    },
    "living_room": {
        "name": "客厅",
        "events": [
            E("l01", "L0", "I0", "患者在客厅安静坐着并正常观看电视", "camera", "activity", "normal"),
            E("l02", "L0", "I0", "患者沿熟悉路线平稳经过客厅", "camera", "movement", "normal"),
            E("l03", "L0", "I0", "客厅通道清晰，环境输入处于正常状态", "manual", "floor", "normal"),
            E("l04", "L1", "I0", "患者在客厅中央持续站立并观察周围", "camera", "dwell", "observe"),
            E("l05", "L1", "I0", "患者短时间内重复经过同一路线", "camera", "movement", "observe"),
            E("l06", "L1", "I1", "客厅地面存在需要留意的零散物品", "manual", "floor", "environment"),
            E("l07", "L1", "I1", "客厅摄像头画面受到局部遮挡", "system", "device", "unknown"),
            E("l08", "L2", "I1", "患者离开预设活动区域后持续未返回", "camera", "movement", "dwell"),
            E("l09", "L2", "I1", "患者坐在沙发上长时间没有姿态变化", "camera", "posture", "dwell"),
            E("l10", "L2", "I2", "患者身体持续前倾并多次尝试起身", "camera", "posture", "instability"),
            E("l11", "L2", "I2", "客厅门保持打开且患者正在接近门口", "sensor", "door", "hazard"),
            E("l12", "L3", "I3", "患者踩上不稳定家具并试图向高处伸手", "camera", "activity", "protection"),
            E("l13", "L3", "I3", "取暖设备持续开启且周围存在易燃物", "sensor", "heater", "protection"),
            E("l14", "L3", "I4", "患者出现快速下降并保持横卧候选姿态", "camera", "posture", "fall"),
            E("l15", "L3", "I2", "患者持续触碰被标记为不安全的电源区域", "camera", "electric", "hazard"),
            E("l16", "L4", "I4", "客厅烟雾报警输入被触发", "sensor", "air_alarm", "urgent", "smoke"),
            E("l17", "L4", "I4", "人工确认患者突然倒地并需要立即核查", "manual", "health", "urgent", "fall"),
            E("l18", "L1", "I1", "客厅环境采样设备处于故障状态", "system", "device", "unknown"),
        ],
    },
    "entrance": {
        "name": "入户门区域",
        "events": [
            E("e01", "L0", "I0", "患者在预定时间经过入户门区域", "camera", "activity", "normal"),
            E("e02", "L0", "I0", "患者从门外返回并正常进入室内", "camera", "movement", "normal"),
            E("e03", "L0", "I0", "入户门处于关闭状态且没有越线事件", "sensor", "door", "normal"),
            E("e04", "L1", "I0", "患者靠近入户门并短暂停留", "camera", "dwell", "observe"),
            E("e05", "L1", "I0", "入户门短暂打开后仍处于观察阶段", "sensor", "door", "observe"),
            E("e06", "L1", "I1", "患者在门口反复查看周围环境", "camera", "activity", "observe"),
            E("e07", "L1", "I1", "入户门摄像头或门磁状态无法确认", "system", "device", "unknown"),
            E("e08", "L2", "I1", "患者短时间内多次操作门把手", "camera", "activity", "repetition"),
            E("e09", "L2", "I1", "入户门持续打开并超过预设观察时间", "sensor", "door", "dwell"),
            E("e10", "L2", "I2", "患者穿过门口边界并向室外方向移动", "camera", "movement", "hazard"),
            E("e11", "L2", "I2", "未经确认的访客正在门外持续停留", "manual", "visitor", "hazard"),
            E("e12", "L3", "I3", "患者持续尝试打开被标记为限制区域的门", "camera", "door", "protection"),
            E("e13", "L3", "I2", "患者已经离开室内且超过预设返回时间", "camera", "movement", "hazard"),
            E("e14", "L3", "I4", "患者在门槛附近出现快速下降并保持低位姿态", "camera", "posture", "fall"),
            E("e15", "L3", "I2", "门口通道被物品阻挡并影响正常通行", "manual", "exit", "hazard"),
            E("e16", "L4", "I4", "入户门附近烟雾报警输入被触发", "sensor", "air_alarm", "urgent", "smoke"),
            E("e17", "L4", "I4", "人工确认患者在室外处于立即需要协助的状态", "manual", "health", "urgent"),
            E("e18", "L1", "I1", "门磁数据过期，无法判断门是否已经关闭", "system", "device", "unknown"),
        ],
    },
    "balcony": {
        "name": "阳台",
        "events": [
            E("a01", "L0", "I0", "患者正常进入阳台并在安全区域内活动", "camera", "activity", "normal"),
            E("a02", "L0", "I0", "患者完成浇花后正常返回室内", "camera", "movement", "normal"),
            E("a03", "L0", "I0", "阳台门与地面状态符合预设正常条件", "sensor", "door", "normal"),
            E("a04", "L1", "I0", "患者靠近阳台边缘观察区域但仍在安全线内", "camera", "dwell", "observe"),
            E("a05", "L1", "I0", "患者在阳台停留时间略长", "camera", "activity", "observe"),
            E("a06", "L1", "I1", "阳台光照变化导致关键点识别质量下降", "system", "device", "unknown"),
            E("a07", "L1", "I1", "人工标记阳台地面存在少量积水", "manual", "floor", "environment"),
            E("a08", "L2", "I1", "患者在阳台持续停留并多次查看室外", "camera", "dwell", "dwell"),
            E("a09", "L2", "I2", "患者站在湿滑区域并出现步态不稳", "camera", "floor", "instability"),
            E("a10", "L2", "I2", "患者位于阳台外侧且阳台门意外关闭", "sensor", "door", "hazard"),
            E("a11", "L2", "I2", "患者反复接近预设边界后又返回", "camera", "movement", "repetition"),
            E("a12", "L3", "I3", "患者踩上椅子并靠近阳台边缘", "camera", "activity", "protection"),
            E("a13", "L3", "I3", "患者身体持续越过预设栏杆安全线", "camera", "boundary", "protection"),
            E("a14", "L3", "I4", "患者在阳台出现快速下降并保持低位姿态", "camera", "posture", "fall"),
            E("a15", "L3", "I2", "阳台返回室内的通道被物品遮挡", "manual", "exit", "hazard"),
            E("a16", "L4", "I4", "阳台附近烟雾报警输入被触发", "sensor", "air_alarm", "urgent", "smoke"),
            E("a17", "L4", "I4", "人工确认患者处于可能从高处坠落的紧急状态", "manual", "boundary", "urgent"),
            E("a18", "L1", "I1", "阳台门磁或摄像头输入处于离线状态", "system", "device", "unknown"),
        ],
    },
}

CAPABILITIES = {
    "independent": "能够理解简短语音，也能够主动按键求助",
    "visual_support": "理解语音不稳定，需要图示和分步骤引导",
    "limited": "难以稳定理解提醒，也不能可靠地主动求助",
    "unknown": "当前没有足够资料判断理解和主动求助能力",
}
RESPONSES = {
    "not_requested": "尚未发出提醒，因此没有回应记录",
    "responded": "患者已经确认收到提醒，但相关事件仍需继续观察",
    "changed_behavior": "患者在提醒后改变了行为，但异常证据尚未完全解除",
    "no_response": "已发出的提醒没有获得可确认的回应",
    "declined": "患者明确拒绝当前提醒，系统不能据此推断其没有理解",
    "requested_help": "患者或现场人员主动提出需要帮助",
    "unknown": "当前观测不足，无法判断患者是否回应",
}
PREVIOUS = {
    "none": ("此前没有执行干预", "none"),
    "I0_effective": ("上一次仅观察，未发现新的升级证据", "effective"),
    "I1_effective": ("上一次I1温和提醒已经产生行为变化", "effective"),
    "I1_ineffective": ("上一次I1温和提醒没有产生可确认效果", "ineffective"),
    "I2_effective": ("上一次I2图示引导已经产生行为变化", "effective"),
    "I2_ineffective": ("上一次I2图示引导没有产生可确认效果", "ineffective"),
    "I2_failed": ("上一次I2图示引导未能成功呈现", "failed"),
    "I3_unavailable": ("上一次I3保护请求因设备未接入而不可用", "unavailable"),
    "I4_pending": ("上一次I4网页求助仍在等待人工接手", "pending"),
}


def previous_level(key: str) -> str:
    return "none" if key == "none" else key.split("_", 1)[0]


def decide(events: Iterable[Event], capability: str, response: str, previous: str) -> dict:
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
            key=lambda x: (RISK_ORDER[x.risk], INTERVENTION_ORDER[x.minimum_intervention]),
            reverse=True,
        )[:3]
    ]
    previous_i = previous_level(previous)
    previous_result = PREVIOUS[previous][1]

    if risk == "L4":
        intervention = "I4"
        reasons.append("L4事件直接进入求助分支")
    elif capability in {"visual_support", "limited"} and intervention == "I1":
        intervention = "I2"
        reasons.append("能力档案需要图示或更强支持")

    if response == "requested_help":
        intervention = "I4"
        reasons.append("收到主动求助")
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
        reasons.append("已有求助仍待接手")

    alarm = intervention == "I4"
    manual_review = (
        capability == "unknown"
        or response == "unknown"
        or any("unknown" in event.tags for event in events)
    )
    return {
        "risk_level": risk,
        "intervention_level": intervention,
        "intervene": intervention != "I0",
        "alarm": alarm,
        "manual_review": manual_review,
        "reason_codes": reasons,
    }


CONTEXT_RESPONSES = {
    "none": ("not_requested", "unknown"),
    "I0_effective": ("not_requested", "unknown"),
    "I1_effective": ("responded", "changed_behavior"),
    "I1_ineffective": ("no_response", "declined", "unknown", "requested_help"),
    "I2_effective": ("responded", "changed_behavior"),
    "I2_ineffective": ("no_response", "declined", "unknown", "requested_help"),
    "I2_failed": ("unknown", "requested_help"),
    "I3_unavailable": ("unknown", "requested_help"),
    "I4_pending": ("no_response", "unknown", "responded"),
}


def compatible_context(rng: random.Random) -> tuple[str, str, str]:
    previous = rng.choice(list(PREVIOUS))
    response = rng.choice(CONTEXT_RESPONSES[previous])
    capability = rng.choice(list(CAPABILITIES))
    return capability, response, previous


def render_input(scene_name: str, events: list[Event], capability: str, response: str, previous: str, variant: int) -> str:
    ordered = list(events)
    if variant == 1:
        ordered.reverse()
    elif variant == 2 and len(ordered) > 2:
        ordered = ordered[1:] + ordered[:1]
    headings = [
        f"当前场景：{scene_name}。系统收到以下文字事件。",
        f"地点为{scene_name}，本次需要综合判断这些事件：",
        f"{scene_name}场景的事件汇总如下。",
    ]
    lines = [headings[variant]]
    for index, event in enumerate(ordered, 1):
        source = SOURCE_LABELS[event.source]
        lead = ["事件", "记录", "词条"][variant]
        lines.append(f"{lead}{index}（{event.risk}，{source}）：{event.text}。")
    lines.extend(
        [
            f"患者能力：{CAPABILITIES[capability]}。",
            f"患者响应：{RESPONSES[response]}。",
            f"前次干预：{PREVIOUS[previous][0]}。",
            "请判断当前最终风险等级、干预等级以及是否需要报警。",
        ]
    )
    return "\n".join(lines)


def group_split(scene_id: str, intervention: str, group_ids: list[str]) -> dict[str, str]:
    ordered = sorted(
        group_ids,
        key=lambda value: hashlib.sha256(f"{SEED}:{scene_id}:{intervention}:{value}".encode()).hexdigest(),
    )
    return {group_id: "train" if i < 28 else "validation" if i < 34 else "test" for i, group_id in enumerate(ordered)}


def generate_groups(scene_id: str, target: str, rng: random.Random, count: int = 40) -> list[dict]:
    events: list[Event] = SCENES[scene_id]["events"]
    eligible = [e for e in events if INTERVENTION_ORDER[e.minimum_intervention] <= INTERVENTION_ORDER[target]]
    anchors = [e for e in events if e.minimum_intervention == target]
    if not anchors:
        raise ValueError(f"{scene_id} has no anchor for {target}")
    groups: list[dict] = []
    signatures: set[tuple] = set()
    attempts = 0
    while len(groups) < count:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError(f"Unable to generate {scene_id}/{target}")
        anchor = rng.choice(anchors)
        selected = [anchor]
        channels = {anchor.channel}
        candidates = [
            e
            for e in eligible
            if e.id != anchor.id
            and (("normal" in e.tags) == ("normal" in anchor.tags))
        ]
        rng.shuffle(candidates)
        for event in candidates:
            if len(selected) >= rng.randint(1, 5):
                break
            if event.channel in channels:
                continue
            selected.append(event)
            channels.add(event.channel)
        capability, response, previous = compatible_context(rng)
        labels = decide(selected, capability, response, previous)
        if labels["intervention_level"] != target:
            continue
        signature = (
            tuple(sorted(event.id for event in selected)),
            capability,
            response,
            previous,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        group_id = f"{scene_id}-{target.lower()}-{len(groups) + 1:03d}"
        groups.append(
            {
                "case_group_id": group_id,
                "scene_id": scene_id,
                "scene_name": SCENES[scene_id]["name"],
                "events": selected,
                "capability": capability,
                "response": response,
                "previous": previous,
                "labels": labels,
            }
        )
    return groups


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    all_groups: list[dict] = []
    for scene_id in SCENES:
        for target in INTERVENTION_ORDER:
            all_groups.extend(generate_groups(scene_id, target, rng))

    split_for_group: dict[str, str] = {}
    for scene_id in SCENES:
        for intervention in INTERVENTION_ORDER:
            ids = [
                group["case_group_id"]
                for group in all_groups
                if group["scene_id"] == scene_id
                and group["labels"]["intervention_level"] == intervention
            ]
            split_for_group.update(group_split(scene_id, intervention, ids))

    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for group in all_groups:
        split = split_for_group[group["case_group_id"]]
        for variant in range(3):
            events: list[Event] = group["events"]
            row = {
                "sample_id": f"{group['case_group_id']}-v{variant + 1}",
                "case_group_id": group["case_group_id"],
                "variant_id": variant + 1,
                "split": split,
                "synthetic": True,
                "policy_version": POLICY_VERSION,
                "scene_id": group["scene_id"],
                "scene_name": group["scene_name"],
                "input_text": render_input(
                    group["scene_name"],
                    events,
                    group["capability"],
                    group["response"],
                    group["previous"],
                    variant,
                ),
                "event_ids": [event.id for event in events],
                "event_levels": [event.risk for event in events],
                "capability_code": group["capability"],
                "response_code": group["response"],
                "previous_code": group["previous"],
                **group["labels"],
            }
            rows_by_split[split].append(row)

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: row["sample_id"])
        write_jsonl(output / f"{split}.jsonl", rows)

    catalog = {
        "dataset": "active-care-synthetic-text-decisions",
        "version": 1,
        "synthetic": True,
        "policy_version": POLICY_VERSION,
        "scenes": [
            {
                "id": scene_id,
                "name": data["name"],
                "events": [asdict(event) for event in data["events"]],
            }
            for scene_id, data in SCENES.items()
        ],
        "capabilities": CAPABILITIES,
        "responses": RESPONSES,
        "previous_interventions": {
            key: {"text": value[0], "result": value[1]} for key, value in PREVIOUS.items()
        },
    }
    (output / "event_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    summary = {
        "dataset": catalog["dataset"],
        "version": 1,
        "seed": SEED,
        "policy_version": POLICY_VERSION,
        "synthetic": True,
        "scene_count": len(SCENES),
        "event_definition_count": sum(len(data["events"]) for data in SCENES.values()),
        "case_group_count": len(all_groups),
        "sample_count": len(all_rows),
        "samples_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "samples_per_scene": dict(sorted(Counter(row["scene_id"] for row in all_rows).items())),
        "samples_per_intervention": dict(sorted(Counter(row["intervention_level"] for row in all_rows).items())),
        "samples_per_risk": dict(sorted(Counter(row["risk_level"] for row in all_rows).items())),
        "alarm_counts": {str(k).lower(): v for k, v in sorted(Counter(row["alarm"] for row in all_rows).items())},
        "warning": "全部样本由原型策略合成，只适合软件与模型流程实验，不代表真实照护、医学或消防决策标准。",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "v1")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
