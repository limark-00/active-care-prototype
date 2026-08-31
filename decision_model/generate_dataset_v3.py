#!/usr/bin/env python3
"""Generate the harder V3 synthetic decision dataset.

V3 removes explicit risk/intervention codes from model input and assigns event
definitions, rather than paraphrases of the same event, to disjoint primary
splits.  It remains a synthetic software experiment and is not care guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .generate_dataset import CAPABILITIES, INTERVENTION_ORDER, RESPONSES
from .generate_dataset_v2 import (
    ALERT_MODES,
    PREVIOUS,
    SCENES,
    SOURCE_LABELS,
    EventV2,
    generated_groups,
    serialize as serialize_v2,
)

SEED = 20260831
DATASET_NAME = "active-care-synthetic-text-decisions-v3"
POLICY_VERSION = "synthetic-care-policy-v2"
PRIMARY_COUNTS = {"train": 1200, "validation": 150, "test": 150}
CHALLENGE_COUNT_PER_SCENE_TARGET = 20

PREVIOUS_NATURAL = {
    "none": "之前还没有进行干预",
    "I0_effective": "此前只保持观察，没有发现新的升级迹象",
    "I1_effective": "之前的温和提醒有效，行为已有改变",
    "I1_ineffective": "之前的温和提醒没有看到明确效果",
    "I1_presented": "温和提醒已经展示，效果还要结合现在的回应判断",
    "I2_effective": "之前的图示引导有效，行为已有改变",
    "I2_ineffective": "之前的图示引导没有看到明确效果",
    "I2_presented": "图示引导已经展示，效果还要结合现在的回应判断",
    "I2_failed": "之前的图示引导没有成功显示",
    "I3_unavailable": "之前请求的保护动作因设备未接入而无法执行",
    "I4_pending": "求助消息已经发出，目前仍在等待照护者接手",
}

STYLE_NAMES = ("plain", "report", "colloquial", "terse", "noisy")


def stable_order(events: list[EventV2], salt: str) -> list[EventV2]:
    return sorted(
        events,
        key=lambda event: hashlib.sha256(
            f"{SEED}:{salt}:{event.id}".encode("utf-8")
        ).hexdigest(),
    )


def assign_event_splits() -> dict[str, dict[str, list[EventV2]]]:
    """Allocate each seen event definition to exactly one primary split.

    Allocation is stratified globally by minimum intervention.  Every
    scene/intervention pair keeps at least one training event when possible;
    remaining definitions are deterministically balanced between validation
    and test.
    """
    allocation = {
        split: {scene_id: [] for scene_id in SCENES}
        for split in ("train", "validation", "test")
    }
    held_out_by_target: dict[str, list[tuple[str, EventV2]]] = defaultdict(list)
    for scene_id, scene in SCENES.items():
        by_target: dict[str, list[EventV2]] = defaultdict(list)
        for event in scene["events"]:
            if not event.ood_only:
                by_target[event.minimum_intervention].append(event)
        for target, events in by_target.items():
            ordered = stable_order(events, f"{scene_id}:{target}")
            allocation["train"][scene_id].append(ordered[0])
            held_out_by_target[target].extend((scene_id, event) for event in ordered[1:])

    for target, values in held_out_by_target.items():
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"{SEED}:heldout:{target}:{item[0]}:{item[1].id}".encode("utf-8")
            ).hexdigest(),
        )
        validation_size = max(2, round(len(ordered) * 0.25))
        test_size = max(2, round(len(ordered) * 0.25))
        for index, (scene_id, event) in enumerate(ordered):
            if index < validation_size:
                split = "validation"
            elif index < validation_size + test_size:
                split = "test"
            else:
                split = "train"
            allocation[split][scene_id].append(event)

    all_ids: dict[str, set[str]] = {}
    for split, by_scene in allocation.items():
        all_ids[split] = {event.id for events in by_scene.values() for event in events}
    if any(all_ids[a] & all_ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("event definitions crossed primary splits")
    for target in INTERVENTION_ORDER:
        for split in allocation:
            if not any(
                event.minimum_intervention == target
                for events in allocation[split].values()
                for event in events
            ):
                raise AssertionError(f"{split} has no {target} event")
    return allocation


def distribute(total: int, keys: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(keys))
    return {key: base + (index < remainder) for index, key in enumerate(keys)}


def soften(text: str) -> str:
    return (
        text.replace("患者", "老人")
        .replace("持续", "一直")
        .replace("多次", "好几次")
        .replace("短时间内", "刚才一会儿")
        .replace("当前", "现在")
        .replace("没有", "没")
    )


def add_typo_noise(text: str) -> str:
    replacements = (("已经", "己经"), ("仍在", "还在"), ("正在", "正"), ("异常", "不太正常"))
    value = text
    for source, target in replacements:
        if source in value:
            value = value.replace(source, target, 1)
            break
    return re.sub(r"[，。；：]", " ", value).strip()


def event_phrase(event: EventV2, style_index: int) -> str:
    base = event.phrases[style_index % len(event.phrases)]
    if style_index == 2:
        return soften(event.phrases[0])
    if style_index == 3:
        return re.sub(r"^(系统|监测人员|本次场景信息)[^：]*：", "", base).strip()
    if style_index == 4:
        return add_typo_noise(soften(event.phrases[0]))
    return base


def render_input(group: dict, style_index: int, *, challenge: bool = False) -> str:
    style_index %= len(STYLE_NAMES)
    events = list(group["events"])
    if style_index in {1, 4}:
        events.reverse()
    scene = group["scene_name"]
    headings = (
        f"地点：{scene}。请根据下面的现场记录进行判断。",
        f"{scene}刚汇总了一组来自摄像头、传感器或人工记录的信息。",
        f"这是{scene}那边刚才的情况。",
        f"{scene}，现场简要记录如下。",
        f"{scene} 这边收到几条不太完整的记录",
    )
    lines = [headings[style_index]]
    for index, event in enumerate(events, 1):
        source = SOURCE_LABELS[event.source].replace("事件", "")
        phrase = event_phrase(event, style_index)
        if style_index == 2:
            lines.append(f"事件{index}：{phrase}。")
        elif style_index == 3:
            lines.append(f"事件{index}：{phrase}")
        else:
            lines.append(f"事件{index}（{source}）：{phrase}。")
    if group.get("conflict_note"):
        lines.append(soften(group["conflict_note"]))
    if group.get("hard_note"):
        lines.append(soften(group["hard_note"]))
    if challenge and style_index == 4:
        lines.append("补充记录可能有延迟 先按现有内容判断")

    ability = CAPABILITIES[group["capability"]]
    response = RESPONSES[group["response"]]
    previous = PREVIOUS_NATURAL[group["previous"]]
    contexts = (
        (f"理解与求助能力：{ability}。", f"当前回应：{response}。", f"之前的处理：{previous}。"),
        (f"能力情况为：{ability}。", f"此前处理是：{previous}。", f"现在的回应是：{response}。"),
        (f"他目前{soften(ability)}。", f"现在{soften(response)}。", f"前面{soften(previous)}。"),
        (f"能力 {ability}", f"回应 {response}", f"历史 {previous}"),
        (f"能力情况 {add_typo_noise(ability)}", f"之前 {add_typo_noise(previous)}", f"回应 {add_typo_noise(response)}"),
    )
    lines.extend(contexts[style_index])
    lines.append("请给出下一步干预、网页警示，以及是否转人工核查。")
    text = "\n".join(lines)
    if re.search(r"(?:^|[^A-Z])[LI][0-4](?:[^0-9]|$)", text):
        raise AssertionError(f"label code leaked into V3 input: {text}")
    return text


def serialize(group: dict, split: str, style_index: int, *, challenge: bool = False) -> dict:
    row = serialize_v2(group, 0, split)
    row["sample_id"] = f"{group['case_group_id']}-{STYLE_NAMES[style_index % len(STYLE_NAMES)]}"
    row["variant_id"] = style_index + 1
    row["input_style"] = STYLE_NAMES[style_index % len(STYLE_NAMES)]
    row["input_text"] = render_input(group, style_index, challenge=challenge)
    row["policy_version"] = POLICY_VERSION
    return row


def make_groups_for_target(
    *,
    split: str,
    target: str,
    total: int,
    allocation: dict[str, dict[str, list[EventV2]]],
    rng: random.Random,
    used: set[tuple],
) -> list[dict]:
    eligible = [
        scene_id
        for scene_id, pool in allocation[split].items()
        if any(event.minimum_intervention == target for event in pool)
    ]
    counts = distribute(total, eligible)
    result: list[dict] = []
    for scene_id, count in counts.items():
        pool = allocation[split][scene_id]
        case_counts = {
            "regular": count * 70 // 100,
            "conflict_unknown": count * 15 // 100,
        }
        case_counts["hard"] = count - sum(case_counts.values())
        for case_type, case_count in case_counts.items():
            if not case_count:
                continue
            result.extend(
                generated_groups(
                    scene_id,
                    target,
                    case_type,
                    case_count,
                    rng,
                    used,
                    f"v3-{split}-{case_type}",
                    pool=pool,
                )
            )
    if len(result) != total:
        raise AssertionError(f"wrong group count for {split}/{target}")
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    allocation = assign_event_splits()
    used: set[tuple] = set()
    rows_by_split: dict[str, list[dict]] = {}
    for split, count_per_target in PRIMARY_COUNTS.items():
        groups: list[dict] = []
        for target in INTERVENTION_ORDER:
            groups.extend(
                make_groups_for_target(
                    split=split,
                    target=target,
                    total=count_per_target,
                    allocation=allocation,
                    rng=rng,
                    used=used,
                )
            )
        rows_by_split[split] = [
            serialize(group, split, index % len(STYLE_NAMES))
            for index, group in enumerate(groups)
        ]
        write_jsonl(output / f"{split}.jsonl", rows_by_split[split])

    ood_rows: list[dict] = []
    natural_rows: list[dict] = []
    for scene_id, scene in SCENES.items():
        ood_pool = [event for event in scene["events"] if event.ood_only]
        test_pool = allocation["test"][scene_id]
        for target in INTERVENTION_ORDER:
            ood_anchor = next(event for event in ood_pool if event.minimum_intervention == target)
            ood_groups = generated_groups(
                scene_id,
                target,
                "ood_event",
                CHALLENGE_COUNT_PER_SCENE_TARGET,
                rng,
                used,
                "v3-ood",
                pool=ood_pool,
                required_anchor=ood_anchor,
            )
            ood_rows.extend(
                serialize(group, "ood_test", index % len(STYLE_NAMES), challenge=True)
                for index, group in enumerate(ood_groups)
            )
            anchors = [event for event in test_pool if event.minimum_intervention == target]
            if anchors:
                natural_groups = generated_groups(
                    scene_id,
                    target,
                    "language_challenge",
                    CHALLENGE_COUNT_PER_SCENE_TARGET,
                    rng,
                    used,
                    "v3-natural",
                    pool=test_pool,
                    required_anchor=anchors[0],
                )
                natural_rows.extend(
                    serialize(group, "natural_test", 4 - index % 3, challenge=True)
                    for index, group in enumerate(natural_groups)
                )
    write_jsonl(output / "ood_test.jsonl", ood_rows)
    write_jsonl(output / "natural_test.jsonl", natural_rows)

    event_ids = {
        split: sorted(event.id for events in by_scene.values() for event in events)
        for split, by_scene in allocation.items()
    }
    catalog = {
        "dataset": DATASET_NAME,
        "version": 3,
        "synthetic": True,
        "policy_version": POLICY_VERSION,
        "input_contract": "模型只读取input_text；L/I代码只保存在输出标签和审计字段中。",
        "split_method": "主数据按事件定义ID互斥切分，不按同一事件的改写句随机切分。",
        "primary_event_ids": event_ids,
        "styles": list(STYLE_NAMES),
        "labels": {
            "intervention_level": list(INTERVENTION_ORDER),
            "alert_mode": list(ALERT_MODES),
            "manual_review": [False, True],
            "abstain": [False, True],
        },
    }
    (output / "event_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    primary = [row for split in ("train", "validation", "test") for row in rows_by_split[split]]
    summary = {
        "dataset": DATASET_NAME,
        "version": 3,
        "seed": SEED,
        "policy_version": POLICY_VERSION,
        "synthetic": True,
        "explicit_label_codes_in_input": False,
        "primary_event_ids_are_disjoint": True,
        "samples_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "primary_samples_per_intervention": dict(sorted(Counter(row["intervention_level"] for row in primary).items())),
        "primary_samples_per_case_type": dict(sorted(Counter(row["case_type"] for row in primary).items())),
        "primary_event_count_per_split": {split: len(ids) for split, ids in event_ids.items()},
        "ood_test_sample_count": len(ood_rows),
        "natural_test_sample_count": len(natural_rows),
        "all_sample_count": len(primary) + len(ood_rows) + len(natural_rows),
        "warning": "V3仍为合成数据，只能评估模型工程与合成策略学习，不能代表真实照护准确率。",
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
        default=Path(__file__).resolve().parent / "data" / "v3",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
