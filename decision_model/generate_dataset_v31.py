#!/usr/bin/env python3
"""Generate V3.1 with stratified unseen-event evaluation.

Every scene has independent event definitions for every intervention level in
train, validation, and test.  Inputs still hide all L/I answer codes.  Labels
remain synthetic prototype policy labels, not real care guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .generate_dataset import INTERVENTION_ORDER
from .generate_dataset_v2 import SCENES, EventV2, generated_groups, phrase_variants
from .generate_dataset_v3 import STYLE_NAMES, serialize, write_jsonl

SEED = 20260901
DATASET_NAME = "active-care-synthetic-text-decisions-v3.1"
POLICY_VERSION = "synthetic-care-policy-v2"
PRIMARY_PER_SCENE_TARGET = {"train": 160, "validation": 40, "test": 40}
CHALLENGE_PER_SCENE_TARGET = 20


EXTRA_SPECS = {
    "kitchen": [
        ("k31", "L2", "I2", "患者反复伸手操作与当前锅具不对应的灶台旋钮", "camera", "stove", ("hazard",)),
        ("k32", "L2", "I2", "盛有热汤的锅具被放在主要通行路线边缘", "manual", "cookware", ("hazard",)),
        ("k33", "L3", "I3", "燃气阀门状态连续异常且患者仍在尝试调节", "sensor", "gas", ("protection",)),
        ("k34", "L3", "I3", "锅具出现干烧迹象且温度仍在快速升高", "sensor", "stove", ("protection",)),
        ("k35", "L3", "I3", "破损电源线位于水槽积水附近并保持通电", "manual", "electric", ("protection",)),
        ("k36", "L3", "I3", "清洁剂储物柜保持打开且患者正在取用不明容器", "camera", "chemical", ("protection",)),
        ("k37", "L4", "I4", "厨房烟雾持续增加且患者倒地后没有可确认回应", "camera", "posture", ("urgent",)),
    ],
    "bathroom": [
        ("b31", "L2", "I2", "淋浴水温持续升高且患者仍停留在出水区域", "sensor", "water_temp", ("hazard",)),
        ("b32", "L2", "I2", "患者反复尝试从湿滑区域跨越较高浴缸边缘", "camera", "movement", ("hazard",)),
        ("b33", "L3", "I3", "地漏堵塞导致积水持续接近通电设备", "sensor", "electric", ("protection",)),
        ("b34", "L3", "I3", "松动的扶手正在受力且患者依赖它保持站立", "manual", "support", ("protection",)),
        ("b35", "L3", "I3", "取暖设备倾倒在潮湿地面并继续运行", "sensor", "heater", ("protection",)),
        ("b36", "L3", "I3", "浴缸水位超过安全观察线且进水仍未停止", "sensor", "water_level", ("protection",)),
        ("b37", "L4", "I4", "患者面部接近水面并持续没有可确认动作", "camera", "posture", ("urgent",)),
    ],
    "bedroom": [
        ("d31", "L2", "I2", "患者夜间反复走向被家具遮挡的狭窄通道", "camera", "movement", ("hazard",)),
        ("d32", "L2", "I2", "患者重复取出不同日期的药盒并混放在床边", "manual", "medicine", ("hazard",)),
        ("d33", "L3", "I3", "取暖设备被织物覆盖且内部温度持续上升", "sensor", "heater", ("protection",)),
        ("d34", "L3", "I3", "患者正准备重复服用已经记录为完成的药物", "manual", "medicine", ("protection",)),
        ("d35", "L3", "I3", "窗户限位装置失效且患者借助家具向窗边攀爬", "camera", "opening", ("protection",)),
        ("d36", "L3", "I3", "床边充电设备明显发热并伴随外壳变形", "sensor", "electric", ("protection",)),
        ("d37", "L4", "I4", "卧室出现浓烟且患者在床边没有可确认回应", "camera", "posture", ("urgent",)),
    ],
    "living_room": [
        ("l31", "L2", "I2", "患者反复将助行器推离起身位置后尝试站立", "camera", "mobility", ("hazard",)),
        ("l32", "L2", "I2", "多根电线横跨主要通道且患者正在靠近", "manual", "floor", ("hazard",)),
        ("l33", "L3", "I3", "倾斜电视柜正在继续位移且患者位于其前方", "camera", "furniture", ("protection",)),
        ("l34", "L3", "I3", "破损插座出现火花且附近设备仍保持供电", "sensor", "electric", ("protection",)),
        ("l35", "L3", "I3", "患者正准备再次服用当天已经确认完成的药物", "manual", "medicine", ("protection",)),
        ("l36", "L3", "I3", "落地灯底座失稳并朝患者活动区域倾倒", "camera", "furniture", ("protection",)),
        ("l37", "L4", "I4", "客厅发生火情且患者倒地后没有可确认回应", "camera", "posture", ("urgent",)),
    ],
    "entrance": [
        ("e31", "L2", "I2", "患者穿着不合脚鞋子并反复尝试跨越门槛", "camera", "movement", ("hazard",)),
        ("e32", "L2", "I2", "患者准备在极端天气提示下独自外出", "manual", "weather", ("hazard",)),
        ("e33", "L3", "I3", "入户门无法保持关闭且患者正在无人陪同下离开", "sensor", "door_lock", ("protection",)),
        ("e34", "L3", "I3", "门外楼梯照明完全失效且患者已经接近首级台阶", "camera", "stairs", ("protection",)),
        ("e35", "L3", "I3", "陌生人员持续拉拽门锁且患者正准备开门", "camera", "visitor", ("protection",)),
        ("e36", "L3", "I3", "门口大件物品倾倒并阻断返回室内的路线", "manual", "exit", ("protection",)),
        ("e37", "L4", "I4", "患者在门外严寒环境中倒地且没有可确认回应", "camera", "posture", ("urgent",)),
    ],
    "balcony": [
        ("a31", "L2", "I2", "患者在强风中反复尝试收取栏杆外侧物品", "camera", "boundary", ("hazard",)),
        ("a32", "L2", "I2", "阳台地面结冰且患者仍准备搬运大型花盆", "manual", "floor", ("hazard",)),
        ("a33", "L3", "I3", "栏杆固定件明显松动且患者正在倚靠栏杆", "camera", "boundary", ("protection",)),
        ("a34", "L3", "I3", "未锁定的梯具正在晃动且患者仍站在高处", "camera", "ladder", ("protection",)),
        ("a35", "L3", "I3", "高处花盆即将坠落且患者位于其下方", "camera", "object", ("protection",)),
        ("a36", "L3", "I3", "阳台电源设备浸入积水并继续保持通电", "sensor", "electric", ("protection",)),
        ("a37", "L4", "I4", "患者悬在栏杆外侧且无法自行返回安全区域", "camera", "boundary", ("urgent",)),
    ],
}


def build_scenes_v31() -> dict[str, dict]:
    result = {}
    for scene_id, scene in SCENES.items():
        events = list(scene["events"])
        for event_id, risk, intervention, text, source, channel, tags in EXTRA_SPECS[scene_id]:
            events.append(
                EventV2(
                    id=event_id,
                    risk=risk,
                    minimum_intervention=intervention,
                    source=source,
                    channel=channel,
                    tags=tuple(tags),
                    phrases=phrase_variants(text, scene["name"]),
                    ood_only=False,
                )
            )
        result[scene_id] = {"name": scene["name"], "events": events}
    ids = [event.id for scene in result.values() for event in scene["events"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate event id")
    return result


SCENES_V31 = build_scenes_v31()


def stable_order(events: list[EventV2], salt: str) -> list[EventV2]:
    return sorted(
        events,
        key=lambda event: hashlib.sha256(
            f"{SEED}:{salt}:{event.id}".encode("utf-8")
        ).hexdigest(),
    )


def assign_event_splits() -> dict[str, dict[str, list[EventV2]]]:
    allocation = {
        split: {scene_id: [] for scene_id in SCENES_V31}
        for split in ("train", "validation", "test")
    }
    for scene_id, scene in SCENES_V31.items():
        by_target: dict[str, list[EventV2]] = defaultdict(list)
        for event in scene["events"]:
            if not event.ood_only:
                by_target[event.minimum_intervention].append(event)
        for target in INTERVENTION_ORDER:
            ordered = stable_order(by_target[target], f"{scene_id}:{target}")
            if len(ordered) < 5:
                raise AssertionError(f"{scene_id}/{target} has fewer than five events")
            allocation["validation"][scene_id].append(ordered[-2])
            allocation["test"][scene_id].append(ordered[-1])
            allocation["train"][scene_id].extend(ordered[:-2])
    for scene_id in SCENES_V31:
        for split in allocation:
            counts = Counter(
                event.minimum_intervention for event in allocation[split][scene_id]
            )
            if any(counts[target] < 1 for target in INTERVENTION_ORDER):
                raise AssertionError(f"missing stratum in {split}/{scene_id}")
    return allocation


def case_counts(total: int) -> dict[str, int]:
    regular = total * 70 // 100
    conflict = total * 15 // 100
    return {
        "regular": regular,
        "conflict_unknown": conflict,
        "hard": total - regular - conflict,
    }


def generate_primary_groups(
    split: str,
    allocation: dict[str, dict[str, list[EventV2]]],
    rng: random.Random,
    used: set[tuple],
) -> list[dict]:
    groups = []
    total = PRIMARY_PER_SCENE_TARGET[split]
    for scene_id in SCENES_V31:
        pool = allocation[split][scene_id]
        for target in INTERVENTION_ORDER:
            for kind, count in case_counts(total).items():
                groups.extend(
                    generated_groups(
                        scene_id,
                        target,
                        kind,
                        count,
                        rng,
                        used,
                        f"v31-{split}-{kind}",
                        pool=pool,
                    )
                )
    return groups


def serialize_v31(group: dict, split: str, style_index: int, *, challenge: bool = False) -> dict:
    row = serialize(group, split, style_index, challenge=challenge)
    row["policy_version"] = POLICY_VERSION
    row["dataset_version"] = "3.1"
    return row


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    used: set[tuple] = set()
    allocation = assign_event_splits()
    rows_by_split = {}
    for split in ("train", "validation", "test"):
        groups = generate_primary_groups(split, allocation, rng, used)
        rows = [
            serialize_v31(group, split, index % len(STYLE_NAMES))
            for index, group in enumerate(groups)
        ]
        rows_by_split[split] = rows
        write_jsonl(output / f"{split}.jsonl", rows)

    ood_rows = []
    natural_rows = []
    ood_used: set[tuple] = set()
    natural_used: set[tuple] = set()
    for scene_id, scene in SCENES_V31.items():
        ood_pool = [event for event in scene["events"] if event.ood_only]
        test_pool = allocation["test"][scene_id]
        for target in INTERVENTION_ORDER:
            ood_anchor = next(
                event for event in ood_pool if event.minimum_intervention == target
            )
            ood_groups = generated_groups(
                scene_id,
                target,
                "ood_event",
                CHALLENGE_PER_SCENE_TARGET,
                rng,
                ood_used,
                "v31-ood",
                pool=ood_pool,
                required_anchor=ood_anchor,
            )
            ood_rows.extend(
                serialize_v31(group, "ood_test", index % len(STYLE_NAMES), challenge=True)
                for index, group in enumerate(ood_groups)
            )
            test_anchor = next(
                event for event in test_pool if event.minimum_intervention == target
            )
            natural_groups = generated_groups(
                scene_id,
                target,
                "language_challenge",
                CHALLENGE_PER_SCENE_TARGET,
                rng,
                natural_used,
                "v31-natural",
                pool=test_pool,
                required_anchor=test_anchor,
            )
            natural_rows.extend(
                serialize_v31(group, "natural_test", 4 - index % 3, challenge=True)
                for index, group in enumerate(natural_groups)
            )
    write_jsonl(output / "ood_test.jsonl", ood_rows)
    write_jsonl(output / "natural_test.jsonl", natural_rows)

    event_ids = {
        split: {
            scene_id: {
                target: sorted(
                    event.id
                    for event in events
                    if event.minimum_intervention == target
                )
                for target in INTERVENTION_ORDER
            }
            for scene_id, events in by_scene.items()
        }
        for split, by_scene in allocation.items()
    }
    catalog = {
        "dataset": DATASET_NAME,
        "version": "3.1",
        "synthetic": True,
        "policy_version": POLICY_VERSION,
        "input_contract": "模型只读取input_text；L/I代码保留为监督与审计字段。",
        "split_method": "按场景和最低干预等级分层后，将事件定义ID互斥切分。",
        "primary_event_ids": event_ids,
        "styles": list(STYLE_NAMES),
        "added_events": {
            scene_id: [asdict(event) for event in scene["events"] if int(event.id[1:]) >= 31]
            for scene_id, scene in SCENES_V31.items()
        },
    }
    (output / "event_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    primary = [row for rows in rows_by_split.values() for row in rows]
    summary = {
        "dataset": DATASET_NAME,
        "version": "3.1",
        "seed": SEED,
        "synthetic": True,
        "explicit_label_codes_in_input": False,
        "stratified_event_ids_are_disjoint": True,
        "scene_count": len(SCENES_V31),
        "event_definition_count": sum(len(scene["events"]) for scene in SCENES_V31.values()),
        "added_event_definition_count": sum(len(events) for events in EXTRA_SPECS.values()),
        "samples_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "primary_samples_per_intervention": dict(sorted(Counter(row["intervention_level"] for row in primary).items())),
        "primary_samples_per_risk": dict(sorted(Counter(row["derived_risk_level"] for row in primary).items())),
        "primary_samples_per_case_type": dict(sorted(Counter(row["case_type"] for row in primary).items())),
        "primary_event_count_per_split": {
            split: sum(len(events) for events in by_scene.values())
            for split, by_scene in allocation.items()
        },
        "ood_test_sample_count": len(ood_rows),
        "natural_test_sample_count": len(natural_rows),
        "all_sample_count": len(primary) + len(ood_rows) + len(natural_rows),
        "warning": "标签仍来自合成策略，只能用于模型工程实验。",
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
        default=Path(__file__).resolve().parent / "data" / "v3_1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
