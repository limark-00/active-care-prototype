import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from decision_model.generate_dataset import CONTEXT_RESPONSES, INTERVENTION_ORDER, SCENES, build


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.summary = build(cls.root)
        cls.rows = {}
        for split in ("train", "validation", "test"):
            cls.rows[split] = [json.loads(line) for line in (cls.root / f"{split}.jsonl").read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_catalog_has_six_scenes_and_many_unique_events(self):
        self.assertEqual(len(SCENES), 6)
        self.assertTrue(all(len(scene["events"]) == 18 for scene in SCENES.values()))
        ids = [event.id for scene in SCENES.values() for event in scene["events"]]
        self.assertEqual(len(ids), 108)
        self.assertEqual(len(set(ids)), len(ids))

    def test_dataset_size_and_balanced_interventions(self):
        self.assertEqual(self.summary["sample_count"], 3600)
        self.assertEqual(self.summary["samples_per_split"], {"train": 2520, "validation": 540, "test": 540})
        self.assertEqual(self.summary["samples_per_intervention"], {f"I{i}": 720 for i in range(5)})
        self.assertEqual(self.summary["samples_per_scene"], {scene: 600 for scene in sorted(SCENES)})

    def test_grouped_variants_never_cross_splits(self):
        locations = {}
        for split, rows in self.rows.items():
            for row in rows:
                locations.setdefault(row["case_group_id"], set()).add(split)
        self.assertEqual(len(locations), 1200)
        self.assertTrue(all(len(splits) == 1 for splits in locations.values()))
        counts = Counter(row["case_group_id"] for rows in self.rows.values() for row in rows)
        self.assertTrue(all(count == 3 for count in counts.values()))

    def test_input_is_text_only_and_does_not_include_target_answer_fields(self):
        for rows in self.rows.values():
            for row in rows[:50]:
                self.assertIn("当前", row["input_text"])
                self.assertIn("请判断", row["input_text"])
                self.assertNotIn("最终风险等级：", row["input_text"])
                self.assertNotIn("干预等级：", row["input_text"])
                self.assertNotIn("是否报警：", row["input_text"])

    def test_policy_safety_invariants(self):
        for rows in self.rows.values():
            for row in rows:
                if row["risk_level"] == "L4":
                    self.assertEqual(row["intervention_level"], "I4")
                self.assertEqual(row["alarm"], row["intervention_level"] == "I4")
                self.assertEqual(row["intervene"], row["intervention_level"] != "I0")
                self.assertIn(row["intervention_level"], INTERVENTION_ORDER)


    def test_previous_action_and_response_are_consistent(self):
        for rows in self.rows.values():
            for row in rows:
                self.assertIn(row["response_code"], CONTEXT_RESPONSES[row["previous_code"]])

    def test_normal_events_are_not_mixed_into_abnormal_anchor_groups(self):
        catalog = {
            event.id: event
            for scene in SCENES.values()
            for event in scene["events"]
        }
        for rows in self.rows.values():
            for row in rows:
                events = [catalog[event_id] for event_id in row["event_ids"]]
                if any("normal" not in event.tags for event in events):
                    self.assertFalse(any("normal" in event.tags for event in events))

    def test_generation_is_deterministic(self):
        second = self.root / "second"
        build(second)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "event_catalog.json", "dataset_summary.json"):
            self.assertEqual((self.root / name).read_bytes(), (second / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
