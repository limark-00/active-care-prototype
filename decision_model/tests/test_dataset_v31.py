import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from decision_model.generate_dataset_v31 import SCENES_V31, build


class DatasetV31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.summary = build(cls.root)
        cls.rows = {
            name: [json.loads(line) for line in (cls.root / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "validation", "test", "ood_test", "natural_test")
        }
        cls.catalog = json.loads((cls.root / "event_catalog.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_catalog_and_sample_sizes(self):
        self.assertEqual(len(SCENES_V31), 6)
        self.assertEqual(self.summary["event_definition_count"], 222)
        self.assertEqual(self.summary["added_event_definition_count"], 42)
        self.assertEqual(
            self.summary["samples_per_split"],
            {"train": 4800, "validation": 1200, "test": 1200},
        )
        self.assertEqual(len(self.rows["ood_test"]), 600)
        self.assertEqual(len(self.rows["natural_test"]), 600)

    def test_each_primary_split_has_every_scene_and_intervention_stratum(self):
        allocation = self.catalog["primary_event_ids"]
        for split in ("train", "validation", "test"):
            for scene_id in SCENES_V31:
                for level in ("I0", "I1", "I2", "I3", "I4"):
                    self.assertTrue(allocation[split][scene_id][level])

    def test_event_ids_are_disjoint_and_rows_respect_allocation(self):
        allocation = self.catalog["primary_event_ids"]
        split_ids = {
            split: {
                event_id
                for scene in allocation[split].values()
                for values in scene.values()
                for event_id in values
            }
            for split in ("train", "validation", "test")
        }
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        self.assertFalse(split_ids["validation"] & split_ids["test"])
        for split in split_ids:
            for row in self.rows[split]:
                self.assertLessEqual(set(row["event_ids"]), split_ids[split])

    def test_primary_interventions_are_balanced_and_risk_is_available(self):
        primary = self.rows["train"] + self.rows["validation"] + self.rows["test"]
        self.assertEqual(
            Counter(row["intervention_level"] for row in primary),
            {f"I{i}": 1440 for i in range(5)},
        )
        self.assertEqual({row["derived_risk_level"] for row in primary}, {f"L{i}" for i in range(5)})

    def test_no_answer_codes_appear_in_input(self):
        pattern = re.compile(r"(?:^|[^A-Z])[LI][0-4](?:[^0-9]|$)")
        for rows in self.rows.values():
            for row in rows:
                self.assertIsNone(pattern.search(row["input_text"]), row["sample_id"])

    def test_generation_is_deterministic(self):
        second = self.root / "second"
        build(second)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "ood_test.jsonl", "natural_test.jsonl", "event_catalog.json", "dataset_summary.json"):
            self.assertEqual((self.root / name).read_bytes(), (second / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
