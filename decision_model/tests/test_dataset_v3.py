import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from decision_model.generate_dataset_v3 import build


class DatasetV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.summary = build(cls.root)
        cls.rows = {
            name: [json.loads(line) for line in (cls.root / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "validation", "test", "ood_test", "natural_test")
        }

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_expected_sizes_and_balanced_primary_targets(self):
        self.assertEqual(
            self.summary["samples_per_split"],
            {"train": 6000, "validation": 750, "test": 750},
        )
        expected = {"train": 1200, "validation": 150, "test": 150}
        for split, per_target in expected.items():
            self.assertEqual(
                Counter(row["intervention_level"] for row in self.rows[split]),
                {f"I{i}": per_target for i in range(5)},
            )

    def test_event_definitions_are_disjoint_across_primary_splits(self):
        ids = {
            split: {event_id for row in self.rows[split] for event_id in row["event_ids"]}
            for split in ("train", "validation", "test")
        }
        self.assertFalse(ids["train"] & ids["validation"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["validation"] & ids["test"])

    def test_input_does_not_expose_risk_or_intervention_codes(self):
        pattern = re.compile(r"(?:^|[^A-Z])[LI][0-4](?:[^0-9]|$)")
        for rows in self.rows.values():
            for row in rows:
                self.assertIsNone(pattern.search(row["input_text"]), row["sample_id"])

    def test_all_training_targets_and_input_styles_are_present(self):
        train = self.rows["train"]
        self.assertEqual({row["input_style"] for row in train}, {"plain", "report", "colloquial", "terse", "noisy"})
        for target in ("manual_review", "abstain"):
            self.assertEqual({row[target] for row in train}, {False, True})
        self.assertEqual(
            {row["alert_mode"] for row in train},
            {"NONE", "PAGE_WARNING", "URGENT_HELP"},
        )

    def test_generation_is_deterministic(self):
        second = self.root / "second"
        build(second)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "ood_test.jsonl", "natural_test.jsonl", "event_catalog.json", "dataset_summary.json"):
            self.assertEqual((self.root / name).read_bytes(), (second / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
