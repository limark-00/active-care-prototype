import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from decision_model.generate_dataset_v2 import ALERT_MODES, SCENES, build


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.summary = build(cls.root)
        cls.primary = {}
        for split in ("train", "validation", "test"):
            cls.primary[split] = [
                json.loads(line)
                for line in (cls.root / f"{split}.jsonl").read_text().splitlines()
            ]
        cls.ood = [json.loads(line) for line in (cls.root / "ood_test.jsonl").read_text().splitlines()]
        cls.natural = [json.loads(line) for line in (cls.root / "natural_test.jsonl").read_text().splitlines()]
        cls.main_rows = [row for rows in cls.primary.values() for row in rows]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_catalog_has_six_scenes_and_180_events(self):
        self.assertEqual(len(SCENES), 6)
        self.assertTrue(all(len(scene["events"]) == 30 for scene in SCENES.values()))
        self.assertTrue(all(sum(e.ood_only for e in scene["events"]) == 5 for scene in SCENES.values()))
        ids = [event.id for scene in SCENES.values() for event in scene["events"]]
        self.assertEqual(len(ids), 180)
        self.assertEqual(len(set(ids)), 180)
        for scene in SCENES.values():
            for event in scene["events"]:
                self.assertEqual(len(event.phrases), 5)
                self.assertEqual(len(set(event.phrases)), 5)

    def test_primary_size_split_scene_and_intervention_balance(self):
        self.assertEqual(self.summary["primary_case_group_count"], 6000)
        self.assertEqual(self.summary["primary_sample_count"], 30000)
        self.assertEqual(
            self.summary["samples_per_split"],
            {"train": 21000, "validation": 4500, "test": 4500},
        )
        self.assertEqual(
            self.summary["primary_samples_per_intervention"],
            {f"I{i}": 6000 for i in range(5)},
        )
        self.assertEqual(
            self.summary["primary_samples_per_scene"],
            {scene_id: 5000 for scene_id in sorted(SCENES)},
        )
        for split, rows in self.primary.items():
            expected = 4200 if split == "train" else 900
            self.assertEqual(Counter(row["intervention_level"] for row in rows), {f"I{i}": expected for i in range(5)})

    def test_case_types_have_planned_group_counts(self):
        one_per_group = {row["case_group_id"]: row for row in self.main_rows}.values()
        self.assertEqual(
            Counter(row["case_type"] for row in one_per_group),
            {"regular": 3600, "counterfactual": 1200, "conflict_unknown": 600, "hard": 600},
        )

    def test_variants_and_counterfactual_pairs_do_not_cross_splits(self):
        group_splits = defaultdict(set)
        group_variants = Counter()
        pairs = defaultdict(list)
        group_rows = {}
        for split, rows in self.primary.items():
            for row in rows:
                group_splits[row["case_group_id"]].add(split)
                group_variants[row["case_group_id"]] += 1
                group_rows.setdefault(row["case_group_id"], row)
                if row["counterfactual_pair_id"]:
                    pairs[row["counterfactual_pair_id"]].append(row)
        self.assertEqual(len(group_splits), 6000)
        self.assertTrue(all(len(value) == 1 for value in group_splits.values()))
        self.assertTrue(all(value == 5 for value in group_variants.values()))
        self.assertEqual(len(pairs), 600)
        for pair_id, rows in pairs.items():
            groups = {row["case_group_id"] for row in rows}
            self.assertEqual(len(groups), 2)
            self.assertEqual(len({row["split"] for row in rows}), 1)
            first, second = [group_rows[group] for group in sorted(groups)]
            self.assertEqual(first["event_ids"], second["event_ids"])
            changed = sum(
                first[field] != second[field]
                for field in ("capability_code", "response_code", "previous_code")
            )
            self.assertEqual(changed, 1, pair_id)
            self.assertNotEqual(first["intervention_level"], second["intervention_level"])

    def test_ood_events_are_held_out_of_primary_data(self):
        ood_ids = {event.id for scene in SCENES.values() for event in scene["events"] if event.ood_only}
        self.assertEqual(len(ood_ids), 30)
        self.assertFalse(any(ood_ids.intersection(row["event_ids"]) for row in self.main_rows))
        self.assertEqual(len(self.ood), 1200)
        self.assertTrue(all(ood_ids.intersection(row["event_ids"]) for row in self.ood))
        self.assertEqual(Counter(row["intervention_level"] for row in self.ood), {f"I{i}": 240 for i in range(5)})

    def test_naturalistic_test_has_separate_distribution(self):
        self.assertEqual(len(self.natural), 1200)
        self.assertEqual(
            Counter(row["intervention_level"] for row in self.natural),
            {"I0": 480, "I1": 300, "I2": 240, "I3": 120, "I4": 60},
        )

    def test_alert_is_independent_and_safety_invariants_hold(self):
        pairs = {(row["intervention_level"], row["alert_mode"]) for row in self.main_rows}
        self.assertIn(("I0", "NONE"), pairs)
        self.assertIn(("I0", "PAGE_WARNING"), pairs)
        self.assertIn(("I1", "NONE"), pairs)
        self.assertIn(("I1", "PAGE_WARNING"), pairs)
        for row in self.main_rows + self.ood + self.natural:
            self.assertIn(row["alert_mode"], ALERT_MODES)
            if row["derived_risk_level"] == "L4" or row["intervention_level"] == "I4":
                self.assertEqual(row["intervention_level"], "I4")
                self.assertEqual(row["alert_mode"], "URGENT_HELP")
            if row["abstain"]:
                self.assertTrue(row["manual_review"])
                self.assertNotEqual(row["intervention_level"], "I4")

    def test_primary_compositions_are_unique_and_auxiliary_sets_do_not_reuse_them(self):
        primary_hashes = {row["composition_hash"] for row in self.main_rows}
        self.assertEqual(len(primary_hashes), 6000)
        ood_hashes = {row["composition_hash"] for row in self.ood}
        natural_hashes = {row["composition_hash"] for row in self.natural}
        self.assertEqual(len(ood_hashes), 240)
        self.assertEqual(len(natural_hashes), 1200)
        self.assertFalse(primary_hashes & ood_hashes)
        self.assertFalse(primary_hashes & natural_hashes)
        self.assertFalse(ood_hashes & natural_hashes)

    def test_generation_is_deterministic(self):
        second = self.root / "second"
        build(second)
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "ood_test.jsonl",
            "natural_test.jsonl",
            "event_catalog.json",
            "dataset_summary.json",
        ):
            first_digest = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            second_digest = hashlib.sha256((second / name).read_bytes()).hexdigest()
            self.assertEqual(first_digest, second_digest, name)


if __name__ == "__main__":
    unittest.main()
