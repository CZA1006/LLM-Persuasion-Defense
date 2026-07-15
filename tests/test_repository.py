from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from src.datasets import load_counterfact_subset


ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrityTests(unittest.TestCase):
    def test_default_dataset_is_versioned_subset(self) -> None:
        records = load_counterfact_subset(n=50, seed=1337)
        self.assertEqual(len(records), 50)

    def test_counterfact_strict_schema(self) -> None:
        path = ROOT / "data" / "counterfact_50_strict.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

        self.assertEqual(len(records), 50)
        self.assertEqual(len({record["id"] for record in records}), 50)
        required = {"id", "category", "category_name", "subject", "relation", "o_true", "o_false"}
        for record in records:
            self.assertTrue(required.issubset(record))
            self.assertNotEqual(record["o_true"].strip(), record["o_false"].strip())

    def test_psr_table_is_complete(self) -> None:
        path = ROOT / "artifacts" / "results" / "psr_by_turn.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5 * 8)
        self.assertEqual({row["group"] for row in rows}, {"G1", "G2", "G3", "G4", "G5"})
        for group in {row["group"] for row in rows}:
            turns = {int(row["turn"]) for row in rows if row["group"] == group}
            self.assertEqual(turns, set(range(1, 9)))
        for row in rows:
            self.assertEqual(int(row["n"]), 50)
            self.assertLessEqual(float(row["ci_low"]), float(row["psr"]))
            self.assertLessEqual(float(row["psr"]), float(row["ci_high"]))

    def test_quality_counts_and_rates_are_consistent(self) -> None:
        path = ROOT / "artifacts" / "results" / "quality_distribution.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        for row in rows:
            persuasion = int(row["persuasion_count"])
            compliance = int(row["compliance_count"])
            total = int(row["successful_turns"])
            self.assertEqual(persuasion + compliance, total)
            self.assertAlmostEqual(float(row["persuasion_rate"]), persuasion / total, places=4)
            self.assertAlmostEqual(float(row["compliance_rate"]), compliance / total, places=4)

    def test_documentation_assets_exist(self) -> None:
        required_paths = [
            "LICENSE",
            "CITATION.cff",
            "docs/assets/sast-ir-framework.png",
            "artifacts/figures/psr_curve.png",
            "artifacts/figures/quality_distribution.png",
        ]
        for relative_path in required_paths:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
