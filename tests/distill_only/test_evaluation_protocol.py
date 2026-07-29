"""Tests for the small protocol-to-condition expansion helper."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "distill_only" / "evaluation_conditions.py"
PROTOCOL = ROOT / "scripts" / "distill_only" / "evaluation_protocol_itut.yaml"
SPEC = importlib.util.spec_from_file_location("evaluation_conditions", HELPER)
assert SPEC is not None and SPEC.loader is not None
conditions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conditions)


class EvaluationProtocolTest(unittest.TestCase):
    def test_screening_is_exact(self) -> None:
        rows = conditions.expand_conditions(PROTOCOL, "screening", ("valid",))
        self.assertEqual(
            [row["name"] for row in rows],
            ["clean", "babble_0db", "speech_0db"],
        )
        self.assertIsNone(rows[0]["noise_type"])
        self.assertIsNone(rows[0]["noise_method"])
        self.assertEqual(rows[1]["noise_method"], "itut")
        self.assertEqual(rows[1]["snr_db"], 0)

    def test_final_condition_counts_and_paths(self) -> None:
        valid = conditions.expand_conditions(PROTOCOL, "final", ("valid",))
        both = conditions.expand_conditions(
            PROTOCOL, "final", ("valid", "test")
        )
        self.assertEqual(len(valid), 16)
        self.assertEqual(len(both), 32)
        self.assertEqual(
            {row["snr_db"] for row in valid if row["noise_type"]},
            {-10, -5, 0, 5, 10},
        )
        self.assertEqual({row["subset"] for row in both}, {"valid", "test"})
        self.assertTrue(
            any("/babble/-10/valid" in row["relative_path"] for row in valid)
        )

    def test_screening_rejects_test_and_duplicate_subsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the valid"):
            conditions.expand_conditions(PROTOCOL, "screening", ("test",))
        with self.assertRaisesRegex(ValueError, "unique"):
            conditions.expand_conditions(PROTOCOL, "final", ("valid", "valid"))

    def test_protocol_schema_and_matrix_are_validated(self) -> None:
        raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.yaml"
            raw["schema_version"] = "wrong"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                conditions.expand_conditions(path, "final", ("valid",))
            raw["schema_version"] = "chapter3-evaluation/v1"
            raw["final_evaluation"]["snrs_db"] = [0]
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SNRs"):
                conditions.expand_conditions(path, "final", ("valid",))


if __name__ == "__main__":
    unittest.main()
