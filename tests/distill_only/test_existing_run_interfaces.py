"""Optional non-destructive smoke checks for local C2/C3a artifacts."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "scripts" / "distill_only" / "inspect_seq2seq_interface.py"
SPEC = importlib.util.spec_from_file_location("chapter3_interface_inspector", TOOL)
assert SPEC is not None and SPEC.loader is not None
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


class ExistingRunInterfaceTest(unittest.TestCase):
    def test_c2_and_c3a_local_interfaces_when_available(self) -> None:
        experiment_dirs = (
            REPO_ROOT
            / "exp"
            / "chapter3_distill_only"
            / "c2_t6_historical_heads",
            REPO_ROOT
            / "exp"
            / "chapter3_distill_only"
            / "c3a_t12_historical_heads",
        )
        missing = [path for path in experiment_dirs if not path.is_dir()]
        if missing:
            self.skipTest(
                "local compatibility artifacts are unavailable: "
                + ", ".join(map(str, missing))
            )
        reports = [inspector.inspect_run(path) for path in experiment_dirs]
        self.assertEqual(
            {report["experiment"] for report in reports},
            {"c2_t6_historical_heads", "c3a_t12_historical_heads"},
        )
        for report in reports:
            with self.subTest(experiment=report["experiment"]):
                self.assertTrue(report["supported"])
                self.assertGreater(report["encoder_output_dim"], 0)
                self.assertGreater(report["decoder_embedding_dim"], 0)
                expected = (
                    report["encoder_output_dim"]
                    != report["decoder_embedding_dim"]
                )
                self.assertEqual(report["projection_needed"], expected)
                if expected:
                    self.assertEqual(
                        report["projection_parameter_count"],
                        report["encoder_output_dim"]
                        * report["decoder_embedding_dim"]
                        + report["decoder_embedding_dim"],
                    )


if __name__ == "__main__":
    unittest.main()
