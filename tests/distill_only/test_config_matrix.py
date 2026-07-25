"""Static composition and safety tests for the public experiment matrix."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_PATH = REPO_ROOT / "scripts" / "distill_only" / "launch.py"
SPEC = importlib.util.spec_from_file_location("chapter3_test_launcher", LAUNCH_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class ConfigMatrixTest(unittest.TestCase):
    def test_every_public_config_is_pruning_free_and_composable(self) -> None:
        names = (
            "b1_t2_teacher_init",
            "b2_c1_t2_random_sequence",
            "c2_t6_historical_heads",
            "c3a_t12_historical_heads",
            "c3b_t12_layer_to_layer",
            "d1_selected_transformer",
            "d2_selected_conformer",
            "e1_selected_clean",
            "e2_selected_noisy",
            "s1_selected_main",
            "s2_optional_two_stage",
        )
        for name in names:
            with self.subTest(name=name):
                config = launcher.load_composed_config(
                    launcher.CONFIG_ROOT / f"{name}.yaml"
                )
                launcher.validate_safe_config(config)
                self.assertEqual(
                    config["model"]["teacher_target_layers"], "0,4,8,12"
                )
                self.assertEqual(config["optimizer"]["_name"], "adam")

    def test_main_and_optional_schedule_contracts(self) -> None:
        main = launcher.load_composed_config(
            launcher.CONFIG_ROOT / "c2_t6_historical_heads.yaml"
        )
        self.assertEqual(main["optimization"]["max_update"], 75000)
        self.assertEqual(main["optimization"]["lr"], [0.002])
        self.assertEqual(main["lr_scheduler"]["warmup_updates"], 15000)
        optional = launcher.load_composed_config(
            launcher.CONFIG_ROOT / "s2_optional_two_stage.yaml"
        )
        self.assertEqual(optional["optimization"]["max_update"], 50000)
        self.assertEqual(optional["model"]["schedule_mode"], "two_stage_50k_25k")

    def test_invalid_direct_mapping_is_rejected_before_launch(self) -> None:
        config = launcher.load_composed_config(
            launcher.CONFIG_ROOT / "c2_t6_historical_heads.yaml"
        )
        config["model"]["distill_head_mode"] = "layer_to_layer"
        config["model"]["student_match_layers"] = "0,4,8,12"
        with self.assertRaises(launcher.PreflightError):
            launcher.validate_safe_config(config)

    def test_effective_batch_override_must_remain_16000(self) -> None:
        config = launcher.load_composed_config(
            launcher.CONFIG_ROOT / "c2_t6_historical_heads.yaml"
        )
        config["dataset"]["max_tokens"] = 2000
        config["optimization"]["update_freq"] = [8]
        launcher.validate_safe_config(config)
        config["optimization"]["update_freq"] = [4]
        with self.assertRaises(launcher.PreflightError):
            launcher.validate_safe_config(config)

    def test_checkpoint_final_update_is_read_and_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(
                {"optimizer_history": [{"num_updates": 75000}]},
                checkpoint,
            )
            self.assertEqual(
                launcher._checkpoint_num_updates(checkpoint), 75000
            )
            torch.save({"optimizer_history": []}, checkpoint)
            with self.assertRaises(launcher.PreflightError):
                launcher._checkpoint_num_updates(checkpoint)


if __name__ == "__main__":
    unittest.main()
