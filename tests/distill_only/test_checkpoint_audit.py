"""Tests for checkpoint-chain and Fairseq-log diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from chapter3_distill_only.checkpoint_audit import (
    compare_mapped_tensors,
    projection_tensors,
    summarize_training_tail,
    summarize_validation_log,
)


class CheckpointAuditTest(unittest.TestCase):
    def test_exact_prefix_handoff(self) -> None:
        weight = torch.arange(6, dtype=torch.float32).view(2, 3)
        source = {
            "student.layer.weight": weight,
            "student.norm.running_mean": torch.zeros(2),
        }
        target = {
            "layer.weight": weight.clone(),
            "norm.running_mean": torch.zeros(2),
        }
        result = compare_mapped_tensors(
            source,
            target,
            source_prefix="student.",
            names=("layer.weight", "norm.running_mean"),
        )
        self.assertTrue(result["parameters_exact"])
        self.assertTrue(result["all_tensors_exact"])

    def test_parameter_and_buffer_mismatches_are_separate(self) -> None:
        source = {
            "weight": torch.ones(2),
            "bn.running_mean": torch.zeros(2),
        }
        target = {
            "encoder.weight": torch.zeros(2),
            "encoder.bn.running_mean": torch.ones(2),
        }
        result = compare_mapped_tensors(
            source,
            target,
            target_prefix="encoder.",
        )
        self.assertEqual(result["parameter_mismatches"], ["weight"])
        self.assertEqual(
            result["mutable_buffer_mismatches"], ["bn.running_mean"]
        )
        self.assertFalse(result["parameters_exact"])
        self.assertFalse(result["parameters_close"])

    def test_fp16_round_trip_is_close_but_not_exact(self) -> None:
        source = {"weight": torch.tensor([0.123456, -2.34567])}
        target = {"weight": source["weight"].half().float()}
        result = compare_mapped_tensors(source, target)
        self.assertFalse(result["parameters_exact"])
        self.assertTrue(result["parameters_close"])

    def test_projection_detection(self) -> None:
        state = {
            "encoder.proj.weight": torch.zeros(768, 384),
            "encoder.proj.bias": torch.zeros(768),
            "decoder.layer.weight": torch.zeros(1),
        }
        self.assertEqual(
            projection_tensors(state),
            {
                "encoder.proj.weight": [768, 384],
                "encoder.proj.bias": [768],
            },
        )

    def test_log_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "\n".join(
                    (
                        '[x][valid][INFO] - {"valid_num_updates": "10", '
                        '"valid_accuracy": "70", "valid_loss": "3"}',
                        '[x][valid][INFO] - {"valid_num_updates": "20", '
                        '"valid_accuracy": "60", "valid_loss": "4"}',
                        '[x][train_inner][INFO] - {"num_updates": "20", '
                        '"accuracy": "50", "gnorm": "12.5", '
                        '"loss_scale": "64", "lr": "0.001"}',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            validation = summarize_validation_log(path)
            self.assertEqual(validation["best"]["update"], 10)
            self.assertEqual(validation["final"]["accuracy"], 60.0)
            tail = summarize_training_tail(path)
            self.assertEqual(tail["gradient_norm"], 12.5)
            self.assertEqual(tail["update"], 20)


if __name__ == "__main__":
    unittest.main()
