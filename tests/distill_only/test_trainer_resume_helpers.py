"""Unit tests for trainer checkpoint/resume state comparisons."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


HELPERS_PATH = Path(__file__).with_name("trainer_resume_helpers.py")
SPEC = importlib.util.spec_from_file_location(
    "chapter3_trainer_resume_helpers", HELPERS_PATH
)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def _snapshot(value: int = 1) -> dict:
    return {
        "model_state": {"weight": torch.tensor([value, 2.0])},
        "optimizer_state": {
            "state": {0: {"step": torch.tensor(float(value))}},
            "param_groups": [{"lr": 0.1}],
        },
        "lr_scheduler_state": {"best": None},
        "num_updates": value,
        "learning_rate": 0.1,
        "iterator_state": {
            "version": 2,
            "epoch": 1,
            "iterations_in_epoch": value,
            "shuffle": False,
        },
        "meter_state": {"sum": 7.0, "count": 2.0},
    }


class StateDigestTest(unittest.TestCase):
    def test_mapping_order_does_not_change_digest(self) -> None:
        first = {"b": [torch.tensor([1, 2]), None], "a": (0.0, True)}
        second = {"a": (0.0, True), "b": [torch.tensor([1, 2]), None]}
        self.assertEqual(
            helpers.state_digest(first),
            helpers.state_digest(second),
        )

    def test_tensor_value_and_dtype_change_digest(self) -> None:
        baseline = helpers.state_digest(torch.tensor([1, 2], dtype=torch.int64))
        self.assertNotEqual(
            baseline,
            helpers.state_digest(torch.tensor([1, 3], dtype=torch.int64)),
        )
        self.assertNotEqual(
            baseline,
            helpers.state_digest(torch.tensor([1, 2], dtype=torch.int32)),
        )


class ResumeContractTest(unittest.TestCase):
    def test_exact_within_stage_snapshot_passes(self) -> None:
        expected = _snapshot()
        actual = _snapshot()
        helpers.require_within_stage_resume(expected=expected, actual=actual)

    def test_optimizer_change_is_rejected(self) -> None:
        expected = _snapshot()
        actual = _snapshot()
        actual["optimizer_state"]["param_groups"][0]["lr"] = 0.2
        with self.assertRaisesRegex(
            helpers.ResumeStateError, "optimizer_state"
        ):
            helpers.require_within_stage_resume(
                expected=expected,
                actual=actual,
            )

    def test_s2_requires_weight_transfer_and_fresh_training_state(self) -> None:
        stage1 = _snapshot()
        fresh = _snapshot(value=0)
        fresh["num_updates"] = 0
        fresh["iterator_state"]["iterations_in_epoch"] = 0
        loaded = _snapshot(value=0)
        loaded["model_state"] = stage1["model_state"]
        loaded["num_updates"] = 0
        loaded["iterator_state"]["iterations_in_epoch"] = 0
        helpers.require_s2_warm_start(
            stage1_model_state=stage1["model_state"],
            stage1_iterator_state=stage1["iterator_state"],
            fresh_stage2=fresh,
            loaded_stage2=loaded,
        )

        loaded["num_updates"] = 1
        with self.assertRaisesRegex(helpers.ResumeStateError, "update zero"):
            helpers.require_s2_warm_start(
                stage1_model_state=stage1["model_state"],
                stage1_iterator_state=stage1["iterator_state"],
                fresh_stage2=fresh,
                loaded_stage2=loaded,
            )

    def test_iterator_validation_rejects_invalid_or_missing_state(self) -> None:
        with self.assertRaises(helpers.ResumeStateError):
            helpers.iterator_position({"epoch": 1})
        with self.assertRaises(helpers.ResumeStateError):
            helpers.iterator_position(
                {"epoch": 0, "iterations_in_epoch": 0}
            )


if __name__ == "__main__":
    unittest.main()
