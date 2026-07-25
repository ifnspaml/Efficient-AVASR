"""Loss-equivalence and generic mapping-validation tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from chapter3_distill_only.criterion import AVHubertDistillOnlyCriterion
from chapter3_distill_only.model import AVHubertDistillOnly
from chapter3_distill_only.student import (
    parse_layer_ids,
    validate_architecture,
    validate_target_mapping,
)


class _StaticDistillationModel:
    def __init__(self, student: torch.Tensor, teacher: torch.Tensor):
        self.student = student
        self.teacher = teacher

    def __call__(self, **unused):
        return {
            "student_hiddens": self.student,
            "teacher_hiddens": self.teacher,
            "student_features_pen": self.student.new_zeros(()),
            "teacher_target_layers": (0, 4, 8, 12),
        }


class _Teacher(nn.Module):
    def extract_intermediate_features(self, **unused):
        value = torch.ones(1, 3, 4)
        return [value + index for index in range(13)]


class _Student(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.return_intermediates = None

    def extract_distillation_features(
        self, source, padding_mask, *, return_intermediates
    ):
        self.return_intermediates = return_intermediates
        value = torch.ones(1, 3, 4)
        representations = (
            [value + index for index in range(self.depth + 1)]
            if return_intermediates
            else value
        )
        return representations, value.pow(2).mean(), padding_mask


class _HistoricalHead(nn.Module):
    def forward(self, value):
        return value.unsqueeze(1).repeat(1, 4, 1, 1)


class HistoricalLossTest(unittest.TestCase):
    def test_l1_plus_log_sigmoid_cosine(self) -> None:
        student = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[-1.0, 0.0]]]]
        )
        teacher = torch.tensor(
            [[[[0.0, 1.0]], [[0.0, 1.0]], [[1.0, -1.0]], [[1.0, 0.0]]]]
        )
        criterion = AVHubertDistillOnlyCriterion(
            task=None,
            distill_loss_type="historical",
            l1_weight=1.0,
            l2_weight=0.0,
            cosine_weight=1.0,
            cosine_type="log_sig",
            feature_penalty_weight=0.0,
        )
        loss, sample_size, logging = criterion(
            _StaticDistillationModel(student, teacher),
            {"id": torch.tensor([0, 1]), "net_input": {}},
        )
        expected = F.l1_loss(student, teacher) - F.logsigmoid(
            F.cosine_similarity(student, teacher, dim=-1)
        ).mean()
        torch.testing.assert_close(loss, expected)
        # Mean historical losses use one Fairseq gradient denominator.
        self.assertEqual(sample_size, 1)
        self.assertEqual(logging["nsentences"], 2)


class MappingValidationTest(unittest.TestCase):
    def test_historical_mode_does_not_require_student_targets(self) -> None:
        validate_target_mapping(
            teacher_target_layers=(0, 4, 8, 12),
            student_match_layers=(),
            distill_head_mode="historical_pred_heads",
            teacher_depth=12,
            student_depth=2,
        )

    def test_unequal_depth_historical_forward_never_requests_intermediates(self) -> None:
        for depth in (2, 6):
            with self.subTest(depth=depth):
                student = _Student(depth)
                checkpoint_cfg = SimpleNamespace(
                    model=SimpleNamespace(), task=SimpleNamespace()
                )
                model = AVHubertDistillOnly(
                    teacher=_Teacher(),
                    student=student,
                    distillation_head=_HistoricalHead(),
                    teacher_target_layers=(0, 4, 8, 12),
                    student_match_layers=(),
                    cfg=SimpleNamespace(
                        distill_head_mode="historical_pred_heads",
                        student_depth=depth,
                        initialization_policy="unit_test",
                    ),
                    student_checkpoint_cfg=checkpoint_cfg,
                    student_task_state={},
                    initialization_report={},
                )
                result = model(source={}, padding_mask=None)
                self.assertFalse(student.return_intermediates)
                self.assertEqual(
                    tuple(result["student_hiddens"].shape), (1, 4, 3, 4)
                )

    def test_layer_mode_checks_student_bounds(self) -> None:
        with self.assertRaises(ValueError):
            validate_target_mapping(
                teacher_target_layers=(0, 4, 8, 12),
                student_match_layers=(0, 4, 8, 12),
                distill_head_mode="layer_to_layer",
                teacher_depth=12,
                student_depth=6,
            )

    def test_parser_and_architecture_validation(self) -> None:
        self.assertEqual(
            parse_layer_ids("0,4,8,12", "targets"), (0, 4, 8, 12)
        )
        with self.assertRaises(ValueError):
            parse_layer_ids("0,4,4", "targets")
        with self.assertRaises(ValueError):
            validate_architecture("transformer", 2, 385, 512, 12, 31)


if __name__ == "__main__":
    unittest.main()
