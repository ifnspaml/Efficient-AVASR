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
    Chapter3AVHubertModel,
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

    def forward_padding_mask(self, features, padding_mask):
        if tuple(padding_mask.shape) != tuple(features.shape[:2]):
            raise AssertionError("test padding mask is not aligned")
        return padding_mask


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


class _MismatchedPaddingStudent(_Student):
    def extract_distillation_features(
        self, source, padding_mask, *, return_intermediates
    ):
        representations, feature_penalty, output_mask = super().extract_distillation_features(
            source,
            padding_mask,
            return_intermediates=return_intermediates,
        )
        return representations, feature_penalty, ~output_mask


class _HistoricalHead(nn.Module):
    def forward(self, value):
        return value.unsqueeze(1).repeat(1, 4, 1, 1)


class _ProbeEncoder(nn.Module):
    def get_intermediate_outputs(self, value, padding_mask=None):
        return [value, value + 1.0]

    def forward(self, value, padding_mask=None, layer=None):
        return value + 1.0, []


class _FeatureProbe(nn.Module):
    extract_distillation_features = (
        Chapter3AVHubertModel.extract_distillation_features
    )

    def __init__(self):
        super().__init__()
        self.modality_dropout = 0.0
        self.audio_dropout = 0.0
        self.modality_fuse = "concat"
        self.layer_norm = nn.LayerNorm(4)
        self.post_extract_proj = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.post_extract_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )
        self.dropout_input = nn.Identity()
        self.encoder = _ProbeEncoder()

    def forward_features(self, source, modality):
        return source

    def forward_padding_mask(self, features, padding_mask):
        return padding_mask


def _source(frames: int = 3):
    return {
        "audio": torch.ones(1, 104, frames),
        "video": torch.ones(1, 1, frames, 2, 2),
    }


def _wrapper(student):
    checkpoint_cfg = SimpleNamespace(
        model=SimpleNamespace(), task=SimpleNamespace()
    )
    return AVHubertDistillOnly(
        teacher=_Teacher(),
        student=student,
        distillation_head=_HistoricalHead(),
        teacher_target_layers=(0, 4, 8, 12),
        student_match_layers=(),
        cfg=SimpleNamespace(
            distill_head_mode="historical_pred_heads",
            student_depth=student.depth,
            initialization_policy="unit_test",
        ),
        student_checkpoint_cfg=checkpoint_cfg,
        student_task_state={},
        initialization_report={},
    )


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


class HistoricalFeatureTest(unittest.TestCase):
    def test_penalty_uses_layer_normalized_fused_feature(self) -> None:
        probe = _FeatureProbe().eval()
        audio = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])
        video = torch.tensor([[[5.0, 4.0], [9.0, 8.0]]])
        source = {"audio": audio, "video": video}

        representations, feature_penalty, output_mask = (
            probe.extract_distillation_features(
                source,
                padding_mask=None,
                return_intermediates=True,
            )
        )
        fused = torch.cat([audio, video], dim=1).transpose(1, 2)
        normalized = probe.layer_norm(fused)
        projected = probe.post_extract_proj(normalized)

        torch.testing.assert_close(
            feature_penalty, normalized.float().pow(2).mean()
        )
        self.assertFalse(
            torch.isclose(feature_penalty, fused.float().pow(2).mean())
        )
        self.assertIsNone(output_mask)

        # Target identifier 0 is the input passed to the sequence encoder,
        # after fused LayerNorm and the optional representation projection.
        torch.testing.assert_close(representations[0], projected)
        torch.testing.assert_close(representations[1], projected + 1.0)


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
                result = model(source=_source(), padding_mask=None)
                self.assertFalse(student.return_intermediates)
                self.assertEqual(
                    tuple(result["student_hiddens"].shape), (1, 4, 3, 4)
                )

    def test_padded_wrapper_preserves_compatible_output_masks(self) -> None:
        padding_mask = torch.tensor([[False, False, True]])
        result = _wrapper(_Student(depth=2))(
            source=_source(), padding_mask=padding_mask
        )
        torch.testing.assert_close(result["padding_mask"], padding_mask)
        torch.testing.assert_close(
            result["teacher_padding_mask"], padding_mask
        )
        torch.testing.assert_close(
            result["student_padding_mask"], padding_mask
        )

    def test_padded_wrapper_rejects_bad_or_incompatible_masks(self) -> None:
        wrapper = _wrapper(_Student(depth=2))
        with self.assertRaises(ValueError):
            wrapper(
                source=_source(),
                padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            _wrapper(_MismatchedPaddingStudent(depth=2))(
                source=_source(),
                padding_mask=torch.tensor([[False, False, True]]),
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
