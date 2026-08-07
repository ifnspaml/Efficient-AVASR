"""Tests for train-only distillation-noise split isolation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from avhubert.hubert_pretraining import AVHubertPretrainingTask
from chapter3_distill_only.model import AVHubertDistillOnly
from chapter3_distill_only.task import AVHubertDistillOnlyTask


class NoiseIsolationTest(unittest.TestCase):
    def test_validation_never_loads_noise_and_config_is_restored(self) -> None:
        task = object.__new__(AVHubertDistillOnlyTask)
        task.cfg = SimpleNamespace(
            noise_prob=0.0,
            noise_snr="0",
            noise_method="rms",
            noise_wav=None,
            distillation_noise_prob=0.25,
            distillation_noise_snr="0",
            distillation_noise_method="rms",
            distillation_noise_manifest_root="/fixed/noise",
            distillation_noise_train_only=True,
            distillation_noise_student_only=False,
        )
        task.datasets = {}
        observed = {}

        def fake_parent(instance, split, **kwargs):
            observed[split] = {
                "probability": instance.cfg.noise_prob,
                "root": instance.cfg.noise_wav,
                "method": instance.cfg.noise_method,
            }
            instance.datasets[split] = SimpleNamespace(
                noise_prob=instance.cfg.noise_prob,
                noise_wav=[] if instance.cfg.noise_wav is None else ["noise"],
                provide_clean_audio=False,
            )

        with patch.object(
            AVHubertPretrainingTask, "load_dataset", new=fake_parent
        ):
            task.load_dataset("valid")
            task.load_dataset("train")

        self.assertEqual(
            observed["valid"],
            {"probability": 0.0, "root": None, "method": "rms"},
        )
        self.assertEqual(
            observed["train"],
            {"probability": 0.25, "root": "/fixed/noise", "method": "rms"},
        )
        self.assertEqual(task.cfg.noise_prob, 0.0)
        self.assertIsNone(task.cfg.noise_wav)
        self.assertEqual(task.cfg.noise_method, "rms")
        self.assertFalse(task.datasets["train"].provide_clean_audio)
        self.assertFalse(task.datasets["valid"].provide_clean_audio)

    def test_student_only_enables_clean_audio_on_noisy_train(self) -> None:
        task = object.__new__(AVHubertDistillOnlyTask)
        task.cfg = SimpleNamespace(
            noise_prob=0.0,
            noise_snr="0",
            noise_method="rms",
            noise_wav=None,
            distillation_noise_prob=0.25,
            distillation_noise_snr="0",
            distillation_noise_method="rms",
            distillation_noise_manifest_root="/fixed/noise",
            distillation_noise_train_only=True,
            distillation_noise_student_only=True,
        )
        task.datasets = {}

        def fake_parent(instance, split, **kwargs):
            instance.datasets[split] = SimpleNamespace(
                noise_prob=instance.cfg.noise_prob,
                noise_wav=[] if instance.cfg.noise_wav is None else ["noise"],
                provide_clean_audio=False,
            )

        with patch.object(
            AVHubertPretrainingTask, "load_dataset", new=fake_parent
        ):
            task.load_dataset("valid")
            task.load_dataset("train")

        self.assertTrue(task.datasets["train"].provide_clean_audio)
        self.assertFalse(task.datasets["valid"].provide_clean_audio)


class TeacherStudentSourceRoutingTest(unittest.TestCase):
    def test_audio_clean_routes_to_teacher_only(self) -> None:
        import torch

        noisy = torch.randn(2, 80, 10)
        clean = torch.randn(2, 80, 10)
        video = torch.randn(2, 1, 10, 88, 88)
        source = {"audio": noisy, "video": video, "audio_clean": clean}

        teacher_source, student_source = AVHubertDistillOnly._teacher_student_sources(
            source
        )
        self.assertIs(teacher_source["audio"], clean)
        self.assertIs(student_source["audio"], noisy)
        self.assertIs(teacher_source["video"], video)
        self.assertIs(student_source["video"], video)

    def test_shared_source_when_audio_clean_absent(self) -> None:
        import torch

        audio = torch.randn(2, 80, 10)
        video = torch.randn(2, 1, 10, 88, 88)
        source = {"audio": audio, "video": video}
        teacher_source, student_source = AVHubertDistillOnly._teacher_student_sources(
            source
        )
        self.assertIs(teacher_source, source)
        self.assertIs(student_source, source)

    def test_forward_passes_routed_sources(self) -> None:
        import torch

        model = object.__new__(AVHubertDistillOnly)
        model.teacher_target_layers = (0,)
        model.distill_head_mode = "historical_pred_heads"
        model.cfg = SimpleNamespace(student_depth=2)

        teacher = MagicMock()
        student = MagicMock()
        head = MagicMock()
        model.teacher = teacher
        model.student = student
        model.distillation_head = head

        hidden = torch.zeros(2, 4, 8)
        teacher.extract_intermediate_features.return_value = [hidden]
        teacher.forward_padding_mask.return_value = torch.zeros(
            2, 4, dtype=torch.bool
        )
        student.extract_distillation_features.return_value = (
            hidden,
            torch.tensor(0.0),
            torch.zeros(2, 4, dtype=torch.bool),
        )
        head.return_value = hidden.unsqueeze(1)

        noisy = torch.randn(2, 80, 10)
        clean = torch.randn(2, 80, 10)
        video = torch.randn(2, 1, 10, 88, 88)
        padding = torch.zeros(2, 10, dtype=torch.bool)
        source = {"audio": noisy, "video": video, "audio_clean": clean}

        model.forward(source, padding_mask=padding)

        teacher_call = teacher.extract_intermediate_features.call_args.kwargs
        student_call = student.extract_distillation_features.call_args.kwargs
        self.assertIs(teacher_call["source"]["audio"], clean)
        self.assertIs(student_call["source"]["audio"], noisy)
        self.assertIs(teacher_call["source"]["video"], video)
        self.assertIs(student_call["source"]["video"], video)


if __name__ == "__main__":
    unittest.main()
