"""Tests for train-only distillation-noise split isolation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from avhubert.hubert_pretraining import AVHubertPretrainingTask
from chapter3_distill_only.task import AVHubertDistillOnlyTask


class NoiseIsolationTest(unittest.TestCase):
    def test_validation_never_loads_noise_and_config_is_restored(self) -> None:
        task = object.__new__(AVHubertDistillOnlyTask)
        task.cfg = SimpleNamespace(
            noise_prob=0.25,
            noise_snr="0",
            noise_method="rms",
            noise_wav="/fixed/noise",
            distillation_noise_prob=0.25,
            distillation_noise_snr="0",
            distillation_noise_method="rms",
            distillation_noise_manifest_root="/fixed/noise",
            distillation_noise_train_only=True,
        )
        task.datasets = {}
        observed = {}

        def fake_parent(instance, split, **kwargs):
            observed[split] = {
                "probability": instance.cfg.noise_prob,
                "root": instance.cfg.noise_wav,
            }
            instance.datasets[split] = SimpleNamespace(
                noise_prob=instance.cfg.noise_prob,
                noise_wav=[] if instance.cfg.noise_wav is None else ["noise"],
            )

        with patch.object(
            AVHubertPretrainingTask, "load_dataset", new=fake_parent
        ):
            task.load_dataset("valid")
            task.load_dataset("train")

        self.assertEqual(observed["valid"], {"probability": 0.0, "root": None})
        self.assertEqual(
            observed["train"],
            {"probability": 0.25, "root": "/fixed/noise"},
        )
        self.assertEqual(task.cfg.noise_prob, 0.25)
        self.assertEqual(task.cfg.noise_wav, "/fixed/noise")


if __name__ == "__main__":
    unittest.main()
