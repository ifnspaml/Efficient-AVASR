"""Distillation task with explicit train-only noise isolation."""

from dataclasses import dataclass, field
from typing import Optional

from fairseq.tasks import register_task
from omegaconf import DictConfig, open_dict

from avhubert.hubert_pretraining import (
    AVHubertPretrainingConfig,
    AVHubertPretrainingTask,
)

DISTILL_TASK_NAME = "av_hubert_distill_only"


@dataclass
class AVHubertDistillOnlyTaskConfig(AVHubertPretrainingConfig):
    distillation_noise_prob: Optional[float] = field(default=None)
    distillation_noise_snr: Optional[str] = field(default=None)
    distillation_noise_method: Optional[str] = field(default=None)
    distillation_noise_manifest_root: Optional[str] = field(default=None)
    distillation_noise_train_only: bool = field(default=True)


@register_task(DISTILL_TASK_NAME, dataclass=AVHubertDistillOnlyTaskConfig)
class AVHubertDistillOnlyTask(AVHubertPretrainingTask):
    """Use the normal AV-HuBERT dataset while isolating distillation noise."""

    @staticmethod
    def _assign(cfg, name, value) -> None:
        if isinstance(cfg, DictConfig):
            with open_dict(cfg):
                cfg[name] = value
        else:
            setattr(cfg, name, value)

    def load_dataset(self, split: str, **kwargs) -> None:
        original = {
            "noise_prob": self.cfg.noise_prob,
            "noise_snr": self.cfg.noise_snr,
            "noise_method": self.cfg.noise_method,
            "noise_wav": self.cfg.noise_wav,
        }
        try:
            probability = (
                self.cfg.distillation_noise_prob
                if self.cfg.distillation_noise_prob is not None
                else self.cfg.noise_prob
            )
            snr = (
                self.cfg.distillation_noise_snr
                if self.cfg.distillation_noise_snr is not None
                else self.cfg.noise_snr
            )
            method = (
                self.cfg.distillation_noise_method
                if self.cfg.distillation_noise_method is not None
                else self.cfg.noise_method
            )
            manifest_root = (
                self.cfg.distillation_noise_manifest_root
                if self.cfg.distillation_noise_manifest_root is not None
                else self.cfg.noise_wav
            )
            if split != "train" and self.cfg.distillation_noise_train_only:
                probability = 0.0
                manifest_root = None

            self._assign(self.cfg, "noise_prob", probability)
            self._assign(self.cfg, "noise_snr", str(snr))
            self._assign(self.cfg, "noise_method", method)
            self._assign(self.cfg, "noise_wav", manifest_root)
            super().load_dataset(split, **kwargs)
        finally:
            for name, value in original.items():
                self._assign(self.cfg, name, value)

        dataset = self.datasets[split]
        if split != "train" and self.cfg.distillation_noise_train_only:
            if dataset.noise_prob != 0 or dataset.noise_wav:
                raise AssertionError(
                    f"Noise leaked into non-training split {split!r}"
                )
