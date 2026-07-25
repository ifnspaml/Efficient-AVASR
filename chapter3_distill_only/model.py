"""Registered distillation-only wrapper for the Chapter 3 experiment suite."""

import copy
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from omegaconf import II, MISSING, open_dict

from .heads import HistoricalPredictionHeads, LayerToLayerAdapters
from .student import (
    Chapter3AVHubertModel,
    build_student_checkpoint_config,
    initialize_student,
    parse_layer_ids,
    strip_pretraining_targets,
    validate_architecture,
    validate_target_mapping,
)

logger = logging.getLogger(__name__)

DISTILL_MODEL_NAME = "av_hubert_distill_only"


@dataclass
class AVHubertDistillOnlyConfig(FairseqDataclass):
    """Configuration for an ordinary, non-pruning distillation run."""

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    teacher_path: str = field(
        default=MISSING, metadata={"help": "frozen AV-HuBERT teacher checkpoint"}
    )

    student_arch: str = field(default="transformer")
    student_depth: int = field(default=2)
    student_embed_dim: int = field(default=768)
    student_ffn_dim: int = field(default=3072)
    student_attention_heads: int = field(default=12)
    student_conformer_kernel: int = field(default=31)
    student_conformer_attention_type: str = field(default="original")
    student_conformer_position_type: str = field(default="abs")
    student_dropout: float = field(default=0.1)
    student_attention_dropout: float = field(default=0.1)
    student_activation_dropout: float = field(default=0.1)
    student_layerdrop: float = field(default=0.0)
    student_layer_norm_first: bool = field(default=False)
    student_dropout_input: float = field(default=0.0)
    student_conv_pos: int = field(default=128)
    student_conv_pos_groups: int = field(default=16)

    distill_head_mode: str = field(default="historical_pred_heads")
    prediction_head_hidden_dim: int = field(default=-1)
    teacher_target_layers: str = field(default="0,4,8,12")
    student_match_layers: str = field(default="")
    distill_loss_type: str = field(default="historical")
    initialization_policy: str = field(
        default="random_sequence_teacher_frontend"
    )

    # Kept in model metadata as well as criterion config so exported manifests
    # can audit the fully resolved scientific protocol.
    l1_weight: float = field(default=1.0)
    l2_weight: float = field(default=0.0)
    cosine_weight: float = field(default=1.0)
    cosine_type: str = field(default="log_sig")
    feature_penalty_weight: float = field(default=0.0)

    modality_dropout: float = field(default=0.0)
    audio_dropout: float = field(default=0.0)
    feature_grad_mult: float = field(default=1.0)

    # Protocol metadata. Fairseq's top-level optimizer/scheduler remains the
    # source of execution truth; launch preflight checks these mirrors.
    schedule_mode: str = field(default="continuous_75k")
    stage1_lr: float = field(default=0.002)
    stage2_lr: float = field(default=0.0001)
    max_update: int = field(default=75000)
    warmup_updates: int = field(default=15000)
    distillation_noise_prob: float = field(default=0.0)
    distillation_noise_snr: str = field(default="0")
    distillation_noise_method: str = field(default="rms")
    distillation_noise_manifest_root: Optional[str] = field(default=None)
    distillation_noise_train_only: bool = field(default=True)

    # Serialized checkpoint construction configs. They avoid relying on
    # architecture-specific code paths when resuming.
    teacher_args: Any = None
    student_args: Any = None


@register_model(DISTILL_MODEL_NAME, dataclass=AVHubertDistillOnlyConfig)
class AVHubertDistillOnly(BaseFairseqModel):
    """Frozen teacher, configurable student, and selectable distillation head."""

    def __init__(
        self,
        teacher,
        student,
        distillation_head,
        teacher_target_layers,
        student_match_layers,
        cfg,
        student_checkpoint_cfg,
        student_task_state,
        initialization_report,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.distillation_head = distillation_head
        self.teacher_target_layers = tuple(teacher_target_layers)
        self.student_match_layers = tuple(student_match_layers)
        self.cfg = cfg
        self.distill_head_mode = cfg.distill_head_mode
        self.initialization_report = initialization_report

        # Stable exporter interface.
        self.student_checkpoint_cfg = student_checkpoint_cfg
        self.student_cfg = student_checkpoint_cfg.model
        self.student_task_cfg = student_checkpoint_cfg.task
        self.student_task_state = copy.deepcopy(student_task_state)
        self.num_updates = 0

    @staticmethod
    def _checkpoint_args(state: Dict[str, Any]) -> Any:
        checkpoint_cfg = state.get("cfg")
        if checkpoint_cfg is None:
            checkpoint_cfg = convert_namespace_to_omegaconf(state["args"])
        if isinstance(checkpoint_cfg, Namespace):
            checkpoint_cfg = convert_namespace_to_omegaconf(checkpoint_cfg)
        return checkpoint_cfg

    @staticmethod
    def _load_avhubert_state(model, state: Dict[str, Any]) -> None:
        checkpoint_model = dict(state["model"])
        # The model is built from this exact fixed checkpoint configuration,
        # before removing any pretraining modules, so every key must match.
        model.load_state_dict(checkpoint_model, strict=True)

    @classmethod
    def build_model(cls, cfg, task):
        validate_architecture(
            cfg.student_arch,
            cfg.student_depth,
            cfg.student_embed_dim,
            cfg.student_ffn_dim,
            cfg.student_attention_heads,
            cfg.student_conformer_kernel,
        )
        teacher_targets = parse_layer_ids(
            cfg.teacher_target_layers, "teacher_target_layers"
        )
        student_matches = parse_layer_ids(
            cfg.student_match_layers, "student_match_layers"
        )

        teacher_state = checkpoint_utils.load_checkpoint_to_cpu(cfg.teacher_path)
        teacher_args = cls._checkpoint_args(teacher_state)
        teacher_args = copy.deepcopy(teacher_args)
        if cfg.normalize != teacher_args.task.normalize:
            raise ValueError(
                "Task normalization must match the frozen teacher checkpoint"
            )
        with open_dict(teacher_args.task):
            teacher_args.task.data = cfg.data
        with open_dict(cfg):
            cfg.teacher_args = copy.deepcopy(teacher_args)

        teacher_task = tasks.setup_task(teacher_args.task)
        teacher_task_state = teacher_state.get("task_state", {})
        if teacher_task_state:
            teacher_task.load_state_dict(teacher_task_state)
        teacher = teacher_task.build_model(teacher_args.model)
        cls._load_avhubert_state(teacher, teacher_state)
        teacher.remove_pretraining_modules()
        if hasattr(teacher.encoder, "layerdrop"):
            teacher.encoder.layerdrop = 0.0
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

        if cfg.student_args is None:
            student_args = build_student_checkpoint_config(teacher_args, cfg)
        else:
            student_args = copy.deepcopy(cfg.student_args)
            with open_dict(student_args.task):
                student_args.task.data = cfg.data
        with open_dict(cfg):
            cfg.student_args = copy.deepcopy(student_args)

        student_task = tasks.setup_task(student_args.task)
        if teacher_task_state:
            student_task.load_state_dict(teacher_task_state)
        student = student_task.build_model(student_args.model)
        if not isinstance(student, Chapter3AVHubertModel):
            raise TypeError(
                "Native student construction must produce "
                f"Chapter3AVHubertModel, received {type(student).__name__}"
            )

        initialization_report = initialize_student(
            student, teacher, cfg.initialization_policy
        )
        strip_pretraining_targets(student)

        teacher_depth = len(teacher.encoder.layers)
        validate_target_mapping(
            teacher_targets,
            student_matches,
            cfg.distill_head_mode,
            teacher_depth,
            cfg.student_depth,
        )
        teacher_dim = teacher.encoder_embed_dim
        if cfg.distill_head_mode == "historical_pred_heads":
            distillation_head = HistoricalPredictionHeads(
                student_dim=cfg.student_embed_dim,
                teacher_dim=teacher_dim,
                target_layers=teacher_targets,
                hidden_dim=cfg.prediction_head_hidden_dim,
            )
        else:
            distillation_head = LayerToLayerAdapters(
                student_dim=cfg.student_embed_dim,
                teacher_dim=teacher_dim,
                teacher_target_layers=teacher_targets,
                student_match_layers=student_matches,
            )

        return cls(
            teacher=teacher,
            student=student,
            distillation_head=distillation_head,
            teacher_target_layers=teacher_targets,
            student_match_layers=student_matches,
            cfg=cfg,
            student_checkpoint_cfg=student_args,
            student_task_state=teacher_task_state,
            initialization_report=initialization_report,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # ``nn.Module.train`` recurses into every child, so restore the frozen
        # teacher to evaluation mode after that recursion.
        self.teacher.eval()
        return self

    @staticmethod
    def _validate_hidden_shapes(teacher_hiddens, student_hiddens) -> None:
        if teacher_hiddens.shape != student_hiddens.shape:
            raise ValueError(
                "Teacher and student distillation tensors must have identical "
                f"shapes, got teacher={tuple(teacher_hiddens.shape)} and "
                f"student={tuple(student_hiddens.shape)}"
            )
        if teacher_hiddens.ndim != 4:
            raise ValueError(
                "Distillation tensors must be B x N x T x D, got "
                f"{tuple(teacher_hiddens.shape)}"
            )

    @staticmethod
    def _validate_source_padding(
        source: Mapping[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> Tuple[int, int]:
        if not isinstance(source, Mapping):
            raise TypeError("source must be a mapping containing audio/video")
        modality_shapes = []
        for name, time_axis, expected_rank in (
            ("audio", 2, 3),
            ("video", 2, 5),
        ):
            value = source.get(name)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.ndim != expected_rank:
                raise ValueError(
                    f"source[{name!r}] must have rank {expected_rank}, "
                    f"received {type(value).__name__} "
                    f"{getattr(value, 'shape', None)}"
                )
            modality_shapes.append((name, value.size(0), value.size(time_axis)))
        if not modality_shapes:
            raise ValueError("source must contain at least one tensor modality")
        batch, frames = modality_shapes[0][1:]
        for name, current_batch, current_frames in modality_shapes[1:]:
            if (current_batch, current_frames) != (batch, frames):
                raise ValueError(
                    "Audio/video batch and time dimensions must match; "
                    f"expected {(batch, frames)}, got "
                    f"{(current_batch, current_frames)} for {name}"
                )
        if padding_mask is not None:
            if not torch.is_tensor(padding_mask):
                raise TypeError("padding_mask must be a tensor or None")
            if padding_mask.dtype != torch.bool:
                raise ValueError(
                    f"padding_mask must be bool, got {padding_mask.dtype}"
                )
            if tuple(padding_mask.shape) != (batch, frames):
                raise ValueError(
                    "Input padding_mask must have shape B x T matching source; "
                    f"expected {(batch, frames)}, got "
                    f"{tuple(padding_mask.shape)}"
                )
        return batch, frames

    @staticmethod
    def _validate_output_padding(
        name: str,
        output_padding_mask: Optional[torch.Tensor],
        hiddens: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor],
    ) -> None:
        expected = (hiddens.size(0), hiddens.size(2))
        if input_padding_mask is None:
            if output_padding_mask is not None:
                raise ValueError(
                    f"{name} produced a padding mask without an input mask"
                )
            return
        if output_padding_mask is None:
            raise ValueError(f"{name} dropped the input padding mask")
        if output_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"{name} output padding mask must be bool, "
                f"got {output_padding_mask.dtype}"
            )
        if tuple(output_padding_mask.shape) != expected:
            raise ValueError(
                f"{name} output padding mask must match representation B x T "
                f"{expected}, got {tuple(output_padding_mask.shape)}"
            )

    def forward(self, source, padding_mask=None, **unused):
        self._validate_source_padding(source, padding_mask)
        self.teacher.eval()
        with torch.no_grad():
            teacher_all = self.teacher.extract_intermediate_features(
                source=source,
                padding_mask=padding_mask,
                mask=False,
            )
            if max(self.teacher_target_layers) >= len(teacher_all):
                raise ValueError(
                    "Teacher did not return every configured target: "
                    f"requested {self.teacher_target_layers}, "
                    f"available 0..{len(teacher_all) - 1}"
                )
            teacher_hiddens = torch.stack(
                [teacher_all[layer] for layer in self.teacher_target_layers],
                dim=1,
            )
            teacher_padding_mask = (
                self.teacher.forward_padding_mask(
                    teacher_all[0], padding_mask
                )
                if padding_mask is not None
                else None
            )

        if self.distill_head_mode == "historical_pred_heads":
            student_final, feature_penalty, student_padding_mask = (
                self.student.extract_distillation_features(
                    source=source,
                    padding_mask=padding_mask,
                    return_intermediates=False,
                )
            )
            student_hiddens = self.distillation_head(student_final)
        elif self.distill_head_mode == "layer_to_layer":
            student_all, feature_penalty, student_padding_mask = (
                self.student.extract_distillation_features(
                    source=source,
                    padding_mask=padding_mask,
                    return_intermediates=True,
                )
            )
            if len(student_all) != self.cfg.student_depth + 1:
                raise ValueError(
                    "Student intermediate-output contract violated: expected "
                    f"{self.cfg.student_depth + 1} tensors, got {len(student_all)}"
                )
            if max(self.student_match_layers) >= len(student_all):
                raise ValueError(
                    "Student did not return every configured match layer: "
                    f"requested {self.student_match_layers}, "
                    f"available 0..{len(student_all) - 1}"
                )
            student_hiddens = self.distillation_head(
                student_all[layer] for layer in self.student_match_layers
            )
        else:
            raise RuntimeError(
                f"Unsupported distill_head_mode {self.distill_head_mode!r}"
            )

        self._validate_hidden_shapes(teacher_hiddens, student_hiddens)
        self._validate_output_padding(
            "teacher",
            teacher_padding_mask,
            teacher_hiddens,
            padding_mask,
        )
        self._validate_output_padding(
            "student",
            student_padding_mask,
            student_hiddens,
            padding_mask,
        )
        if teacher_padding_mask is None:
            if student_padding_mask is not None:
                raise ValueError(
                    "Teacher/student output padding masks are incompatible"
                )
        elif not torch.equal(teacher_padding_mask, student_padding_mask):
            raise ValueError(
                "Teacher/student output padding masks differ after feature "
                "alignment"
            )
        return {
            "teacher_hiddens": teacher_hiddens,
            "student_hiddens": student_hiddens,
            "student_features_pen": feature_penalty,
            "teacher_target_layers": self.teacher_target_layers,
            "padding_mask": student_padding_mask,
            "teacher_padding_mask": teacher_padding_mask,
            "student_padding_mask": student_padding_mask,
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def get_student_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.student.parameters())

    def get_prediction_head_num_params(self) -> int:
        return sum(
            parameter.numel() for parameter in self.distillation_head.parameters()
        )

    def get_teacher_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.teacher.parameters())

    def get_export_state(self) -> Dict[str, Any]:
        """Return a student-native Fairseq checkpoint payload.

        The payload deliberately excludes the frozen teacher and all
        distillation heads.  Export tooling may add trainer metadata, hashes,
        and atomic-write handling around this core state.
        """

        return {
            "cfg": copy.deepcopy(self.student_checkpoint_cfg),
            "model": self.student.state_dict(),
            "task_state": copy.deepcopy(self.student_task_state),
            "optimizer_history": [
                {
                    "criterion_name": "av_hubert_distill_only",
                    "optimizer_name": "Adam",
                    "lr_scheduler_state": {"best": None},
                    "num_updates": 0,
                }
            ],
            "last_optimizer_state": None,
            "extra_state": {
                "epoch": 1,
                "batch_offset": 0,
                "train_iterator": {
                    "epoch": 1,
                    "iterations_in_epoch": 0,
                },
                "chapter3_distill_only": {
                    "teacher_target_layers": self.teacher_target_layers,
                    "distill_head_mode": self.distill_head_mode,
                    "initialization_policy": self.cfg.initialization_policy,
                }
            },
        }
