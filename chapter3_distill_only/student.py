"""Student construction and initialization for Chapter 3 experiments."""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
from fairseq.models import register_model
from omegaconf import DictConfig, open_dict

from avhubert.hubert import AVHubertConfig, AVHubertModel

from .conformer import HistoricalConformerEncoder

logger = logging.getLogger(__name__)

STUDENT_MODEL_NAME = "chapter3_av_hubert"
STUDENT_ARCHITECTURES = {"transformer", "conformer"}
INITIALIZATION_POLICIES = {
    "teacher_sequence_teacher_frontend",
    "random_sequence_teacher_frontend",
    "warm_start_distilled",
}


def parse_layer_ids(value: Any, field_name: str) -> Tuple[int, ...]:
    """Parse a comma-separated string or iterable of integer layer IDs."""

    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        raw_values: Iterable[Any] = stripped.split(",")
    elif isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raise TypeError(
            f"{field_name} must be a comma-separated string or a sequence, "
            f"got {type(value).__name__}"
        )

    try:
        parsed = tuple(int(item) for item in raw_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field_name}: {value!r}") from error
    if any(layer < 0 for layer in parsed):
        raise ValueError(f"{field_name} cannot contain negative IDs: {parsed}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field_name} must be unique: {parsed}")
    return parsed


def validate_architecture(
    arch: str,
    depth: int,
    embed_dim: int,
    ffn_dim: int,
    attention_heads: int,
    conformer_kernel: int,
) -> None:
    if arch not in STUDENT_ARCHITECTURES:
        raise ValueError(
            f"student_arch must be one of {sorted(STUDENT_ARCHITECTURES)}, "
            f"got {arch!r}"
        )
    if depth <= 0:
        raise ValueError("student_depth must be positive")
    if embed_dim <= 0 or ffn_dim <= 0 or attention_heads <= 0:
        raise ValueError("Student dimensions and attention-head count must be positive")
    if embed_dim % attention_heads != 0:
        raise ValueError(
            f"student_embed_dim={embed_dim} must be divisible by "
            f"student_attention_heads={attention_heads}"
        )
    if arch == "conformer" and (
        conformer_kernel <= 0 or conformer_kernel % 2 == 0
    ):
        raise ValueError(
            "student_conformer_kernel must be positive and odd"
        )


def validate_target_mapping(
    teacher_target_layers: Tuple[int, ...],
    student_match_layers: Tuple[int, ...],
    distill_head_mode: str,
    teacher_depth: int,
    student_depth: int,
) -> None:
    if not teacher_target_layers:
        raise ValueError("teacher_target_layers cannot be empty")
    if max(teacher_target_layers) > teacher_depth:
        raise ValueError(
            f"Teacher target {max(teacher_target_layers)} exceeds teacher "
            f"depth {teacher_depth}"
        )
    if distill_head_mode == "historical_pred_heads":
        # The final representation is the only student representation read.
        return
    if distill_head_mode != "layer_to_layer":
        raise ValueError(
            "distill_head_mode must be 'historical_pred_heads' or "
            f"'layer_to_layer', got {distill_head_mode!r}"
        )
    if len(student_match_layers) != len(teacher_target_layers):
        raise ValueError(
            "Layer-to-layer mode requires one student_match_layers entry "
            "per teacher target"
        )
    if not student_match_layers:
        raise ValueError(
            "student_match_layers cannot be empty in layer-to-layer mode"
        )
    if max(student_match_layers) > student_depth:
        raise ValueError(
            f"Student match layer {max(student_match_layers)} exceeds student "
            f"depth {student_depth}"
        )


@dataclass
class Chapter3AVHubertConfig(AVHubertConfig):
    """Native, exportable student model configuration."""

    student_arch: str = field(
        default="transformer",
        metadata={"help": "student sequence block: transformer or conformer"},
    )
    student_conformer_kernel: int = field(default=31)
    student_conformer_attention_type: str = field(default="original")
    student_conformer_position_type: str = field(default="abs")


@register_model(STUDENT_MODEL_NAME, dataclass=Chapter3AVHubertConfig)
class Chapter3AVHubertModel(AVHubertModel):
    """AV-HuBERT student with a configurable Transformer/Conformer encoder."""

    def __init__(self, cfg, task_cfg, dictionaries, **kwargs) -> None:
        self._assert_pruning_disabled(cfg)
        validate_architecture(
            arch=cfg.student_arch,
            depth=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            ffn_dim=cfg.encoder_ffn_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            conformer_kernel=cfg.student_conformer_kernel,
        )
        super().__init__(cfg, task_cfg, dictionaries, **kwargs)
        self.cfg = cfg
        if cfg.student_arch == "conformer":
            self.encoder = HistoricalConformerEncoder(
                depth=cfg.encoder_layers,
                embed_dim=cfg.encoder_embed_dim,
                ffn_dim=cfg.encoder_ffn_embed_dim,
                attention_heads=cfg.encoder_attention_heads,
                kernel_size=cfg.student_conformer_kernel,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                layerdrop=cfg.encoder_layerdrop,
                attention_type=cfg.student_conformer_attention_type,
                position_type=cfg.student_conformer_position_type,
                max_positions=getattr(cfg, "max_positions", 100000),
            )
        # This registration represents the deployed student backbone, not a
        # HuBERT pretraining model.  Construct it in its native stripped form
        # so an exported checkpoint reloads with Fairseq's default strict=True.
        self.remove_pretraining_modules()
        if hasattr(self, "label_embs_concat"):
            self.register_parameter("label_embs_concat", None)

    @staticmethod
    def _assert_pruning_disabled(cfg) -> None:
        forbidden_flags = (
            "resnet_prune_conv_channels",
            "encoder_prune_attention_heads",
            "encoder_prune_attention_layer",
            "encoder_prune_ffn_intermediate",
            "encoder_prune_ffn_layer",
        )
        enabled = [name for name in forbidden_flags if bool(getattr(cfg, name, False))]
        if enabled:
            raise ValueError(
                "Chapter 3 distillation-only students cannot enable pruning: "
                + ", ".join(enabled)
            )
        if getattr(cfg, "encoder_merge_type", None) is not None:
            raise ValueError(
                "Chapter 3 distillation-only students cannot enable "
                "attention-weight merging"
            )

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.cfg, task.dictionaries)

    def extract_distillation_features(
        self,
        source,
        padding_mask=None,
        *,
        return_intermediates: bool,
    ):
        """Return the exact representations needed by the selected objective.

        Historical prediction heads take only the final encoder output, so that
        path uses the normal encoder forward and never constructs a list of all
        layers. Layer-to-layer mode explicitly requests the inclusive
        ``[input, layer1, ..., layerN]`` representation list.
        """

        source_audio, source_video = source["audio"], source["video"]
        features_audio = self.forward_features(source_audio, modality="audio")
        features_video = self.forward_features(source_video, modality="video")
        if self.training and np.random.random() < self.modality_dropout:
            if np.random.random() < self.audio_dropout:
                features_audio = 0 * features_audio
            else:
                features_video = 0 * features_video
        if self.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == "add":
            features = features_audio + features_video
        else:
            raise ValueError(
                f"Unsupported modality_fuse setting {self.modality_fuse!r}"
            )

        features = self.layer_norm(features.transpose(1, 2))
        # Historical Distil-AVHuBERT returns ``feat`` after the fused-feature
        # LayerNorm and computes ``feat.float().pow(2).mean()`` in
        # pretrain_expert.py.  Keep this before the optional projection.
        feature_penalty = features.float().pow(2).mean()
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)

        if return_intermediates:
            representations = self.encoder.get_intermediate_outputs(
                features, padding_mask=padding_mask
            )
        else:
            representations, _ = self.encoder(
                features, padding_mask=padding_mask, layer=None
            )
        return representations, feature_penalty, padding_mask


def _set_config_value(config: Any, key: str, value: Any) -> None:
    if isinstance(config, DictConfig):
        with open_dict(config):
            config[key] = value
    elif isinstance(config, MutableMapping):
        config[key] = value
    else:
        setattr(config, key, value)


def build_student_checkpoint_config(teacher_args: Any, cfg: Any) -> Any:
    """Derive a native student checkpoint config from the frozen teacher."""

    student_args = copy.deepcopy(teacher_args)
    model_cfg = student_args.model

    overrides = {
        "_name": STUDENT_MODEL_NAME,
        "student_arch": cfg.student_arch,
        "student_conformer_kernel": cfg.student_conformer_kernel,
        "student_conformer_attention_type": (
            cfg.student_conformer_attention_type
        ),
        "student_conformer_position_type": (
            cfg.student_conformer_position_type
        ),
        "encoder_layers": cfg.student_depth,
        "encoder_embed_dim": cfg.student_embed_dim,
        "encoder_ffn_embed_dim": cfg.student_ffn_dim,
        "encoder_attention_heads": cfg.student_attention_heads,
        # Historical models use standard D / H attention dimensions.
        "encoder_attention_heads_dim": (
            cfg.student_embed_dim // cfg.student_attention_heads
        ),
        "encoder_ffn_embed_dim_detailed": [],
        "encoder_attention_heads_detailed": [],
        "encoder_use_attention": [],
        "encoder_use_ffn": [],
        "layer_norm_first": cfg.student_layer_norm_first,
        "dropout_input": cfg.student_dropout_input,
        "modality_dropout": cfg.modality_dropout,
        "audio_dropout": cfg.audio_dropout,
        "feature_grad_mult": cfg.feature_grad_mult,
        "resnet_prune_conv_channels": False,
        "encoder_prune_attention_heads": False,
        "encoder_prune_attention_layer": False,
        "encoder_prune_ffn_intermediate": False,
        "encoder_prune_ffn_layer": False,
        "encoder_merge_type": None,
    }

    architecture_overrides = {
        "dropout": cfg.student_dropout,
        "attention_dropout": cfg.student_attention_dropout,
        "activation_dropout": cfg.student_activation_dropout,
        "encoder_layerdrop": cfg.student_layerdrop,
        "conv_pos": cfg.student_conv_pos,
        "conv_pos_groups": cfg.student_conv_pos_groups,
    }
    overrides.update(architecture_overrides)

    for key, value in overrides.items():
        _set_config_value(model_cfg, key, value)
    _set_config_value(student_args.task, "data", cfg.data)
    return student_args


def _copy_matching_parameters(
    source: Mapping[str, torch.Tensor],
    destination: Mapping[str, torch.Tensor],
    prefixes: Tuple[str, ...],
    report: Dict[str, Any],
    required: bool = False,
) -> Dict[str, torch.Tensor]:
    updated = dict(destination)
    for name, destination_value in destination.items():
        if not name.startswith(prefixes):
            continue
        source_value = source.get(name)
        if source_value is None:
            report["skipped"][name] = "missing_from_teacher"
            continue
        if source_value.shape != destination_value.shape:
            report["skipped"][name] = (
                f"shape:{tuple(source_value.shape)}!="
                f"{tuple(destination_value.shape)}"
            )
            continue
        updated[name] = source_value.detach().clone()
        report["copied"].append(name)

    if required:
        relevant = [name for name in destination if name.startswith(prefixes)]
        not_copied = [name for name in relevant if name not in report["copied"]]
        if not_copied:
            preview = ", ".join(not_copied[:8])
            raise ValueError(
                "Teacher-derived initialization is not shape-compatible; "
                f"failed keys include: {preview}"
            )
    return updated


def initialize_student(
    student: Chapter3AVHubertModel,
    teacher: AVHubertModel,
    policy: str,
) -> Dict[str, Any]:
    """Apply one of the named, auditable initialization policies."""

    if policy not in INITIALIZATION_POLICIES:
        raise ValueError(
            f"Unknown initialization_policy {policy!r}; expected one of "
            f"{sorted(INITIALIZATION_POLICIES)}"
        )

    report: Dict[str, Any] = {
        "policy": policy,
        "copied": [],
        "skipped": {},
        "notes": [],
    }
    if policy == "warm_start_distilled":
        report["notes"].append(
            "Initialization is delegated to Fairseq's "
            "checkpoint.finetune_from_model loader."
        )
        return report

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    frontend_prefixes = (
        "feature_extractor_audio.",
        "feature_extractor_video.",
        "post_extract_proj.",
        "layer_norm.",
    )
    updated = _copy_matching_parameters(
        teacher_state,
        student_state,
        frontend_prefixes,
        report,
        required=False,
    )

    if policy == "teacher_sequence_teacher_frontend":
        if getattr(student.cfg, "student_arch", "transformer") != "transformer":
            raise ValueError(
                "teacher_sequence_teacher_frontend requires a Transformer student"
            )
        if student.cfg.encoder_layers != 2:
            raise ValueError(
                "The controlled teacher-derived sequence initialization is "
                "defined for the two-block Transformer student"
            )
        sequence_prefixes = ("encoder.pos_conv.", "encoder.layers.")
        updated = _copy_matching_parameters(
            teacher_state,
            updated,
            sequence_prefixes,
            report,
            required=True,
        )
    else:
        report["notes"].append(
            "Sequence encoder remains randomly initialized; only "
            "shape-compatible frontend/fusion parameters were copied."
        )

    student.load_state_dict(updated, strict=True)
    report["copied"].sort()
    report["copied_count"] = len(report["copied"])
    report["skipped_count"] = len(report["skipped"])
    return report


def strip_pretraining_targets(model: AVHubertModel) -> None:
    """Remove teacher-pretraining targets while retaining fine-tuning masks."""

    model.remove_pretraining_modules()
    if hasattr(model, "label_embs_concat"):
        model.register_parameter("label_embs_concat", None)
