from dataclasses import dataclass, field
from typing import Any, Optional
from argparse import Namespace

import torch
import torch.nn as nn

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES

from omegaconf import II, MISSING, open_dict

try:
    from .historical_prediction_heads import HistoricalPredictionHeads
except ImportError:  # Fairseq user-dir modules are also imported top-level.
    from historical_prediction_heads import HistoricalPredictionHeads


DISTILL_MODE_CHOICES = ChoiceEnum(
    ["layer2layer", "predlayer", "historical_pred_heads"]
)

@dataclass
class AVHubertDistillConfig(FairseqDataclass):
    modality_dropout: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    teacher_path: str = field(
        default=MISSING, metadata={"help": "path to teacher model"}
    )
    student_path: str = field(
        default=MISSING, metadata={"help": "path to student model"}
    )
    distill_layers: str = field(
        default="0.4,8,12", metadata={"help": "distill layer indices (use period to separate groups and comma to separate layers within a group)"}
    )
    distill_mode: DISTILL_MODE_CHOICES = field(
        default="layer2layer",
        metadata={
            "help": "distillation method: 'layer2layer', legacy 'predlayer', "
            "or Stage-L-equivalent 'historical_pred_heads'"
        },
    )
    pruning_units: str = field(
        default="",
        metadata={"help": "pruning units as a comma-separated list"}
    )
    # this holds the loaded hubert args
    teacher_args: Any = None
    student_args: Any = None


@register_model("av_hubert_distill", dataclass=AVHubertDistillConfig)
class AVHubertDistill(BaseFairseqModel):
    def __init__(
        self,
        teacher,
        student,
        distill_layers,
        distill_linear_projs,
        cfg,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.cfg = cfg
        
        self.distill_mode = cfg.distill_mode
        self.distill_layers = distill_layers
        self.distill_linear_projs = distill_linear_projs

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
    
        arg_overrides = {
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "feature_grad_mult": cfg.feature_grad_mult,
            "modality_dropout": cfg.modality_dropout,
        }
        # build teacher
        if cfg.teacher_args is None:
            teacher_state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.teacher_path, arg_overrides
            )
            teacher_args = teacher_state.get("cfg", None)
            if teacher_args is None:
                teacher_args = convert_namespace_to_omegaconf(teacher_state["args"])
            cfg.teacher_args = teacher_args
        else:
            teacher_state = None
            teacher_args = cfg.teacher_args
            if isinstance(teacher_args, Namespace):
                cfg.teacher_args = teacher_args = convert_namespace_to_omegaconf(teacher_args)

        assert cfg.normalize == teacher_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        teacher_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(teacher_args.task)
        if teacher_state is not None:
            task_pretrain.load_state_dict(teacher_state['task_state'])

        teacher = task_pretrain.build_model(teacher_args.model)
        if teacher_state is not None:
            # set strict=False because we omit some modules
            del teacher_state['model']['mask_emb']
            teacher.load_state_dict(teacher_state["model"], strict=False)
        teacher.remove_pretraining_modules()
        for p in teacher.parameters():
            p.requires_grad = False
        
        # build student
        if cfg.student_args is None:
            student_state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.student_path, arg_overrides
            )
            student_args = student_state.get("cfg", None)
            if student_args is None:
                student_args = convert_namespace_to_omegaconf(student_state["args"])
            pruning_units = cfg.pruning_units.split(",")
            with open_dict(student_args):
                student_args.model["resnet_prune_conv_channels"] = "conv" in pruning_units
                student_args.model["encoder_prune_attention_heads"] = "head" in pruning_units
                student_args.model["encoder_prune_attention_layer"] = "attlayer" in pruning_units
                student_args.model["encoder_prune_ffn_intermediate"] = "interm" in pruning_units
                student_args.model["encoder_prune_ffn_layer"] = "ffnlayer" in pruning_units
            cfg.student_args = student_args
        else:
            student_state = None
            student_args = cfg.student_args
            if isinstance(student_args, Namespace):
                cfg.student_args = student_args = convert_namespace_to_omegaconf(student_args)

        assert cfg.normalize == student_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        student_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(student_args.task)
        student = task_pretrain.build_model(student_args.model)
        if student_state is not None:
            task_pretrain.load_state_dict(student_state['task_state'])
        if student_state is not None:
            # set strict=False because we omit some modules
            del student_state['model']['mask_emb']
            student.load_state_dict(student_state["model"], strict=False)
        student.remove_pretraining_modules()

        # Create linear layers which transform student hiddens to teacher hiddens
        distill_layer_groups = [[int(l) for l in g.split(",")] for g in cfg.distill_layers.split(".")]
        distill_layers = []
        for g in distill_layer_groups:
            distill_layers.extend(g)
        student_embed_dim = student.encoder_embed_dim
        teacher_embed_dim = teacher.encoder_embed_dim

        if cfg.distill_mode == "layer2layer":
            distill_linear_projs = nn.ModuleList()
            for g in distill_layer_groups:      # layers in the same group share a linear layer
                tmp_linear = nn.Linear(student_embed_dim, teacher_embed_dim)
                cls._init_layer_transform(tmp_linear)
                for _ in range(len(g)):
                    distill_linear_projs.append(tmp_linear)
        elif cfg.distill_mode == "predlayer":      # same as DistilHuBERT
            # use independent linear layers, cannot be shared
            distill_linear_projs = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(student_embed_dim, teacher_embed_dim),
                    nn.GELU(),
                ) for _ in range(len(distill_layers))
            )
        elif cfg.distill_mode == "historical_pred_heads":
            # Stage-L semantics: expand the final student representation into
            # independent target chunks, apply GELU, then project each chunk.
            distill_linear_projs = HistoricalPredictionHeads(
                student_dim=student_embed_dim,
                teacher_dim=teacher_embed_dim,
                target_layers=distill_layers,
            )
        else:
            raise ValueError(f"Invalid distill mode: {cfg.distill_mode}")
        
        if student_state["extra_state"].get("distill_linear_projs", None) is not None:
            distill_linear_projs.load_state_dict(student_state["extra_state"]["distill_linear_projs"])
        
        for n, p in student.named_parameters():
            if "log_alpha" not in n:
                p.param_group = "main"
        for p in distill_linear_projs.parameters():
            p.param_group = "main"
        for n, p in student.named_parameters():
            if "log_alpha" in n:
                p.param_group = "log_alpha"

        return AVHubertDistill(teacher, student, distill_layers, distill_linear_projs, cfg)

    def forward(self, **kwargs):
        self.teacher.eval()
        with torch.no_grad():
            teacher_hiddens = self.teacher.extract_intermediate_features(source=kwargs["source"], padding_mask=kwargs["padding_mask"])
            teacher_hiddens = torch.stack(
                [teacher_hiddens[idx] for idx in self.distill_layers], dim=1
            )   # (batch, layer, time, feature)
        
        if self.distill_mode == "historical_pred_heads":
            # Use the normal encoder forward so this is the true final student
            # representation, including the final encoder LayerNorm when used.
            student_final, _ = self.student.extract_features(
                source=kwargs["source"],
                padding_mask=kwargs["padding_mask"],
                mask=False,
            )
            student_hiddens = self.distill_linear_projs(student_final)
        else:
            student_hiddens = self.student.extract_intermediate_features(source=kwargs["source"], padding_mask=kwargs["padding_mask"])
            new_student_hiddens = []
            for idx, proj in zip(self.distill_layers, self.distill_linear_projs):
                if self.distill_mode == "layer2layer":
                    new_student_hiddens.append(proj(student_hiddens[idx]))
                elif self.distill_mode == "predlayer":
                    new_student_hiddens.append(proj(student_hiddens[-1]))
                else:
                    raise ValueError(f"Invalid distill mode: {self.distill_mode}")
            student_hiddens = torch.stack(new_student_hiddens, dim=1)   # (batch, layer, time, feature)

        return {"teacher_hiddens": teacher_hiddens, "student_hiddens": student_hiddens}

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def get_student_num_params(self, component: Optional[str] = None):
        return self.student.get_num_params(component)

    def get_original_num_params(self, component: Optional[str] = None):
        return self.teacher.get_num_params(component)

    @classmethod
    def _init_layer_transform(cls, module: nn.Linear) -> None:
        module.weight.data.copy_(torch.eye(len(module.weight)))
        module.bias.data.fill_(0)
