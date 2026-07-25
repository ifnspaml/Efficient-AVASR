"""Ordinary distillation criterion without pruning or merge objectives."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics

DISTILL_CRITERION_NAME = "av_hubert_distill_only"


@dataclass
class AVHubertDistillOnlyCriterionConfig(FairseqDataclass):
    distill_loss_type: str = field(
        default="historical",
        metadata={"help": "historical or weighted component formulation"},
    )
    l1_weight: float = field(default=1.0)
    l2_weight: float = field(default=0.0)
    cosine_weight: float = field(default=1.0)
    cosine_type: str = field(default="log_sig")
    feature_penalty_weight: float = field(default=0.0)


@register_criterion(
    DISTILL_CRITERION_NAME, dataclass=AVHubertDistillOnlyCriterionConfig
)
class AVHubertDistillOnlyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        distill_loss_type,
        l1_weight,
        l2_weight,
        cosine_weight,
        cosine_type,
        feature_penalty_weight,
    ) -> None:
        super().__init__(task)
        if distill_loss_type not in {"historical", "weighted"}:
            raise ValueError(
                "distill_loss_type must be 'historical' or 'weighted'"
            )
        if cosine_type not in {"raw", "log_sig"}:
            raise ValueError("cosine_type must be 'raw' or 'log_sig'")
        for name, value in {
            "l1_weight": l1_weight,
            "l2_weight": l2_weight,
            "cosine_weight": cosine_weight,
            "feature_penalty_weight": feature_penalty_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.distill_loss_type = distill_loss_type
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.cosine_weight = float(cosine_weight)
        self.cosine_type = cosine_type
        self.feature_penalty_weight = float(feature_penalty_weight)

    def forward(self, model, sample, reduce=True):
        if not reduce:
            raise ValueError(
                "The distillation-only criterion requires reduced losses"
            )
        net_output = model(**sample["net_input"])
        teacher = net_output["teacher_hiddens"]
        student = net_output["student_hiddens"]
        if teacher.shape != student.shape:
            raise ValueError(
                f"Distillation shape mismatch: {teacher.shape} != {student.shape}"
            )

        zero = student.new_zeros(())
        loss_l1 = (
            F.l1_loss(student, teacher, reduction="mean")
            if self.l1_weight
            else zero
        )
        loss_l2 = (
            F.mse_loss(student, teacher, reduction="mean")
            if self.l2_weight
            else zero
        )
        if self.cosine_weight:
            cosine = F.cosine_similarity(student, teacher, dim=-1)
            if self.cosine_type == "raw":
                loss_cosine = -cosine.mean()
            else:
                # Historical implementation:
                # -log(sigmoid(cosine_similarity)).
                loss_cosine = -F.logsigmoid(cosine).mean()
        else:
            loss_cosine = zero
        feature_penalty = net_output.get("student_features_pen", zero)

        loss = (
            self.l1_weight * loss_l1
            + self.l2_weight * loss_l2
            + self.cosine_weight * loss_cosine
            + self.feature_penalty_weight * feature_penalty
        )
        # Every component above is already a global mean over B/N/T/D.
        # Returning a sentence-count sample size would divide the gradient a
        # second time in Fairseq.  The historical distiller and the current
        # joint-DP criterion therefore use one mean-loss unit per worker.
        sample_size = 1
        nsentences = sample["id"].numel()
        target_layers = net_output["teacher_target_layers"]
        logging_output = {
            "loss": loss.detach(),
            "loss_l1": loss_l1.detach(),
            "loss_l2": loss_l2.detach(),
            "loss_cosine": loss_cosine.detach(),
            "feature_penalty": feature_penalty.detach(),
            "sample_size": sample_size,
            "denominator": 1,
            "nsentences": nsentences,
        }
        with torch.no_grad():
            absolute = (student - teacher).abs()
            cosine_by_layer = -F.logsigmoid(
                F.cosine_similarity(student, teacher, dim=-1)
            )
            for index, layer in enumerate(target_layers):
                logging_output[f"l1_target_{layer}"] = (
                    absolute[:, index].mean().detach()
                )
                logging_output[f"log_sig_cos_target_{layer}"] = (
                    cosine_by_layer[:, index].mean().detach()
                )
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        denominator = sum(
            log.get("denominator", 1) for log in logging_outputs
        )
        if denominator == 0:
            return
        metric_names = {
            key
            for logging_output in logging_outputs
            for key in logging_output
            if key.startswith(("loss", "feature_penalty", "l1_target_", "log_sig_"))
        }
        for name in sorted(metric_names):
            value = sum(
                utils.item(log.get(name, 0)) for log in logging_outputs
            )
            metrics.log_scalar(
                name, value / denominator, denominator, round=6
            )
        metrics.log_scalar(
            "nsentences",
            sum(log.get("nsentences", 0) for log in logging_outputs),
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
