import logging
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass, ChoiceEnum

logger = logging.getLogger(__name__)
COSINE_TYPE_CHOICES = ChoiceEnum(["raw", "log_sig"])
MERGE_TYPE_CHOICES = ChoiceEnum(["QKV", "QK", "QV", "KV"])

@dataclass
class AVHubertDistillCriterionConfig(FairseqDataclass):
    l2_weight: float = field(
        default=0.0,
        metadata={"help": "weight of MSE loss"},
    )
    l1_weight: float = field(
        default=1.0,
        metadata={"help": "weight of L1 loss"},
    )
    cos_weight: float = field(
        default=1.0,
        metadata={"help": "weight of cosine similarity loss"},
    )
    cos_type: COSINE_TYPE_CHOICES = field(
        default="raw",
        metadata={"help": "type of the cosine similarity loss"},
    )
    use_reg: bool = field(
        default=False,
        metadata={"help": "use l0 regularization"},
    )
    sparsity_warmup_updates: int = field(
        default=0,
        metadata={"help": "target sparsity warm-up steps"},
    )
    target_sparsity: float = field(
        default=0.0,
        metadata={"help": "model target sparsity"}
    )
    use_component_sparsities: bool = field(
        default=False,
        metadata={"help": "use separate sparsities for feature extractor and encoder"},
    )
    feature_target_sparsity: float = field(
        default=0.0,
        metadata={"help": "feature extractor target sparsity. 'use_component_sparsity' must  be True"}
    )
    encoder_target_sparsity: float = field(
        default=0.0,
        metadata={"help": "encoder target sparsity. 'use_component_sparsity' must  be True"}
    )
    use_merge: bool = field(
        default=False,
        metadata={"help": "use merging"},
    )
    merge_type: Optional[MERGE_TYPE_CHOICES] = field(
        default=None,
        metadata={"help": "specify the weight matrices to be merged"},
    )
    merge_step: int = field(
        default=2500,
        metadata={"help": "steps between merge"},
    )
    beta_warmup_steps: int = field(
        default=0,
        metadata={"help": "beta warm-up steps"}
    )


@register_criterion("av_hubert_distill", dataclass=AVHubertDistillCriterionConfig)
class AVHubertDistillCriterion(FairseqCriterion):
    def __init__(self, task, l2_weight, l1_weight, cos_weight, cos_type, use_reg, sparsity_warmup_updates, target_sparsity, use_component_sparsities , feature_target_sparsity, encoder_target_sparsity, use_merge, merge_type, merge_step, beta_warmup_steps):
        super().__init__(task)
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight
        assert cos_type in ["raw", "log_sig"], cos_type
        self.cos_type = cos_type
        self.use_reg = use_reg
        self.use_merge = use_merge

        # lambdas for Lagrangian
        if self.use_reg:
            self.use_component_sparsities = use_component_sparsities
            self.lambda1 = nn.Parameter(torch.tensor(0.0))
            self.lambda2 = nn.Parameter(torch.tensor(0.0))
            self.lambda1.param_group = "lambda"
            self.lambda2.param_group = "lambda"

            if self.use_component_sparsities:
                self.lambda3 = nn.Parameter(torch.tensor(0.0))
                self.lambda4 = nn.Parameter(torch.tensor(0.0))
                self.lambda3.param_group = "lambda"
                self.lambda4.param_group = "lambda"

                self.feature_target_sparsity = feature_target_sparsity
                self.encoder_target_sparsity = encoder_target_sparsity

            self.sparsity_warmup_updates = sparsity_warmup_updates
            self.target_sparsity = target_sparsity

        if self.use_merge:
            self.merge_type = merge_type
            self.merge_step = merge_step
            self.beta_warmup_steps = beta_warmup_steps

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce

        if self.use_merge:
            with torch.no_grad():
                if model.num_updates % self.merge_step == 0:
                    beta = (model.num_updates + self.merge_step) / (self.beta_warmup_steps + self.merge_step) if model.num_updates < self.beta_warmup_steps else 1
                    model.student.merge(beta, self.merge_type)

        net_output = model(**sample["net_input"])
        teacher = net_output["teacher_hiddens"]
        student = net_output["student_hiddens"]

        loss_mse = 0
        loss_l1 = 0
        loss_cos = 0
        if self.l2_weight != 0:
            loss_mse = F.mse_loss(student, teacher)
        if self.l1_weight != 0:
            loss_l1 = F.l1_loss(student, teacher)
        if self.cos_weight != 0:    # maximize cosine similarity
            if self.cos_type == "raw":
                loss_cos = -F.cosine_similarity(student, teacher, dim=-1).mean()
            elif self.cos_type == "log_sig":
                loss_cos = -F.cosine_similarity(student, teacher, dim=-1).sigmoid().log().mean()
            else:
                raise ValueError

        loss = self.l2_weight * loss_mse + self.l1_weight * loss_l1 + self.cos_weight * loss_cos
        
        if self.use_reg:
            if self.use_component_sparsities:
                if model.num_updates >= self.sparsity_warmup_updates:
                    cur_feature_target_sparsity = self.feature_target_sparsity
                    cur_encoder_target_sparsity = self.encoder_target_sparsity
                else:
                    cur_feature_target_sparsity = self.feature_target_sparsity * (model.num_updates / self.sparsity_warmup_updates)
                    cur_encoder_target_sparsity = self.encoder_target_sparsity * (model.num_updates / self.sparsity_warmup_updates)
                cur_feature_expected_sparsity = 1. - model.get_student_num_params("feature") / model.get_original_num_params("feature")
                cur_encoder_expected_sparsity = 1. - model.get_student_num_params("encoder") / model.get_original_num_params("encoder")
                loss_reg = self.lambda1 * (cur_feature_expected_sparsity - cur_feature_target_sparsity) + self.lambda2 * (cur_feature_expected_sparsity - cur_feature_target_sparsity)**2 \
                            + self.lambda3 * (cur_encoder_expected_sparsity - cur_encoder_target_sparsity) + self.lambda4 * (cur_encoder_expected_sparsity - cur_encoder_target_sparsity)**2
            else:
                if model.num_updates >= self.sparsity_warmup_updates:
                    cur_target_sparsity = self.target_sparsity
                else:
                    cur_target_sparsity = self.target_sparsity * (model.num_updates / self.sparsity_warmup_updates)
                cur_expected_sparsity = 1. - model.get_student_num_params() / model.get_original_num_params()
                loss_reg = self.lambda1 * (cur_expected_sparsity - cur_target_sparsity) \
                    + self.lambda2 * (cur_expected_sparsity - cur_target_sparsity)**2
        else:
            loss_reg = 0

        loss = loss + loss_reg

        sample_size = sample["id"].numel()

        logging_output = {
            "loss": loss,
            "loss_l1": loss_l1,
            "loss_l2": loss_mse,
            "loss_cos": loss_cos,
            "nsentences": sample_size,
            "sample_size": sample_size,
            "denominator": 1,
        }
        
        if self.use_reg:
            if self.use_component_sparsities:
                logging_output.update({
                    "loss_reg": loss_reg,
                    "feature_sparsity_expected": cur_feature_expected_sparsity,
                    "feature_sparsity_target": cur_feature_target_sparsity,
                    "encoder_sparsity_expected": cur_encoder_expected_sparsity,
                    "encoder_sparsity_target": cur_encoder_target_sparsity,
                    "lambda1": self.lambda1,
                    "lambda2": self.lambda2, 
                    "lambda3": self.lambda3,
                    "lambda4": self.lambda4,
                })
            else:
                logging_output.update({
                    "loss_reg": loss_reg,
                    "sparsity_expected": cur_expected_sparsity,
                    "sparsity_target": cur_target_sparsity,
                    "lambda1": self.lambda1,
                    "lambda2": self.lambda2, 
                })

        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training"""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_l1_sum = utils.item(sum(log.get("loss_l1", 0) for log in logging_outputs))
        loss_l2_sum = utils.item(sum(log.get("loss_l2", 0) for log in logging_outputs))
        loss_cos_sum = utils.item(sum(log.get("loss_cos", 0) for log in logging_outputs))
        loss_reg_sum = utils.item(sum(log.get("loss_reg", 0) for log in logging_outputs))
        lambda1_sum = utils.item(sum(log.get("lambda1", 0) for log in logging_outputs))
        lambda2_sum = utils.item(sum(log.get("lambda2", 0) for log in logging_outputs))
        lambda3_sum = utils.item(sum(log.get("lambda3", 0) for log in logging_outputs))
        lambda4_sum = utils.item(sum(log.get("lambda4", 0) for log in logging_outputs))
        sparsity_expected_sum = utils.item(sum(log.get("sparsity_expected", 0) for log in logging_outputs))
        sparsity_target_sum = utils.item(sum(log.get("sparsity_target", 0) for log in logging_outputs))
        feature_sparsity_expected_sum = utils.item(sum(log.get("feature_sparsity_expected", 0) for log in logging_outputs))
        feature_sparsity_target_sum = utils.item(sum(log.get("feature_sparsity_target", 0) for log in logging_outputs))
        encoder_sparsity_expected_sum = utils.item(sum(log.get("encoder_sparsity_expected", 0) for log in logging_outputs))
        encoder_sparsity_target_sum = utils.item(sum(log.get("encoder_sparsity_target", 0) for log in logging_outputs))
        denominator = utils.item(sum(log.get("denominator", 0) for log in logging_outputs))


        metrics.log_scalar("loss", loss_sum / denominator, denominator, round=3)
        metrics.log_scalar("loss_l1", loss_l1_sum / denominator, denominator, round=3)
        metrics.log_scalar("loss_l2", loss_l2_sum / denominator, denominator, round=3)
        metrics.log_scalar("loss_cos", loss_cos_sum / denominator, denominator, round=3)
        metrics.log_scalar("loss_reg", loss_reg_sum / denominator, denominator, round=3)
        metrics.log_scalar("lambda1", lambda1_sum / denominator, denominator, round=3)
        metrics.log_scalar("lambda2", lambda2_sum / denominator, denominator, round=3)
        metrics.log_scalar("lambda3", lambda3_sum / denominator, denominator, round=3)
        metrics.log_scalar("lambda4", lambda4_sum / denominator, denominator, round=3)
        metrics.log_scalar("sparsity_expected", sparsity_expected_sum / denominator, denominator, round=3)
        metrics.log_scalar("sparsity_target", sparsity_target_sum / denominator, denominator, round=3)
        metrics.log_scalar("feature_sparsity_expected", feature_sparsity_expected_sum / denominator, denominator, round=3)
        metrics.log_scalar("feature_sparsity_target", feature_sparsity_target_sum / denominator, denominator, round=3)
        metrics.log_scalar("encoder_sparsity_expected", encoder_sparsity_expected_sum / denominator, denominator, round=3)
        metrics.log_scalar("encoder_sparsity_target", encoder_sparsity_target_sum / denominator, denominator, round=3)
        metrics.log_scalar("denominator", denominator, denominator, round=1)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
    