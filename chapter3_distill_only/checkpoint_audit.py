"""Checkpoint-chain and training-log diagnostics for Chapter 3 experiments.

The helpers in this module deliberately operate on state dictionaries instead
of constructing models.  This makes it possible to prove that a deployed
student was copied exactly from the distillation checkpoint and that the
frozen fine-tuning backbone was loaded from that deployed student.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


ENCODER_STUDENT_PREFIX = "student."
FINETUNE_BACKBONE_PREFIX = "encoder.w2v_model."
FINETUNE_PROJECTION_PREFIX = "encoder.proj."
_NON_PARAMETER_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)
_LOG_JSON = re.compile(r" - (\{.*\})\s*$")


def is_mutable_buffer(name: str) -> bool:
    """Return whether *name* is a BatchNorm-style mutable state buffer."""

    return name.endswith(_NON_PARAMETER_SUFFIXES)


def compare_mapped_tensors(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    source_prefix: str = "",
    target_prefix: str = "",
    names: Optional[Iterable[str]] = None,
    excluded_names: Iterable[str] = (),
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """Compare tensors after applying independent source/target prefixes.

    ``names`` are unprefixed logical names.  The result separates mutable
    buffers from parameters because frozen models can still update BatchNorm
    running statistics while executing under ``torch.no_grad``.
    """

    excluded = set(excluded_names)
    if names is None:
        logical_names = sorted(
            key[len(source_prefix) :]
            for key in source
            if key.startswith(source_prefix)
        )
    else:
        logical_names = sorted(set(names))

    missing_source = []
    missing_target = []
    shape_mismatches = []
    parameter_mismatches = []
    parameter_not_close = []
    buffer_mismatches = []
    equal_parameters = 0
    equal_buffers = 0
    compared = 0
    maximum_parameter_absolute_difference = 0.0

    for name in logical_names:
        if name in excluded:
            continue
        source_name = f"{source_prefix}{name}"
        target_name = f"{target_prefix}{name}"
        source_tensor = source.get(source_name)
        target_tensor = target.get(target_name)
        if source_tensor is None:
            missing_source.append(source_name)
            continue
        if target_tensor is None:
            missing_target.append(target_name)
            continue
        compared += 1
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                {
                    "name": name,
                    "source": list(source_tensor.shape),
                    "target": list(target_tensor.shape),
                }
            )
            continue
        equal = torch.equal(source_tensor, target_tensor)
        if is_mutable_buffer(name):
            if equal:
                equal_buffers += 1
            else:
                buffer_mismatches.append(name)
        elif equal:
            equal_parameters += 1
        else:
            parameter_mismatches.append(name)
            source_float = source_tensor.detach().float()
            target_float = target_tensor.detach().float()
            if source_float.numel():
                maximum_parameter_absolute_difference = max(
                    maximum_parameter_absolute_difference,
                    float((source_float - target_float).abs().max().item()),
                )
            if not torch.allclose(
                source_float, target_float, atol=atol, rtol=rtol
            ):
                parameter_not_close.append(name)

    return {
        "logical_tensor_count": len(logical_names) - len(excluded),
        "compared_tensor_count": compared,
        "equal_parameter_tensor_count": equal_parameters,
        "equal_mutable_buffer_count": equal_buffers,
        "missing_source": missing_source,
        "missing_target": missing_target,
        "shape_mismatches": shape_mismatches,
        "parameter_mismatches": parameter_mismatches,
        "parameter_not_close": parameter_not_close,
        "mutable_buffer_mismatches": buffer_mismatches,
        "comparison_tolerance": {"absolute": atol, "relative": rtol},
        "maximum_parameter_absolute_difference": (
            maximum_parameter_absolute_difference
        ),
        "parameters_exact": not (
            missing_source
            or missing_target
            or shape_mismatches
            or parameter_mismatches
        ),
        "parameters_close": not (
            missing_source
            or missing_target
            or shape_mismatches
            or parameter_not_close
        ),
        "all_tensors_exact": not (
            missing_source
            or missing_target
            or shape_mismatches
            or parameter_mismatches
            or buffer_mismatches
        ),
    }


def checkpoint_num_updates(checkpoint: Mapping[str, Any]) -> Optional[int]:
    history = checkpoint.get("optimizer_history") or []
    if history:
        value = history[-1].get("num_updates")
        if value is not None:
            return int(value)
    return None


def nested_get(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def parse_fairseq_json_log(path: Path, channel: str) -> list[Dict[str, Any]]:
    """Extract Fairseq JSON records for a logger channel."""

    records: list[Dict[str, Any]] = []
    marker = f"][{channel}]["
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            match = _LOG_JSON.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def summarize_validation_log(path: Path) -> Dict[str, Any]:
    records = parse_fairseq_json_log(path, "valid")
    points = []
    for record in records:
        update = record.get("valid_num_updates")
        accuracy = record.get("valid_accuracy")
        loss = record.get("valid_loss")
        if update is None:
            continue
        points.append(
            {
                "update": int(float(update)),
                "accuracy": float(accuracy) if accuracy is not None else None,
                "loss": float(loss) if loss is not None else None,
            }
        )
    valid_points = [point for point in points if point["accuracy"] is not None]
    best = max(valid_points, key=lambda point: point["accuracy"]) if valid_points else None
    return {
        "points": points,
        "best": best,
        "final": points[-1] if points else None,
    }


def summarize_training_tail(path: Path) -> Dict[str, Any]:
    records = parse_fairseq_json_log(path, "train_inner")
    if not records:
        return {}
    last = records[-1]
    return {
        "update": int(float(last["num_updates"])),
        "accuracy": float(last["accuracy"]) if "accuracy" in last else None,
        "gradient_norm": float(last["gnorm"]) if "gnorm" in last else None,
        "loss_scale": float(last["loss_scale"]) if "loss_scale" in last else None,
        "learning_rate": float(last["lr"]) if "lr" in last else None,
    }


def projection_tensors(state: Mapping[str, torch.Tensor]) -> Dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in state.items()
        if key.startswith(FINETUNE_PROJECTION_PREFIX)
    }
