#!/usr/bin/env python3
"""Validate and resolve the immutable Chapter 3 Stage-K experiment matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    output_id: str
    stage1_config: str
    stage2_config: str
    matching: str
    targets: str
    cosine: str
    stage1_l1: float
    stage2_l1: float


MATRIX = {
    item.experiment_id: item
    for item in (
        Experiment("k-ref", "k_ref", "kref_l2l_t04812_raw_l1_0p1", "kref_l2l_t04812_raw_l1_1p0", "layer2layer", "0,4,8,12", "raw", 0.1, 1.0),
        Experiment("k-t", "k_t", "kt_l2l_t812_raw_l1_0p1", "kt_l2l_t812_raw_l1_1p0", "layer2layer", "8,12", "raw", 0.1, 1.0),
        Experiment("k0", "k0", "k0_pred_t812_raw_l1_0p1", "k0_pred_t812_raw_l1_1p0", "historical_pred_heads", "8,12", "raw", 0.1, 1.0),
        Experiment("k1", "k1", "k1_pred_t812_log_sig_l1_0p1", "k1_pred_t812_log_sig_l1_1p0", "historical_pred_heads", "8,12", "log_sig", 0.1, 1.0),
        Experiment("k2", "k2", "k2_pred_t812_raw_l1_0p1", "k2_pred_t812_raw_l1_0p1", "historical_pred_heads", "8,12", "raw", 0.1, 0.1),
        Experiment("k3", "k3", "k3_pred_t812_log_sig_l1_0p1", "k3_pred_t812_log_sig_l1_0p1", "historical_pred_heads", "8,12", "log_sig", 0.1, 0.1),
    )
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_base_protocol(stage1_root: Path, stage2_root: Path) -> None:
    first = _load(stage1_root / "base_joint_dp_50k.yaml")
    second = _load(stage2_root / "base_post_distill_25k.yaml")
    fixed = (
        (first, "stage1", 50000, [4], True),
        (second, "stage2", 25000, [4], False),
    )
    for cfg, label, updates, update_freq, use_reg in fixed:
        _expect(cfg["common"]["seed"], 1337, f"{label} seed")
        _expect(cfg["task"]["noise_prob"], 0.25, f"{label} noise probability")
        _expect(cfg["task"]["noise_snr"], 0, f"{label} noise SNR")
        _expect(cfg["optimization"]["max_update"], updates, f"{label} updates")
        _expect(cfg["optimization"]["clip_norm"], 10.0, f"{label} clip norm")
        _expect(cfg["optimization"]["update_freq"], update_freq, f"{label} update frequency")
        _expect(cfg["criterion"]["use_reg"], use_reg, f"{label} regularization")
    _expect(first["criterion"]["sparsity_warmup_updates"], 10000, "stage1 sparsity warm-up")
    _expect(second["optimizer"]["_name"], "adam", "stage2 optimizer")
    _expect(second["lr_scheduler"]["total_num_update"], 25000, "stage2 schedule budget")


def resolve(
    experiment_id: str, stage1_root: Path, stage2_root: Path
) -> tuple[Experiment, Path, Path]:
    try:
        experiment = MATRIX[experiment_id]
    except KeyError as error:
        raise ValueError(
            f"unknown Stage-K experiment {experiment_id!r}; "
            f"choose from {', '.join(MATRIX)}"
        ) from error
    validate_base_protocol(stage1_root, stage2_root)
    stage1 = stage1_root / f"{experiment.stage1_config}.yaml"
    stage2 = stage2_root / f"{experiment.stage2_config}.yaml"
    if not stage1.is_file() or not stage2.is_file():
        raise ValueError(f"missing paired Stage-K configs: {stage1}, {stage2}")
    first_entry = _load(stage1)
    second_entry = _load(stage2)
    _expect(
        first_entry.get("defaults"),
        ["base_joint_dp_50k", f"stage_k_matrix/{experiment.stage1_config}"],
        "Stage-1 defaults",
    )
    _expect(
        second_entry.get("defaults"),
        ["base_post_distill_25k", f"stage_k_matrix/{experiment.stage2_config}"],
        "Stage-2 defaults",
    )
    first = _load(stage1_root / "stage_k_matrix" / stage1.name)
    second = _load(stage2_root / "stage_k_matrix" / stage2.name)
    for cfg, label, weight in (
        (first, "Stage 1", experiment.stage1_l1),
        (second, "Stage 2", experiment.stage2_l1),
    ):
        _expect(cfg["model"]["distill_mode"], experiment.matching, f"{label} matching")
        expected_layers = (
            "0.4,8,12" if experiment.targets == "0,4,8,12" else "8,12"
        )
        _expect(cfg["model"]["distill_layers"], expected_layers, f"{label} targets")
        _expect(cfg["criterion"]["cos_type"], experiment.cosine, f"{label} cosine")
        _expect(float(cfg["criterion"]["l1_weight"]), weight, f"{label} L1")
    _expect(first["criterion"]["target_sparsity"], 0.70, "Stage-1 target sparsity")
    _expect(first["model"]["pruning_units"], "conv,head,interm", "Stage-1 pruning units")
    return experiment, stage1.resolve(), stage2.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=tuple(MATRIX))
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        experiment, stage1, stage2 = resolve(
            args.experiment, args.stage1_root.resolve(), args.stage2_root.resolve()
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    result = asdict(experiment)
    result.update(
        {
            "stage1_path": str(stage1),
            "stage2_path": str(stage2),
            "pruning_units": "conv,head,interm",
            "target_sparsity": 0.70,
        }
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "\t".join(
                str(result[key])
                for key in (
                    "output_id",
                    "stage1_config",
                    "stage2_config",
                    "matching",
                    "targets",
                    "cosine",
                    "stage1_l1",
                    "stage2_l1",
                    "stage1_path",
                    "stage2_path",
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
