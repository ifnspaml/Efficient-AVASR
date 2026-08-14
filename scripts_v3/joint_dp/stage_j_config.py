#!/usr/bin/env python3
"""Resolve Stage-J and strictly audit the completed K0 encoder reuse."""

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
    pruning_units: str
    target_sparsity: float
    encoder_source: str


MATRIX = {
    item.experiment_id: item
    for item in (
        Experiment("j1", "j1", "j1_transformer_target75", "j1_transformer_target75", "historical_pred_heads", "8,12", "raw", 0.1, 1.0, "head,interm", 0.75, "stage-j"),
        Experiment("j2", "j2", "j2_hybrid_target70", "j2_hybrid_target70", "historical_pred_heads", "8,12", "raw", 0.1, 1.0, "conv,head,interm", 0.70, "stage-k-k0"),
        Experiment("j3", "j3", "j3_hybrid_target80", "j3_hybrid_target80", "historical_pred_heads", "8,12", "raw", 0.1, 1.0, "conv,head,interm", 0.80, "stage-j"),
    )
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not a mapping: {path}")
    return value


def _get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing required value {path}")
        current = current[key]
    return current


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _expect_paths(config: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for path, value in expected.items():
        _expect(_get(config, path), value, f"{label} {path}")


def validate_base_protocol(stage1_root: Path, stage2_root: Path) -> None:
    first = _load_yaml(stage1_root / "base_joint_dp_50k.yaml")
    second = _load_yaml(stage2_root / "base_post_distill_25k.yaml")
    _expect_paths(first, {
        "common.seed": 1337, "task.noise_prob": 0.25, "task.noise_snr": 0,
        "optimization.max_update": 50000, "optimization.clip_norm": 10.0,
        "optimization.update_freq": [4], "criterion.use_reg": True,
        "criterion.sparsity_warmup_updates": 10000,
        "optimizer._name": "composite", "optimizer.groups.main.lr": [0.002],
        "optimizer.groups.log_alpha.lr": [0.02], "optimizer.groups.lambda.lr": [-0.02],
        "optimizer.groups.main.lr_scheduler.warmup_updates": 15000,
    }, "Stage-1 base")
    _expect_paths(second, {
        "common.seed": 1337, "task.noise_prob": 0.25, "task.noise_snr": 0,
        "optimization.max_update": 25000, "optimization.clip_norm": 10.0,
        "optimization.update_freq": [4], "optimization.lr": [0.0001],
        "criterion.use_reg": False, "optimizer._name": "adam",
        "lr_scheduler.warmup_updates": 5000,
        "lr_scheduler.total_num_update": 25000,
    }, "Stage-2 base")


def resolve(experiment_id: str, stage1_root: Path, stage2_root: Path) -> tuple[Experiment, Path, Path]:
    if experiment_id not in MATRIX:
        raise ValueError(f"unknown Stage-J experiment {experiment_id!r}; choose from {', '.join(MATRIX)}")
    experiment = MATRIX[experiment_id]
    validate_base_protocol(stage1_root, stage2_root)
    stage1 = stage1_root / f"{experiment.stage1_config}.yaml"
    stage2 = stage2_root / f"{experiment.stage2_config}.yaml"
    _expect(_load_yaml(stage1).get("defaults"), ["base_joint_dp_50k", f"stage_j_matrix/{experiment.stage1_config}"], "Stage-1 defaults")
    _expect(_load_yaml(stage2).get("defaults"), ["base_post_distill_25k", f"stage_j_matrix/{experiment.stage2_config}"], "Stage-2 defaults")
    first = _load_yaml(stage1_root / "stage_j_matrix" / stage1.name)
    second = _load_yaml(stage2_root / "stage_j_matrix" / stage2.name)
    for cfg, label, weight in ((first, "Stage 1", experiment.stage1_l1), (second, "Stage 2", experiment.stage2_l1)):
        _expect_paths(cfg, {
            "model.distill_mode": experiment.matching,
            "model.distill_layers": experiment.targets,
            "criterion.cos_type": experiment.cosine,
            "criterion.l1_weight": weight,
        }, label)
    _expect(_get(first, "model.pruning_units"), experiment.pruning_units, "Stage-1 pruning units")
    _expect(float(_get(first, "criterion.target_sparsity")), experiment.target_sparsity, "Stage-1 target sparsity")
    return experiment, stage1.resolve(), stage2.resolve()


K0_STAGE1 = {
    "common.seed": 1337, "task.noise_prob": 0.25, "task.noise_snr": 0,
    "criterion.l1_weight": 0.1, "criterion.cos_type": "raw", "criterion.use_reg": True,
    "criterion.sparsity_warmup_updates": 10000, "criterion.target_sparsity": 0.7,
    "model.distill_mode": "historical_pred_heads", "model.distill_layers": "8,12",
    "model.pruning_units": "conv,head,interm", "optimization.max_update": 50000,
    "optimization.clip_norm": 10.0, "optimization.update_freq": [4],
    "optimizer._name": "composite", "optimizer.groups.main.lr": [0.002],
    "optimizer.groups.log_alpha.lr": [0.02], "optimizer.groups.lambda.lr": [-0.02],
    "optimizer.groups.main.lr_scheduler.warmup_updates": 15000,
}
K0_STAGE2 = {
    "common.seed": 1337, "task.noise_prob": 0.25, "task.noise_snr": 0,
    "criterion.l1_weight": 1.0, "criterion.cos_type": "raw", "criterion.use_reg": False,
    "model.distill_mode": "historical_pred_heads", "model.distill_layers": "8,12",
    "optimization.max_update": 25000, "optimization.clip_norm": 10.0,
    "optimization.update_freq": [4], "optimization.lr": [0.0001],
    "optimizer._name": "adam", "lr_scheduler.warmup_updates": 5000,
    "lr_scheduler.total_num_update": 25000,
}


def audit_k0(k0_root: Path) -> dict[str, Any]:
    """Fail closed unless K0 is the completed, scientifically identical J2 encoder."""
    root = k0_root.resolve()
    metadata_path = root / "run_metadata.json"
    stage1_path = root / "joint_dp/.hydra/config.yaml"
    stage2_path = root / "post_distill/.hydra/config.yaml"
    stage1_log = root / "joint_dp/hydra_train.log"
    stage2_log = root / "post_distill/hydra_train.log"
    checkpoints = {
        "stage1": root / "joint_dp/checkpoints/checkpoint_last.pt",
        "pruned": root / "pruned/student.pt",
        "stage2": root / "post_distill/checkpoints/checkpoint_last.pt",
        "export": root / "export/student.pt",
    }
    required = [metadata_path, stage1_path, stage2_path, stage1_log, stage2_log, *checkpoints.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("incomplete K0 reuse source; missing: " + ", ".join(missing))
    metadata = _load_json(metadata_path)
    _expect_paths(metadata, {
        "schema_version": "chapter3-stage-k-joint-dp/v1",
        "experiment_id": "k0", "output_id": "k0", "seed": 1337,
        "encoder_source": "stage-k",
    }, "K0 metadata")
    invocations = metadata.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("K0 metadata has no recorded invocation")
    protocol = invocations[-1].get("protocol", {})
    _expect_paths(protocol, {
        "matching": "historical_pred_heads", "teacher_targets": ["8", "12"],
        "cosine": "raw", "stage1_l1": 0.1, "stage2_l1": 1.0,
        "pruning_units": "conv,head,interm", "target_sparsity": 0.7,
        "stage1_updates": 50000, "stage2_updates": 25000,
        "seed": 1337, "noise_probability": 0.25, "noise_snr_db": 0,
    }, "K0 metadata protocol")
    stage1 = _load_yaml(stage1_path)
    stage2 = _load_yaml(stage2_path)
    _expect_paths(stage1, K0_STAGE1, "K0 resolved Stage 1")
    _expect_paths(stage2, K0_STAGE2, "K0 resolved Stage 2")
    first_text = stage1_log.read_text(encoding="utf-8", errors="replace")
    second_text = stage2_log.read_text(encoding="utf-8", errors="replace")
    if "Stopping training due to num_updates: 50000 >= max_update: 50000" not in first_text:
        raise ValueError("K0 Stage 1 log does not prove completion at 50000 updates")
    if "Stopping training due to num_updates: 25000 >= max_update: 25000" not in second_text:
        raise ValueError("K0 Stage 2 log does not prove completion at 25000 updates")
    realized = metadata.get("realized_sparsity")
    if not isinstance(realized, dict):
        raise ValueError("K0 metadata lacks realized sparsity")
    _expect(realized.get("original_parameters"), 103133416, "K0 original parameter count")
    _expect(realized.get("realized_parameters"), 31029874, "K0 deployed parameter count")
    expected_final = None
    for line in reversed(first_text.splitlines()):
        if '"sparsity_expected"' in line and '"num_updates": "50000"' in line:
            payload = json.loads(line[line.index("{") :])
            expected_final = float(payload["sparsity_expected"])
            break
    if expected_final is None:
        raise ValueError("K0 Stage 1 log lacks final expected sparsity")
    return {
        "root": str(root), "metadata": str(metadata_path),
        "resolved_stage1": str(stage1_path), "resolved_stage2": str(stage2_path),
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "teacher": _get(stage1, "model.teacher_path"),
        "stage1_student": _get(stage1, "model.student_path"),
        "expected_sparsity_final": expected_final,
        "realized_sparsity_global": float(realized["realized_sparsity"]),
        "deployed_parameter_count": int(realized["realized_parameters"]),
        "components": realized.get("components", {}),
        "audit": "passed: resolved configs, completion logs, checkpoint chain, and sparsity metadata",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=tuple(MATRIX))
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--k0-root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        experiment, stage1, stage2 = resolve(args.experiment, args.stage1_root.resolve(), args.stage2_root.resolve())
        k0 = None
        if args.experiment == "j2":
            if args.k0_root is None:
                raise ValueError("J2 requires --k0-root for strict completed-K0 reuse audit")
            k0 = audit_k0(args.k0_root)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    result = asdict(experiment)
    result.update({"stage1_path": str(stage1), "stage2_path": str(stage2), "k0_reuse": k0})
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("\t".join(str(result[key]) for key in (
            "output_id", "stage1_config", "stage2_config", "matching", "targets",
            "cosine", "stage1_l1", "stage2_l1", "pruning_units", "target_sparsity",
            "encoder_source", "stage1_path", "stage2_path",
        )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
