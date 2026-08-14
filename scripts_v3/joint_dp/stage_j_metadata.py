#!/usr/bin/env python3
"""Record Stage-J provenance without conflating configured, expected, and realized sparsity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_sparsity import expected_from_log, parameter_report
from stage_j_config import audit_k0


SCHEMA = "chapter3-stage-j-joint-dp/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not a mapping: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def record(args: argparse.Namespace) -> None:
    path = args.metadata.resolve()
    identity = {
        "schema_version": SCHEMA, "experiment_id": args.experiment,
        "seed": args.seed, "encoder_source": args.encoder_source,
        "stage1_config": str(args.stage1_config.resolve()),
        "stage2_config": str(args.stage2_config.resolve()),
    }
    value = _read(path)
    if value:
        for key, expected in identity.items():
            if value.get(key) != expected:
                raise ValueError(f"incompatible Stage-J metadata: {key} is {value.get(key)!r}, expected {expected!r}")
    else:
        value = {
            **identity, "created_at": _now(), "invocations": [],
            "target_sparsity_configured": args.target_sparsity,
            "expected_sparsity_final": None,
            "realized_sparsity_global": None,
            "deployed_parameter_count": None,
            "component_parameter_counts": None,
        }
    if args.reuse_k0_root:
        reuse = audit_k0(args.reuse_k0_root)
        value.update({
            "expected_sparsity_final": reuse["expected_sparsity_final"],
            "realized_sparsity_global": reuse["realized_sparsity_global"],
            "deployed_parameter_count": reuse["deployed_parameter_count"],
            "component_parameter_counts": reuse["components"],
            "reuse_audit": reuse,
        })
    value["invocations"].append({
        "timestamp": _now(), "requested_stage": args.stage,
        "source": {"branch": args.branch, "commit": args.commit, "dirty": args.dirty},
        "protocol": {
            "matching": args.matching, "teacher_targets": args.targets.split(","),
            "cosine": args.cosine, "stage1_l1": args.stage1_l1,
            "stage2_l1": args.stage2_l1, "pruning_units": args.pruning_units,
            "target_sparsity_configured": args.target_sparsity,
            "stage1_updates": 50000, "stage2_updates": 25000,
            "seed": args.seed, "noise_probability": 0.25, "noise_snr_db": 0,
        },
        "checkpoints": {
            "teacher": args.teacher, "stage1_student_input": args.student_input,
            "stage1": args.joint_checkpoint, "pruned": args.pruned_checkpoint,
            "stage2": args.stage2_checkpoint, "exported_student": args.export_checkpoint,
            "fine_tuned": args.finetune_checkpoint,
        },
        "resolved_config_outputs": {"stage1": args.stage1_resolved, "stage2": args.stage2_resolved},
        "fine_tuning": {
            "config": args.finetune_config, "run_name": args.run_name,
            "updates": 60000, "freeze_updates": 48000, "peak_lr": args.finetune_lr,
            "clip_norm": 0.0, "update_freq": 8,
            "noise_probability": 0.25, "noise_snr_db": 0,
        },
        "evaluation": {
            "protocol": args.evaluation_protocol, "phase": args.evaluation_phase,
            "subsets": args.evaluation_subsets.split(","),
            "selection_policy": "screen on validation clean/babble0/speech0; test is reporting-only",
        },
    })
    _write(path, value)


def update_expected(args: argparse.Namespace) -> None:
    path = args.metadata.resolve()
    value = _read(path)
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"missing compatible Stage-J metadata: {path}")
    value["expected_sparsity_final"] = expected_from_log(args.stage1_log.resolve())
    _write(path, value)


def update_realized(args: argparse.Namespace) -> None:
    path = args.metadata.resolve()
    value = _read(path)
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"missing compatible Stage-J metadata: {path}")
    report = parameter_report(args.project, args.teacher, args.pruned)
    value["realized_sparsity_global"] = report["realized_sparsity_global"]
    value["deployed_parameter_count"] = report["deployed_parameter_count"]
    value["component_parameter_counts"] = report["components"]
    value["sparsity_audit"] = report
    _write(path, value)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    rec = sub.add_parser("record")
    for name in ("experiment", "encoder-source", "stage", "branch", "commit", "matching", "targets", "cosine", "pruning-units", "teacher", "student-input", "joint-checkpoint", "pruned-checkpoint", "stage2-checkpoint", "export-checkpoint", "finetune-checkpoint", "stage1-resolved", "stage2-resolved", "finetune-config", "run-name", "evaluation-protocol", "evaluation-phase", "evaluation-subsets"):
        rec.add_argument(f"--{name}", required=True)
    rec.add_argument("--metadata", type=Path, required=True)
    rec.add_argument("--stage1-config", type=Path, required=True)
    rec.add_argument("--stage2-config", type=Path, required=True)
    rec.add_argument("--seed", type=int, required=True)
    rec.add_argument("--stage1-l1", type=float, required=True)
    rec.add_argument("--stage2-l1", type=float, required=True)
    rec.add_argument("--target-sparsity", type=float, required=True)
    rec.add_argument("--finetune-lr", type=float, required=True)
    rec.add_argument("--reuse-k0-root", type=Path)
    rec.add_argument("--dirty", action="store_true")
    expected = sub.add_parser("expected-sparsity")
    expected.add_argument("--metadata", type=Path, required=True)
    expected.add_argument("--stage1-log", type=Path, required=True)
    realized = sub.add_parser("realized-sparsity")
    realized.add_argument("--metadata", type=Path, required=True)
    realized.add_argument("--project", type=Path, required=True)
    realized.add_argument("--teacher", type=Path, required=True)
    realized.add_argument("--pruned", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "record":
            record(args)
        elif args.action == "expected-sparsity":
            update_expected(args)
        else:
            update_realized(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
