#!/usr/bin/env python3
"""Write auditable Stage-K invocation metadata and realized sparsity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "chapter3-stage-k-joint-dp/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not a mapping: {path}")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
        "schema_version": SCHEMA,
        "experiment_id": args.experiment,
        "output_id": args.output_id,
        "seed": args.seed,
        "stage1_config": str(args.stage1_config.resolve()),
        "stage2_config": str(args.stage2_config.resolve()),
        "teacher_checkpoint": args.teacher,
        "stage1_student_input": args.student_input,
    }
    value = _read(path)
    if value:
        for key, expected in identity.items():
            if value.get(key) != expected:
                raise ValueError(
                    f"incompatible existing Stage-K metadata: {key} is "
                    f"{value.get(key)!r}, expected {expected!r}"
                )
    else:
        value = {
            **identity,
            "created_at": _now(),
            "invocations": [],
            "realized_sparsity": None,
        }
    value["invocations"].append(
        {
            "timestamp": _now(),
            "stage": args.stage,
            "source": {
                "branch": args.branch,
                "commit": args.commit,
                "dirty": args.dirty,
            },
            "protocol": {
                "matching": args.matching,
                "teacher_targets": args.targets.split(","),
                "cosine": args.cosine,
                "stage1_l1": args.stage1_l1,
                "stage2_l1": args.stage2_l1,
                "pruning_units": "conv,head,interm",
                "target_sparsity": 0.70,
                "stage1_updates": 50000,
                "stage2_updates": 25000,
                "seed": args.seed,
                "noise_probability": 0.25,
                "noise_snr_db": 0,
            },
            "checkpoints": {
                "teacher": args.teacher,
                "stage1_student_input": args.student_input,
                "outputs": {
                    "stage1": args.joint_checkpoint,
                    "pruned": args.pruned_checkpoint,
                    "stage2": args.stage2_checkpoint,
                    "exported_student": args.export_checkpoint,
                    "fine_tuned": args.finetune_checkpoint,
                },
                "stage_inputs": {
                    "prune": args.prune_input,
                    "stage2": args.stage2_input,
                    "export": args.export_input,
                    "finetune": args.finetune_input,
                    "evaluation": args.evaluation_input,
                },
            },
            "resolved_config_outputs": {
                "stage1": args.stage1_resolved,
                "stage2": args.stage2_resolved,
            },
            "fine_tuning": {
                "config": args.finetune_config,
                "run_name": args.run_name,
                "updates": 60000,
                "freeze_updates": 48000,
                "peak_lr": args.finetune_lr,
                "clip_norm": 0.0,
                "update_freq": 8,
                "noise_probability": 0.25,
                "noise_snr_db": 0,
            },
            "evaluation": {
                "protocol": args.evaluation_protocol,
                "phase": args.evaluation_phase,
                "subsets": args.evaluation_subsets.split(","),
                "selection_policy": "validation-only; test is reporting-only",
            },
        }
    )
    _atomic_write(path, value)


def realized_sparsity(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "fairseq"))
    # Register the checkpoint model/task exactly as the joint-DP user-dir does.
    import avhubert  # noqa: F401

    from fairseq.checkpoint_utils import load_model_ensemble

    def load(checkpoint: Path):
        models, _ = load_model_ensemble([str(checkpoint.resolve())], strict=False)
        if len(models) != 1:
            raise ValueError(f"expected one model in {checkpoint}")
        model = models[0]
        if hasattr(model, "student"):
            model = model.student
        model.remove_pretraining_modules()
        return model

    teacher = load(args.teacher)
    student = load(args.pruned)
    original_total = float(teacher.get_num_params())
    realized_total = float(student.get_num_params())
    report = {
        "recorded_at": _now(),
        "teacher_checkpoint": str(args.teacher.resolve()),
        "pruned_checkpoint": str(args.pruned.resolve()),
        "original_parameters": int(original_total),
        "realized_parameters": int(realized_total),
        "realized_sparsity": 1.0 - realized_total / original_total,
        "components": {},
    }
    for component in ("feature", "encoder"):
        original = float(teacher.get_num_params(component))
        realized = float(student.get_num_params(component))
        report["components"][component] = {
            "original_parameters": int(original),
            "realized_parameters": int(realized),
            "realized_sparsity": 1.0 - realized / original,
        }
    path = args.metadata.resolve()
    value = _read(path)
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"missing compatible Stage-K metadata: {path}")
    value["realized_sparsity"] = report
    _atomic_write(path, value)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--metadata", type=Path, required=True)
    record_parser.add_argument("--experiment", required=True)
    record_parser.add_argument("--output-id", required=True)
    record_parser.add_argument("--seed", type=int, required=True)
    record_parser.add_argument("--stage", required=True)
    record_parser.add_argument("--branch", required=True)
    record_parser.add_argument("--commit", required=True)
    record_parser.add_argument("--dirty", action="store_true")
    record_parser.add_argument("--stage1-config", type=Path, required=True)
    record_parser.add_argument("--stage2-config", type=Path, required=True)
    record_parser.add_argument("--matching", required=True)
    record_parser.add_argument("--targets", required=True)
    record_parser.add_argument("--cosine", required=True)
    record_parser.add_argument("--stage1-l1", type=float, required=True)
    record_parser.add_argument("--stage2-l1", type=float, required=True)
    record_parser.add_argument("--teacher", required=True)
    record_parser.add_argument("--student-input", required=True)
    record_parser.add_argument("--joint-checkpoint", required=True)
    record_parser.add_argument("--pruned-checkpoint", required=True)
    record_parser.add_argument("--stage2-checkpoint", required=True)
    record_parser.add_argument("--export-checkpoint", required=True)
    record_parser.add_argument("--finetune-checkpoint", required=True)
    record_parser.add_argument("--prune-input", required=True)
    record_parser.add_argument("--stage2-input", required=True)
    record_parser.add_argument("--export-input", required=True)
    record_parser.add_argument("--finetune-input", required=True)
    record_parser.add_argument("--evaluation-input", required=True)
    record_parser.add_argument("--stage1-resolved", required=True)
    record_parser.add_argument("--stage2-resolved", required=True)
    record_parser.add_argument("--finetune-config", required=True)
    record_parser.add_argument("--run-name", required=True)
    record_parser.add_argument("--finetune-lr", type=float, required=True)
    record_parser.add_argument("--evaluation-protocol", required=True)
    record_parser.add_argument("--evaluation-phase", required=True)
    record_parser.add_argument("--evaluation-subsets", required=True)
    sparsity_parser = subparsers.add_parser("realized-sparsity")
    sparsity_parser.add_argument("--metadata", type=Path, required=True)
    sparsity_parser.add_argument("--project", type=Path, required=True)
    sparsity_parser.add_argument("--teacher", type=Path, required=True)
    sparsity_parser.add_argument("--pruned", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "record":
            record(args)
        else:
            realized_sparsity(args)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
