#!/usr/bin/env python3
"""Import read-only A0 joint-DP artifacts into the isolated result namespace."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (REPO_ROOT / "exp" / "chapter3_distill_only").resolve()
BASELINE_BRANCH = "dev_li_pro6000"
BASELINE_COMMIT = "4502130ce4470fe8b0456fd5b466bef043f383f9"
DEFAULT_TEACHER = Path(
    "/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/base_vox_iter5.pt"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.manifest import (  # noqa: E402
    atomic_write_json,
    git_provenance,
    sha256_file,
    utc_now,
)


def _condition_path(value: str) -> tuple[str, Path]:
    try:
        condition, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "WER artifact must have form CONDITION=PATH"
        ) from exc
    if not condition:
        raise argparse.ArgumentTypeError("WER condition cannot be empty")
    return condition, Path(raw_path)


def _wer_value(path: Path) -> float:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    try:
        return float(first_line.split(":", 1)[1].strip().rstrip("%"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse numeric WER from {path}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--wer-artifact",
        type=_condition_path,
        action="append",
        required=True,
        help="repeatable CONDITION=PATH validation/test WER artifact",
    )
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deployed-backbone-parameters", type=int, required=True)
    parser.add_argument("--macs", type=int, required=True)
    parser.add_argument("--flops", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if os.path.commonpath((str(args.output.resolve()), str(OUTPUT_ROOT))) != str(OUTPUT_ROOT):
        print(f"error: reference output must be below {OUTPUT_ROOT}", file=sys.stderr)
        return 2
    if args.teacher_checkpoint.resolve() != DEFAULT_TEACHER.resolve():
        print(f"error: teacher must be fixed at {DEFAULT_TEACHER}", file=sys.stderr)
        return 2
    conditions = [condition for condition, _ in args.wer_artifact]
    if len(set(conditions)) != len(conditions):
        print("error: duplicate WER artifact condition", file=sys.stderr)
        return 2
    paths = [
        args.checkpoint,
        args.config,
        args.teacher_checkpoint,
        *(path for _, path in args.wer_artifact),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print("error: missing reference artifacts: " + ", ".join(missing), file=sys.stderr)
        return 2
    if args.flops != 2 * args.macs:
        print("error: reference FLOPs must equal 2 * MACs", file=sys.stderr)
        return 2
    if args.deployed_backbone_parameters <= 0 or args.macs <= 0:
        print("error: reference parameters and MACs must be positive", file=sys.stderr)
        return 2
    try:
        wer = {
            condition: {
                "value": _wer_value(path),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for condition, path in args.wer_artifact
        }
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    record = {
        "schema_version": "chapter3-joint-dp-reference/v1",
        "created_at": utc_now(),
        "read_only": True,
        "method": "A0_joint_dp_reference",
        "baseline": {
            "branch": BASELINE_BRANCH,
            "commit": BASELINE_COMMIT,
        },
        "seed": args.seed,
        "schedule": {
            "stage1": {
                "updates": 50000,
                "method": "joint_distillation_and_pruning",
            },
            "stage2": {
                "updates": 25000,
                "method": "further_distillation",
            },
            "total_encoder_updates": 75000,
        },
        "measurements": {
            "deployed_backbone_parameters": args.deployed_backbone_parameters,
            "macs": args.macs,
            "flops": args.flops,
            "flop_definition": "2 * MACs",
            "wer": wer,
        },
        "repository": git_provenance(REPO_ROOT),
        "artifacts": {
            "teacher_checkpoint": {
                "path": str(args.teacher_checkpoint.resolve()),
                "sha256": sha256_file(args.teacher_checkpoint),
            },
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint),
            },
            "config": {
                "path": str(args.config.resolve()),
                "sha256": sha256_file(args.config),
            },
            "wer": wer,
        },
    }
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot inspect existing output: {exc}", file=sys.stderr)
            return 2
        comparable_new = dict(record)
        comparable_old = dict(existing)
        comparable_new.pop("created_at", None)
        comparable_old.pop("created_at", None)
        if comparable_new != comparable_old:
            print(f"error: refusing to replace different reference {args.output}", file=sys.stderr)
            return 2
    else:
        atomic_write_json(args.output, record)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
