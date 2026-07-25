#!/usr/bin/env python3
"""Profile a distillation checkpoint with the fixed audiovisual convention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "fairseq"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chapter3_distill_only.manifest import atomic_write_json, utc_now  # noqa: E402
from chapter3_distill_only.profiling import profile_distillation_model  # noqa: E402


def profile_checkpoint(checkpoint: Path, user_dir: Path) -> dict:
    from fairseq import checkpoint_utils

    # Importing chapter3_distill_only.profiling above has already imported the
    # package and registered its Fairseq components. Calling
    # utils.import_user_module here would import the same user module twice and
    # Fairseq correctly rejects that as non-unique.
    expected_user_dir = REPO_ROOT / "chapter3_distill_only"
    if user_dir.resolve() != expected_user_dir.resolve():
        raise RuntimeError(
            f"user-dir must be the isolated runtime package: {expected_user_dir}"
        )
    models, _ = checkpoint_utils.load_model_ensemble([str(checkpoint)], strict=False)
    if len(models) != 1:
        raise RuntimeError(f"expected one model, loaded {len(models)}")
    model = models[0]
    model.cpu()
    result = profile_distillation_model(model)
    result.update(
        {
            "checkpoint": str(checkpoint.resolve()),
            "measured_at": utc_now(),
        }
    )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=REPO_ROOT / "chapter3_distill_only",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        print(f"error: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
        return 2
    try:
        result = profile_checkpoint(args.checkpoint, args.user_dir)
        atomic_write_json(args.output, result)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
