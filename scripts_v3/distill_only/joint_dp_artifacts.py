#!/usr/bin/env python3
"""Write Chapter 3 joint-DP prune/export artifacts to explicit paths."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def _prepare_imports(project: Path):
    sys.path.insert(0, str(project / "avhubert"))
    sys.path.insert(0, str(project / "fairseq"))
    from fairseq.checkpoint_utils import load_model_ensemble
    from prune import prune_from_ckpt
    from save_final_ckpt import save_from_ckpt

    return load_model_ensemble, prune_from_ckpt, save_from_ckpt


def _atomic_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify(project: Path, checkpoint: Path) -> None:
    load_model_ensemble, _, _ = _prepare_imports(project)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state or "cfg" not in state:
        raise ValueError("artifact is not a native Fairseq checkpoint")
    models, _ = load_model_ensemble([str(checkpoint)], strict=False)
    if len(models) != 1:
        raise ValueError("artifact did not reload as exactly one model")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prune", "export", "verify"))
    parser.add_argument("--project-path", type=Path, required=True)
    parser.add_argument("--distilled-checkpoint", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_path.resolve()
    if args.action == "verify":
        if args.checkpoint is None:
            parser.error("verify requires --checkpoint")
        verify(project, args.checkpoint.resolve())
        return 0
    if None in (args.distilled_checkpoint, args.original_checkpoint, args.output):
        parser.error(f"{args.action} requires distilled, original, and output paths")
    _, prune_from_ckpt, save_from_ckpt = _prepare_imports(project)
    distilled = args.distilled_checkpoint.resolve()
    original = args.original_checkpoint.resolve()
    output = args.output.resolve()
    if args.action == "prune":
        payload = prune_from_ckpt(str(distilled), str(original))
    else:
        payload = save_from_ckpt(str(distilled), str(original))
    _atomic_save(payload, output)
    verify(project, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
