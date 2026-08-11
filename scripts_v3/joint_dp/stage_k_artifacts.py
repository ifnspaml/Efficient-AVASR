#!/usr/bin/env python3
"""Materialize, export, and verify Stage-K joint-DP checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def _runtime(project: Path):
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "fairseq"))
    import avhubert  # noqa: F401
    from fairseq.checkpoint_utils import load_model_ensemble

    return load_model_ensemble


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


def _wrapper(load_model_ensemble, checkpoint: Path):
    models, _ = load_model_ensemble([str(checkpoint)])
    if len(models) != 1 or not hasattr(models[0], "student"):
        raise ValueError(f"expected one joint-DP wrapper in {checkpoint}")
    return models[0]


def prune(load_model_ensemble, distilled: Path, original: Path) -> dict[str, Any]:
    wrapper = _wrapper(load_model_ensemble, distilled)
    student = wrapper.student
    (
        resnet_config,
        encoder_use_attention,
        encoder_use_ffn,
        encoder_attention_heads,
        encoder_ffn_embed_dim,
    ) = student.prune()
    pruning_cfg = {
        "resnet_prune_conv_channels": False,
        "encoder_prune_attention_heads": False,
        "encoder_prune_attention_layer": False,
        "encoder_prune_ffn_intermediate": False,
        "encoder_prune_ffn_layer": False,
        "encoder_use_attention": encoder_use_attention,
        "encoder_attention_heads_detailed": encoder_attention_heads,
        "encoder_use_ffn": encoder_use_ffn,
        "encoder_ffn_embed_dim_detailed": encoder_ffn_embed_dim,
        "resnet_layers_detailed": resnet_config,
    }
    state = torch.load(original, map_location="cpu", weights_only=False)
    state["model"] = student.state_dict()
    state["cfg"]["model"].update(pruning_cfg)
    state.setdefault("extra_state", {})["distill_linear_projs"] = (
        wrapper.distill_linear_projs.state_dict()
    )
    return state


def export(load_model_ensemble, distilled: Path, original: Path) -> dict[str, Any]:
    wrapper = _wrapper(load_model_ensemble, distilled)
    state = torch.load(original, map_location="cpu", weights_only=False)
    state["model"] = wrapper.student.state_dict()
    state.setdefault("extra_state", {})["distill_linear_projs"] = (
        wrapper.distill_linear_projs.state_dict()
    )
    return state


def verify(load_model_ensemble, checkpoint: Path) -> None:
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
    load_model_ensemble = _runtime(project)
    if args.action == "verify":
        if args.checkpoint is None:
            parser.error("verify requires --checkpoint")
        verify(load_model_ensemble, args.checkpoint.resolve())
        return 0
    if None in (args.distilled_checkpoint, args.original_checkpoint, args.output):
        parser.error(f"{args.action} requires distilled, original, and output paths")
    distilled = args.distilled_checkpoint.resolve()
    original = args.original_checkpoint.resolve()
    output = args.output.resolve()
    if args.action == "prune":
        payload = prune(load_model_ensemble, distilled, original)
    else:
        payload = export(load_model_ensemble, distilled, original)
    _atomic_save(payload, output)
    verify(load_model_ensemble, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
