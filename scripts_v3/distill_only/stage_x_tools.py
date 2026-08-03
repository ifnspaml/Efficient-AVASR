#!/usr/bin/env python3
"""Phase-gated AV-HuBERT-large target, reset, and budget utilities."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _get(value: Any, *path: str) -> Any:
    for key in path:
        if isinstance(value, dict):
            value = value[key]
        else:
            value = getattr(value, key)
    return value


def checkpoint_architecture(checkpoint: Path) -> tuple[int, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("cfg")
    if cfg is None:
        raise ValueError("checkpoint does not contain cfg metadata")
    try:
        depth = int(_get(cfg, "model", "encoder_layers"))
        dimension = int(_get(cfg, "model", "encoder_embed_dim"))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint does not expose encoder depth/dimension") from error
    if depth <= 0 or dimension <= 0:
        raise ValueError("encoder depth and dimension must be positive")
    return depth, dimension


def relative_targets(depth: int) -> tuple[int, int, int, int]:
    if depth < 3:
        raise ValueError("relative four-target mapping requires depth >= 3")
    one_third = math.floor(depth / 3 + 0.5)
    two_thirds = math.floor(2 * depth / 3 + 0.5)
    targets = (0, one_third, two_thirds, depth)
    if len(set(targets)) != len(targets) or not all(0 <= item <= depth for item in targets):
        raise ValueError(f"invalid relative target mapping for depth {depth}: {targets}")
    return targets


def _prepare_runtime(project: Path):
    sys.path.insert(0, str(project / "avhubert"))
    sys.path.insert(0, str(project / "fairseq"))
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


def reset_encoder(project: Path, checkpoint: Path, output: Path, seed: int) -> None:
    load_model_ensemble = _prepare_runtime(project)
    models, _ = load_model_ensemble([str(checkpoint)], strict=False)
    if len(models) != 1 or hasattr(models[0], "student"):
        raise ValueError("reset input must be one physically materialized native encoder")
    model = models[0]
    original_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    if any("log_alpha" in key for key in original_shapes):
        raise ValueError("reset input still contains architecture-search parameters")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def reset(module: torch.nn.Module) -> None:
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    model.apply(reset)
    reset_state = model.state_dict()
    if {key: tuple(value.shape) for key, value in reset_state.items()} != original_shapes:
        raise ValueError("encoder reset changed the materialized architecture")
    if any("log_alpha" in key for key in reset_state):
        raise ValueError("encoder reset retained architecture-search parameters")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["model"] = reset_state
    state["criterion"] = None
    state["optimizer_history"] = []
    state["last_optimizer_state"] = None
    extra = state.setdefault("extra_state", {})
    extra.pop("distill_linear_projs", None)
    _atomic_save(state, output)
    reloaded, _ = load_model_ensemble([str(output)], strict=False)
    if len(reloaded) != 1:
        raise ValueError("reset checkpoint failed structural reload")


def _load_deployed(project: Path, checkpoint: Path):
    load_model_ensemble = _prepare_runtime(project)
    models, _ = load_model_ensemble([str(checkpoint)], strict=False)
    if len(models) != 1:
        raise ValueError("checkpoint must reload as one model")
    wrapper = models[0]
    if hasattr(wrapper, "student"):
        deployed = wrapper.student
        heads = getattr(wrapper, "distill_linear_projs", None)
        training_only = sum(p.numel() for p in heads.parameters()) if heads else 0
        training_only += sum(
            p.numel() for name, p in deployed.named_parameters() if "log_alpha" in name
        )
    else:
        deployed = wrapper
        training_only = 0
    deployed.eval()
    deployed_parameters = sum(
        parameter.numel()
        for name, parameter in deployed.named_parameters()
        if "log_alpha" not in name
    )
    return deployed, deployed_parameters, training_only


def profile(project: Path, checkpoint: Path) -> dict[str, float | int]:
    from ptflops import get_model_complexity_info

    model, deployed_parameters, training_only = _load_deployed(project, checkpoint)

    def data(_: Any) -> dict[str, Any]:
        return {
            "source": {
                "audio": torch.ones((1, 104, 100)),
                "video": torch.ones((1, 1, 100, 88, 88)),
            },
            "features_only": True,
            "mask": False,
        }

    macs, _ = get_model_complexity_info(
        model,
        (1, 768),
        input_constructor=data,
        as_strings=False,
        backend="aten",
        print_per_layer_stat=False,
        verbose=False,
    )
    return {
        "deployed_parameters": deployed_parameters,
        "training_only_parameters": training_only,
        "macs_100_frames": float(macs),
        "flops_100_frames": float(macs) * 2.0,
        "macs_per_frame": float(macs) / 100.0,
        "flops_per_frame": float(macs) * 2.0 / 100.0,
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    targets = subparsers.add_parser("resolve-targets")
    targets.add_argument("--checkpoint", type=Path, required=True)
    targets.add_argument("--output", type=Path)
    reset = subparsers.add_parser("reset-encoder")
    reset.add_argument("--checkpoint", type=Path, required=True)
    reset.add_argument("--output", type=Path, required=True)
    reset.add_argument("--seed", type=int, default=1337)
    budget = subparsers.add_parser("budget-report")
    budget.add_argument("--candidate", type=Path, required=True)
    budget.add_argument("--reference", type=Path, required=True)
    budget.add_argument("--parameter-tolerance", type=float)
    budget.add_argument("--flop-tolerance", type=float)
    budget.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_path.resolve()
    if args.action == "resolve-targets":
        depth, dimension = checkpoint_architecture(args.checkpoint.resolve())
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "teacher_depth": depth,
            "teacher_embed_dim": dimension,
            "resolved_targets": list(relative_targets(depth)),
        }
        _write_report(report, args.output)
    elif args.action == "reset-encoder":
        if args.seed < 0:
            parser.error("seed must be nonnegative")
        reset_encoder(project, args.checkpoint.resolve(), args.output.resolve(), args.seed)
    else:
        if (args.parameter_tolerance is None) != (args.flop_tolerance is None):
            parser.error("both tolerances must be supplied together")
        candidate = profile(project, args.candidate.resolve())
        reference = profile(project, args.reference.resolve())
        parameter_difference = abs(candidate["deployed_parameters"] - reference["deployed_parameters"]) / reference["deployed_parameters"]
        flop_difference = abs(candidate["flops_100_frames"] - reference["flops_100_frames"]) / reference["flops_100_frames"]
        matched = None
        if args.parameter_tolerance is not None:
            if args.parameter_tolerance < 0 or args.flop_tolerance < 0:
                parser.error("tolerances must be nonnegative")
            matched = parameter_difference <= args.parameter_tolerance and flop_difference <= args.flop_tolerance
        _write_report(
            {
                "candidate": candidate,
                "reference": reference,
                "relative_parameter_difference": parameter_difference,
                "relative_flop_difference": flop_difference,
                "matched": matched,
            },
            args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
