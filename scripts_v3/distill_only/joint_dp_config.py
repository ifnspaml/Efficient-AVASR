#!/usr/bin/env python3
"""Validate paired Chapter 3 v3 joint-DP loss configurations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def _criterion(path: Path) -> tuple[str | None, float | None]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    criterion = value.get("criterion") or {}
    cosine = criterion.get("cos_type")
    l1 = criterion.get("l1_weight")
    return (None if cosine in (None, "???") else str(cosine),
            None if l1 in (None, "???") else float(l1))


def resolve(
    experiment: str,
    stage1: Path,
    stage2: Path,
    requested_cosine: str | None,
    requested_l1: float | None,
) -> tuple[str, float]:
    if not stage1.is_file() or not stage2.is_file():
        raise ValueError("both stage configuration files must exist")
    first = _criterion(stage1)
    second = _criterion(stage2)
    if experiment.startswith("k"):
        if None in first or None in second or first != second:
            raise ValueError("K-series stage configs must encode one identical loss")
        cosine, l1 = first
        if requested_cosine is not None and requested_cosine != cosine:
            raise ValueError("--cosine-type disagrees with the K configuration")
        if requested_l1 is not None and requested_l1 != l1:
            raise ValueError("--l1-weight disagrees with the K configuration")
    elif experiment.startswith("j"):
        if requested_cosine is None or requested_l1 is None:
            raise ValueError("J-series requires --cosine-type and --l1-weight")
        cosine, l1 = requested_cosine, requested_l1
    else:
        raise ValueError("joint-DP experiment must be a K- or J-series config")
    if cosine not in {"raw", "log_sig"}:
        raise ValueError("cosine type must be raw or log_sig")
    if l1 is None or not math.isfinite(l1) or l1 < 0:
        raise ValueError("L1 weight must be finite and nonnegative")
    return cosine, l1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--cosine-type", choices=("raw", "log_sig"))
    parser.add_argument("--l1-weight", type=float)
    args = parser.parse_args()
    try:
        cosine, l1 = resolve(
            args.experiment, args.stage1, args.stage2,
            args.cosine_type, args.l1_weight,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"{cosine}\t{l1:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
