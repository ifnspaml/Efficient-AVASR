#!/usr/bin/env python3
"""Validate and expand the Chapter 3 evaluation protocol.

The shell runner consumes the tab-separated output so protocol policy remains
in one small, independently testable module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SUBSETS = {"valid", "test"}
SUPPORTED_NOISE_TYPES = {"babble", "music", "speech"}
FINAL_SNRS = {-10, -5, 0, 5, 10}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation protocol must contain a mapping")
    if value.get("schema_version") != "chapter3-evaluation/v1":
        raise ValueError("unsupported evaluation protocol schema")
    if value.get("noise_method") not in {"itut", "rms"}:
        raise ValueError("noise_method must be 'itut' or 'rms'")
    roots = value.get("noise_roots")
    if not isinstance(roots, dict) or set(roots) != SUPPORTED_NOISE_TYPES:
        raise ValueError("noise_roots must define babble, music, and speech")
    return value


def _suffix(snr: int) -> str:
    return f"m{-snr}db" if snr < 0 else (f"p{snr}db" if snr > 0 else "0db")


def expand_conditions(
    protocol_path: Path, phase: str, subsets: tuple[str, ...]
) -> list[dict[str, Any]]:
    protocol = _load(protocol_path)
    evaluation_seed = protocol.get("evaluation_seed")
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
        raise ValueError("evaluation_seed must be an integer")
    if not subsets or len(set(subsets)) != len(subsets):
        raise ValueError("subsets must be non-empty and unique")
    if not set(subsets) <= SUPPORTED_SUBSETS:
        raise ValueError("subsets must contain only valid and/or test")
    if phase == "screening" and subsets != ("valid",):
        raise ValueError("screening evaluation supports only the valid subset")
    method = str(protocol["noise_method"])
    roots = {key: str(value) for key, value in protocol["noise_roots"].items()}
    rows: list[dict[str, Any]] = []
    if phase == "screening":
        screening = protocol.get("screening_validation")
        if not isinstance(screening, dict) or screening.get("subset") != "valid":
            raise ValueError("screening_validation must target valid")
        if screening.get("include_clean") is not True:
            raise ValueError("screening_validation must include clean")
        entries = screening.get("conditions")
        if not isinstance(entries, list):
            raise ValueError("screening conditions must be a list")
        rows.append(_condition("clean", "valid", None, None, None, phase, method))
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("screening condition must be a mapping")
            noise_type = entry.get("noise_type")
            snr = entry.get("snr_db")
            if noise_type not in SUPPORTED_NOISE_TYPES or not isinstance(snr, int):
                raise ValueError("invalid screening noise type or SNR")
            expected = f"{noise_type}_{_suffix(snr)}"
            if entry.get("name") != expected:
                raise ValueError(f"screening condition name must be {expected}")
            rows.append(
                _condition(
                    expected, "valid", noise_type, roots[noise_type], snr, phase, method
                )
            )
        if [row["name"] for row in rows] != [
            "clean",
            "babble_0db",
            "speech_0db",
        ]:
            raise ValueError("screening conditions must be clean, babble_0db, speech_0db")
        for row in rows:
            row["evaluation_seed"] = evaluation_seed
        return rows
    if phase != "final":
        raise ValueError("phase must be screening or final")
    final = protocol.get("final_evaluation")
    if not isinstance(final, dict):
        raise ValueError("final_evaluation must be a mapping")
    noise_types = final.get("noise_types")
    snrs = final.get("snrs_db")
    if set(noise_types or []) != SUPPORTED_NOISE_TYPES:
        raise ValueError("final noise types must be babble, music, and speech")
    if set(snrs or []) != FINAL_SNRS:
        raise ValueError("final SNRs must be -10,-5,0,5,10")
    if final.get("include_clean") is not True:
        raise ValueError("final evaluation must include clean")
    for subset in subsets:
        rows.append(_condition("clean", subset, None, None, None, phase, method))
        for noise_type in noise_types:
            for snr in snrs:
                rows.append(
                    _condition(
                        f"{noise_type}_{_suffix(snr)}",
                        subset,
                        noise_type,
                        roots[noise_type],
                        int(snr),
                        phase,
                        method,
                    )
                )
    for row in rows:
        row["evaluation_seed"] = evaluation_seed
    return rows


def _condition(
    name: str,
    subset: str,
    noise_type: str | None,
    noise_root: str | None,
    snr: int | None,
    phase: str,
    method: str,
) -> dict[str, Any]:
    kind = noise_type or "clean"
    snr_part = "" if snr is None else str(snr)
    relative = f"{phase}/{method}/{kind}/{snr_part + '/' if snr_part else ''}{subset}"
    return {
        "name": name,
        "subset": subset,
        "noise_type": noise_type,
        "noise_root": noise_root,
        "snr_db": snr,
        "noise_method": method if noise_type else None,
        "relative_path": relative,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("screening", "final"), required=True)
    parser.add_argument("--subsets", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    subsets = tuple(args.subsets.split(","))
    rows = expand_conditions(args.protocol.resolve(), args.phase, subsets)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(
                "\t".join(
                    "-" if row[key] is None else str(row[key])
                    for key in (
                        "name",
                        "subset",
                        "noise_type",
                        "noise_root",
                        "snr_db",
                        "noise_method",
                        "evaluation_seed",
                        "relative_path",
                    )
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
