#!/usr/bin/env python3
"""Generate validation-only JSON/Markdown comparisons from run manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.manifest import (  # noqa: E402
    ManifestStore,
    atomic_write_json,
    utc_now,
)


def _get(value: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return default
        current = current[component]
    return current


def _sum_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        return sum(_sum_numeric(item) for item in value.values())
    return 0.0


def manifest_row(path: Path) -> Dict[str, Any]:
    manifest = ManifestStore(path).read()
    config = _get(manifest, "immutable.resolved_config", {})
    validation = _get(manifest, "measurements.wer.validation", {})
    profile = _get(manifest, "measurements.profile", {})
    memory = _get(manifest, "measurements.peak_memory", {})
    return {
        "manifest": str(ManifestStore(path).path.resolve()),
        "experiment": _get(manifest, "immutable.experiment"),
        "seed": _get(manifest, "immutable.seed"),
        "state": manifest["state"],
        "architecture": _get(config, "model.student_arch"),
        "depth": _get(config, "model.student_depth"),
        "embed_dim": _get(config, "model.student_embed_dim"),
        "ffn_dim": _get(config, "model.student_ffn_dim"),
        "objective": _get(config, "model.distill_head_mode"),
        "clean_validation_wer": _get(validation, "clean.value"),
        "babble_0db_validation_wer": _get(validation, "babble_0db.value"),
        "deployed_parameters": profile.get("deployed_backbone_parameters"),
        "prediction_head_parameters": profile.get(
            "training_only_prediction_head_parameters"
        ),
        "macs": profile.get("macs"),
        "flops": profile.get("flops"),
        "runtime_seconds": _sum_numeric(_get(manifest, "runtime.duration_seconds", {})),
        "peak_memory_bytes": max(
            (
                int(item.get("peak_gpu_memory_bytes", 0))
                for item in memory.values()
                if isinstance(item, Mapping)
            ),
            default=0,
        ),
    }


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return str(value)
    return str(value)


def markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    columns = (
        ("experiment", "Experiment"),
        ("architecture", "Arch"),
        ("depth", "Depth"),
        ("objective", "Objective"),
        ("clean_validation_wer", "Valid clean WER"),
        ("babble_0db_validation_wer", "Valid babble 0 dB WER"),
        ("deployed_parameters", "Params"),
        ("flops", "FLOPs"),
        ("runtime_seconds", "Runtime s"),
        ("peak_memory_bytes", "Peak bytes"),
    )
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_display(row.get(key)) for key, _ in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        rows = [manifest_row(path) for path in args.manifest]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = {
        "schema_version": "chapter3-distill-comparison/v1",
        "generated_at": utc_now(),
        "selection_data_policy": "validation-only",
        "rows": rows,
    }
    if args.json_output:
        atomic_write_json(args.json_output, report)
    rendered = markdown(rows)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.markdown_output.with_name(f".{args.markdown_output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.markdown_output)
    if not args.json_output and not args.markdown_output:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
