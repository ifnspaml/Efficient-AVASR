#!/usr/bin/env python3
"""Read-only parameter-count audit for joint-DP and materialized students."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _runtime(project: Path):
    sys.path[:0] = [str(project), str(project / "fairseq")]
    import avhubert  # noqa: F401
    from fairseq.checkpoint_utils import load_model_ensemble

    return load_model_ensemble


def _model(loader, checkpoint: Path):
    models, _ = loader([str(checkpoint.resolve())], strict=False)
    if len(models) != 1:
        raise ValueError(f"expected one model in {checkpoint}")
    model = models[0]
    if hasattr(model, "student"):
        model = model.student
    model.remove_pretraining_modules()
    return model


def parameter_report(project: Path, original: Path, pruned: Path) -> dict[str, Any]:
    loader = _runtime(project.resolve())
    teacher = _model(loader, original)
    student = _model(loader, pruned)

    def counts(component: str | None = None) -> dict[str, Any]:
        before = int(teacher.get_num_params(component))
        after = int(student.get_num_params(component))
        return {
            "original_parameters": before,
            "deployed_parameters": after,
            "realized_sparsity": 1.0 - after / before,
        }

    global_counts = counts()
    return {
        "original_checkpoint": str(original.resolve()),
        "pruned_checkpoint": str(pruned.resolve()),
        "deployed_parameter_count": global_counts["deployed_parameters"],
        "realized_sparsity_global": global_counts["realized_sparsity"],
        "global": global_counts,
        "components": {name: counts(name) for name in ("feature", "encoder")},
    }


def expected_from_log(path: Path) -> float:
    pattern = re.compile(r'"sparsity_expected":\s*"?([-+0-9.eE]+)')
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        match = pattern.search(line)
        if match and ('"num_updates": "50000"' in line or '"valid_num_updates": "50000"' in line):
            return float(match.group(1))
    raise ValueError(f"no final 50k expected-sparsity diagnostic in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--pruned", type=Path, required=True)
    parser.add_argument("--stage1-log", type=Path)
    args = parser.parse_args()
    try:
        report = parameter_report(args.project, args.original, args.pruned)
        if args.stage1_log:
            report["expected_sparsity_final"] = expected_from_log(args.stage1_log)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
