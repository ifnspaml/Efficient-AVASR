#!/usr/bin/env python3
"""Create an immutable C→D, D→E, or final scientific selection lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (REPO_ROOT / "exp" / "chapter3_distill_only").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.selection import (  # noqa: E402
    SELECTION_KINDS,
    SelectionError,
    create_selection,
    freeze_selected_manifest,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=SELECTION_KINDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument(
        "--conformer-match",
        type=Path,
        help="required chapter3-conformer-match/v1 artifact for c_to_d",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="explicit destination Hydra override, repeatable",
    )
    parser.add_argument(
        "--experiment-override",
        action="append",
        default=[],
        metavar="EXPERIMENT:KEY=VALUE",
        help="destination-specific materialization override, repeatable",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if os.path.commonpath((str(args.output.resolve()), str(OUTPUT_ROOT))) != str(OUTPUT_ROOT):
        print(f"error: selection output must be below {OUTPUT_ROOT}", file=sys.stderr)
        return 2
    for candidate in [*args.candidate, args.selected]:
        candidate_path = candidate.resolve()
        if candidate_path.name != "manifest.v1.json":
            candidate_path = candidate_path / "manifest.v1.json"
        if os.path.commonpath((str(candidate_path), str(OUTPUT_ROOT))) != str(OUTPUT_ROOT):
            print(
                f"error: candidate manifest must be below {OUTPUT_ROOT}: {candidate}",
                file=sys.stderr,
            )
            return 2
    overrides_by_experiment = {}
    for value in args.experiment_override:
        try:
            experiment, override = value.split(":", 1)
        except ValueError:
            print(
                "error: --experiment-override must be EXPERIMENT:KEY=VALUE",
                file=sys.stderr,
            )
            return 2
        if "=" not in override:
            print("error: experiment override must contain '='", file=sys.stderr)
            return 2
        overrides_by_experiment.setdefault(experiment, []).append(override)
    try:
        record = create_selection(
            args.output,
            kind=args.kind,
            candidates=args.candidate,
            selected=args.selected,
            metric=args.metric,
            rationale=args.rationale,
            approver=args.approver,
            materialized_overrides=args.override,
            overrides_by_experiment=overrides_by_experiment,
            conformer_match=args.conformer_match,
        )
        if args.kind == "final":
            freeze_selected_manifest(args.output)
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
