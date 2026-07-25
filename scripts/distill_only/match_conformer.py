#!/usr/bin/env python3
"""Choose a Conformer FFN size from pre-profiled multiples of 128."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.manifest import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from chapter3_distill_only.profiling import (  # noqa: E402
    select_conformer_ffn_match,
    validate_config_profile,
)


def _candidate(value: str) -> Dict[str, Any]:
    try:
        raw_ffn, raw_path = value.split("=", 1)
        ffn_dim = int(raw_ffn)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must have form FFN_DIM=profile.json") from exc
    profile = read_json(Path(raw_path))
    try:
        validated = validate_config_profile(
            profile, expected_arch="conformer"
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if validated["student_ffn_dim"] != ffn_dim:
        raise argparse.ArgumentTypeError(
            f"CLI FFN {ffn_dim} does not match profile FFN "
            f"{validated['student_ffn_dim']} in {raw_path}"
        )
    profile["profile_path"] = str(Path(raw_path).resolve())
    profile["profile_sha256"] = sha256_file(Path(raw_path))
    return profile


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--candidate", type=_candidate, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        target = read_json(args.target)
        validate_config_profile(target, expected_arch="transformer")
        result = select_conformer_ffn_match(target, args.candidate)
        result.update(
            {
                "schema_version": "chapter3-conformer-match/v1",
                "created_at": utc_now(),
                "target_profile_path": str(args.target.resolve()),
                "target_profile_sha256": sha256_file(args.target),
            }
        )
        atomic_write_json(args.output, result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["selected"], indent=2, sort_keys=True))
    print(result["hydra_override"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
