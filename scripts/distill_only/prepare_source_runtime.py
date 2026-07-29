#!/usr/bin/env python3
"""Prepare Fairseq binary extensions for an existing pinned Chapter 3 run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.manifest import (  # noqa: E402
    ManifestStore,
    sha256_file,
    utc_now,
)
from chapter3_distill_only.source_worktree import (  # noqa: E402
    SourceWorktreeError,
    prepare_fairseq_runtime,
    repository_snapshot,
    require_manifest_source_snapshot,
    verify_source_worktree,
)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_existing_run(
    run_dir: Path,
    *,
    python_executable: Path,
    check_only: bool = False,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    repair_source = repository_snapshot(REPO_ROOT, require_clean=True)
    store = ManifestStore(run_dir)
    manifest = store.read()
    original_manifest_hash = sha256_file(store.path)
    snapshot = require_manifest_source_snapshot(manifest)
    prepared = prepare_fairseq_runtime(
        snapshot,
        python_executable,
        runtime_root=run_dir / "source" / "runtime" / "fairseq",
        build_if_missing=not check_only,
    )
    observation = verify_source_worktree(prepared)
    result = {
        "schema_version": "chapter3-source-runtime-repair/v1",
        "prepared_at": utc_now(),
        "run_dir": str(run_dir),
        "manifest_path": str(store.path),
        "manifest_sha256": original_manifest_hash,
        "manifest_unchanged": sha256_file(store.path) == original_manifest_hash,
        "repair_implementation": {
            **repair_source,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_commit": prepared["commit"],
        "source_tree": prepared["tree"],
        "worktree_path": prepared["worktree_path"],
        "runtime_artifacts": prepared["runtime_artifacts"],
        "integrity_observation": observation,
    }
    if not check_only:
        audit_path = (
            run_dir / "source" / "runtime" / "fairseq_runtime.v1.json"
        )
        result["audit_path"] = str(audit_path)
        _atomic_json(audit_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter whose ABI will run Fairseq",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify an already importable runtime without compiling or writing",
    )
    args = parser.parse_args()
    try:
        result = prepare_existing_run(
            args.run_dir,
            python_executable=args.python,
            check_only=args.check_only,
        )
    except (SourceWorktreeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
