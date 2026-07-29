"""Commit-pinned Git worktrees for reproducible Chapter 3 runs."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .manifest import utc_now


SOURCE_SNAPSHOT_SCHEMA = "chapter3-source-worktree/v1"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


class SourceWorktreeError(RuntimeError):
    """Raised when a pinned source worktree is missing or has changed."""


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and process.returncode:
        raise SourceWorktreeError(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{process.stderr.strip()}"
        )
    return process.stdout.strip()


def default_source_worktree_root(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    return (
        repo_root.parent
        / f"{repo_root.name}_worktrees"
        / "chapter3_distill_only"
    )


def source_integrity_digest(commit: str, tree: str) -> str:
    payload = json.dumps(
        {"commit": commit, "tree": tree},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def repository_snapshot(repo_root: Path, *, require_clean: bool) -> Dict[str, Any]:
    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise SourceWorktreeError(
            "source checkout must be clean before creating a run:\n" + status
        )
    return {
        "repository_root": str(root),
        "commit": commit,
        "tree": tree,
        "integrity_digest": source_integrity_digest(commit, tree),
        "status_porcelain": status.splitlines(),
        "clean": not bool(status),
    }


def _worktree_name(run_identifier: str, run_dir: Path, commit: str) -> str:
    safe_identifier = _SAFE_COMPONENT.sub("-", run_identifier).strip("-._")
    if not safe_identifier:
        safe_identifier = "run"
    run_digest = hashlib.sha256(
        str(run_dir.resolve()).encode("utf-8")
    ).hexdigest()[:8]
    return f"{safe_identifier}-{run_digest}-{commit[:12]}"


def create_source_worktree(
    repo_root: Path,
    worktree_root: Path,
    *,
    run_identifier: str,
    run_dir: Path,
    require_clean: bool = True,
) -> Dict[str, Any]:
    """Create and verify a detached worktree for a new experiment."""

    source = repository_snapshot(repo_root, require_clean=require_clean)
    worktree_root = worktree_root.resolve()
    repository_root = Path(source["repository_root"])
    if worktree_root == repository_root or repository_root in worktree_root.parents:
        raise SourceWorktreeError(
            "source worktree root must be outside the development checkout"
        )
    resolved_run_dir = run_dir.resolve()
    if (
        worktree_root == resolved_run_dir
        or resolved_run_dir in worktree_root.parents
        or worktree_root in resolved_run_dir.parents
    ):
        raise SourceWorktreeError(
            "source worktree root must be outside experiment outputs"
        )
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_root / _worktree_name(
        run_identifier, resolved_run_dir, source["commit"]
    )
    if worktree_path.exists():
        raise SourceWorktreeError(
            f"refusing to reuse an unowned source worktree: {worktree_path}"
        )
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            source["commit"],
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        raise SourceWorktreeError(
            f"cannot create source worktree {worktree_path}: "
            f"{process.stderr.strip()}"
        )
    snapshot = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "repository_root": str(repository_root),
        "worktree_root": str(worktree_root),
        "worktree_path": str(worktree_path.resolve()),
        "commit": source["commit"],
        "tree": source["tree"],
        "integrity_digest": source["integrity_digest"],
        "created_at": utc_now(),
    }
    verify_source_worktree(snapshot)
    return snapshot


def verify_source_worktree(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify commit, tree, digest, and cleanliness of a pinned worktree."""

    if snapshot.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA:
        raise SourceWorktreeError(
            "legacy run has no supported pinned-source metadata; resume it "
            "from its originally recorded commit or start a new run"
        )
    required = (
        "repository_root",
        "worktree_path",
        "commit",
        "tree",
        "integrity_digest",
    )
    missing = [key for key in required if not snapshot.get(key)]
    if missing:
        raise SourceWorktreeError(
            "source-worktree metadata is incomplete: " + ", ".join(missing)
        )
    worktree = Path(str(snapshot["worktree_path"])).resolve()
    if not worktree.is_dir():
        raise SourceWorktreeError(
            f"pinned source worktree is missing: {worktree}; do not bind the "
            "run to current HEAD"
        )
    actual_commit = _git(worktree, "rev-parse", "HEAD")
    actual_tree = _git(worktree, "rev-parse", "HEAD^{tree}")
    expected_commit = str(snapshot["commit"])
    expected_tree = str(snapshot["tree"])
    if actual_commit != expected_commit:
        raise SourceWorktreeError(
            "pinned worktree commit mismatch: "
            f"{actual_commit} != {expected_commit}"
        )
    if actual_tree != expected_tree:
        raise SourceWorktreeError(
            f"pinned worktree tree mismatch: {actual_tree} != {expected_tree}"
        )
    expected_digest = source_integrity_digest(expected_commit, expected_tree)
    if snapshot["integrity_digest"] != expected_digest:
        raise SourceWorktreeError(
            "pinned worktree integrity digest does not match its commit/tree"
        )
    status = _git(
        worktree, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise SourceWorktreeError(
            f"pinned source worktree has unexpected changes at {worktree}:\n"
            + status
        )
    return {
        "verified_at": utc_now(),
        "worktree_path": str(worktree),
        "commit": actual_commit,
        "tree": actual_tree,
        "integrity_digest": expected_digest,
        "clean": True,
    }


def require_manifest_source_snapshot(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    immutable = manifest.get("immutable")
    snapshot = (
        immutable.get("source_snapshot")
        if isinstance(immutable, Mapping)
        else None
    )
    if not isinstance(snapshot, Mapping):
        recorded_commit = None
        if isinstance(immutable, Mapping):
            provenance = immutable.get("provenance")
            if isinstance(provenance, Mapping):
                implementation = provenance.get("implementation")
                if isinstance(implementation, Mapping):
                    recorded_commit = implementation.get("commit")
        suffix = (
            f" Recorded source commit: {recorded_commit}."
            if recorded_commit
            else ""
        )
        raise SourceWorktreeError(
            "legacy manifest has no pinned source worktree and will not be "
            "silently rebound to current HEAD."
            + suffix
            + " Start a new run, or explicitly migrate from the recorded "
            "commit after verifying it is available locally."
        )
    return dict(snapshot)


def render_stage_slurm_script(
    *,
    worktree_path: Path,
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> str:
    """Render a safely quoted stage script rooted in the pinned worktree."""

    worktree = worktree_path.resolve()
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(worktree))}"]
    for key, value in sorted((environment or {}).items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append("exec " + shlex.join(map(str, command)))
    return "\n".join(lines) + "\n"
