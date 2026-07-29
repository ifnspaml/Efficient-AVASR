"""Versioned, atomic result manifests for Chapter 3 experiments.

The training launcher is intentionally the only writer of ``manifest.v1.json``.
This module keeps the file format small and dependency free so that manifests
can also be inspected from login nodes without importing PyTorch or Fairseq.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional


SCHEMA_VERSION = "chapter3-distill-manifest/v1"
MANIFEST_FILENAME = "manifest.v1.json"
LIFECYCLE = (
    "prepared",
    "encoder_running",
    "encoder_complete",
    "exported",
    "finetune_running",
    "finetune_complete",
    "validation_complete",
    "selection_frozen",
    "test_complete",
)
RUN_KIND_ROOT = "root_pipeline"
RUN_KIND_DERIVED_FINETUNE = "derived_finetune"
RUN_KIND_EVALUATION = "evaluation"
RUN_KINDS = (
    RUN_KIND_ROOT,
    RUN_KIND_DERIVED_FINETUNE,
    RUN_KIND_EVALUATION,
)
RUN_KIND_LIFECYCLES = {
    RUN_KIND_ROOT: LIFECYCLE,
    RUN_KIND_DERIVED_FINETUNE: (
        "exported",
        "finetune_running",
        "finetune_complete",
    ),
    RUN_KIND_EVALUATION: (
        "prepared",
        "evaluation_running",
        "evaluation_complete",
    ),
}


def manifest_run_kind(manifest: Mapping[str, Any]) -> str:
    """Return the immutable run kind, treating old manifests as root runs."""

    immutable = manifest.get("immutable")
    if not isinstance(immutable, Mapping):
        return RUN_KIND_ROOT
    return str(immutable.get("run_kind", RUN_KIND_ROOT))


def lifecycle_for_kind(run_kind: str) -> tuple[str, ...]:
    try:
        return RUN_KIND_LIFECYCLES[run_kind]
    except KeyError as exc:
        raise ManifestError(f"invalid manifest run kind {run_kind!r}") from exc


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, inconsistent, or mutated unsafely."""


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    """Serialize *value* deterministically for content hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    """Atomically replace *path* with a durable JSON representation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON file {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"expected a JSON object in {source}")
    return value


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(f"git {' '.join(arguments)} failed in {repo}: {message}")
    return process.stdout.decode("utf-8", errors="replace").strip()


def git_provenance(repo: os.PathLike[str] | str) -> Dict[str, Any]:
    """Capture commit, branch, remote, status, and a hash of local changes."""

    root = Path(repo).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD", "--binary", "--no-ext-diff"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if diff.returncode:
        raise ManifestError(
            f"cannot hash local changes in {root}: "
            f"{diff.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return {
        "path": str(root),
        "branch": _git(root, "branch", "--show-current"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "remote": _git(root, "remote", "get-url", "origin", check=False) or None,
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(diff.stdout),
        "dirty_state_sha256": sha256_bytes(
            status.encode("utf-8") + b"\0" + diff.stdout
        ),
        "dirty": bool(status),
    }


def hash_files(paths: Iterable[os.PathLike[str] | str]) -> Dict[str, str]:
    """Return stable absolute-path-to-SHA256 mappings for existing files."""

    output: Dict[str, str] = {}
    for raw_path in sorted({str(Path(path).resolve()) for path in paths}):
        path = Path(raw_path)
        if path.is_file():
            output[raw_path] = sha256_file(path)
    return output


def build_provenance(
    repo_root: os.PathLike[str] | str,
    *,
    baseline_branch: str,
    baseline_commit: str,
    old_repo: os.PathLike[str] | str,
    archived_configs: Iterable[os.PathLike[str] | str] = (),
) -> Dict[str, Any]:
    """Build the immutable Git/reference portion of a run manifest."""

    return {
        "baseline": {
            "branch": baseline_branch,
            "commit": baseline_commit,
        },
        "implementation": git_provenance(repo_root),
        "historical_reference": {
            **git_provenance(old_repo),
            "archived_config_sha256": hash_files(archived_configs),
        },
    }


def _deep_merge(target: MutableMapping[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if (
            key in target
            and isinstance(target[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


class ManifestStore:
    """Concurrency-safe reader/writer for one run manifest."""

    def __init__(self, path: os.PathLike[str] | str):
        candidate = Path(path)
        self.path = (
            candidate
            if candidate.name == MANIFEST_FILENAME or candidate.suffix == ".json"
            else candidate / MANIFEST_FILENAME
        )
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> Dict[str, Any]:
        manifest = read_json(self.path)
        self.validate(manifest)
        return manifest

    @staticmethod
    def validate(manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported manifest schema {manifest.get('schema_version')!r}"
            )
        if not isinstance(manifest.get("immutable"), Mapping):
            raise ManifestError("manifest immutable section must be an object")
        run_kind = manifest_run_kind(manifest)
        lifecycle = lifecycle_for_kind(run_kind)
        if manifest.get("state") not in lifecycle:
            raise ManifestError(
                f"invalid {run_kind} lifecycle state {manifest.get('state')!r}"
            )
        expected_digest = sha256_json(manifest["immutable"])
        if manifest.get("immutable_sha256") != expected_digest:
            raise ManifestError("manifest immutable section digest mismatch")

    def create(
        self,
        immutable: Mapping[str, Any],
        *,
        runtime: Optional[Mapping[str, Any]] = None,
        initial_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a manifest at its first state, or return an identical one."""

        immutable_copy = copy.deepcopy(dict(immutable))
        run_kind = str(immutable_copy.get("run_kind", RUN_KIND_ROOT))
        lifecycle = lifecycle_for_kind(run_kind)
        state = initial_state or lifecycle[0]
        if state != lifecycle[0]:
            raise ManifestError(
                f"{run_kind} manifests must start at {lifecycle[0]!r}, got {state!r}"
            )
        now = utc_now()
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "created_at": now,
            "updated_at": now,
            "immutable": immutable_copy,
            "immutable_sha256": sha256_json(immutable_copy),
            "runtime": copy.deepcopy(dict(runtime or {})),
            "measurements": {},
            "artifacts": {},
            "failure": None,
            "history": [{"state": state, "timestamp": now}],
        }
        with self._lock():
            if self.path.exists():
                current = self.read()
                if current["immutable_sha256"] != value["immutable_sha256"]:
                    raise ManifestError(
                        f"{self.path} already belongs to a different immutable run"
                    )
                return current
            atomic_write_json(self.path, value)
        return value

    def update(
        self,
        values: Mapping[str, Any],
        *,
        expected_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deep-merge mutable sections without changing lifecycle or immutable data."""

        forbidden = {"schema_version", "immutable", "immutable_sha256", "state", "history"}
        overlap = forbidden.intersection(values)
        if overlap:
            raise ManifestError(f"update cannot modify protected fields: {sorted(overlap)}")
        with self._lock():
            current = self.read()
            if expected_state is not None and current["state"] != expected_state:
                raise ManifestError(
                    f"expected state {expected_state}, found {current['state']}"
                )
            _deep_merge(current, values)
            current["updated_at"] = utc_now()
            atomic_write_json(self.path, current)
        return current

    def transition(
        self,
        new_state: str,
        *,
        values: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Move exactly one step through the declared lifecycle.

        Repeating the current state is idempotent, which makes interrupted wrapper
        scripts safe to retry without skipping a stage.
        """

        with self._lock():
            current = self.read()
            run_kind = manifest_run_kind(current)
            lifecycle = lifecycle_for_kind(run_kind)
            if new_state not in lifecycle:
                raise ManifestError(
                    f"invalid {run_kind} lifecycle state {new_state!r}"
                )
            next_state = {
                before: after
                for before, after in zip(lifecycle, lifecycle[1:])
            }
            old_state = current["state"]
            if old_state != new_state and next_state.get(old_state) != new_state:
                raise ManifestError(
                    f"invalid lifecycle transition {old_state!r} -> {new_state!r}"
                )
            if values:
                protected = {"schema_version", "immutable", "immutable_sha256", "state", "history"}
                overlap = protected.intersection(values)
                if overlap:
                    raise ManifestError(
                        f"transition values cannot modify protected fields: {sorted(overlap)}"
                    )
                _deep_merge(current, values)
            now = utc_now()
            if old_state != new_state:
                current["state"] = new_state
                current["history"].append({"state": new_state, "timestamp": now})
            current["updated_at"] = now
            atomic_write_json(self.path, current)
        return current

    def record_failure(
        self,
        *,
        stage: str,
        message: str,
        returncode: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.update(
            {
                "failure": {
                    "stage": stage,
                    "message": message,
                    "returncode": returncode,
                    "timestamp": utc_now(),
                }
            }
        )
