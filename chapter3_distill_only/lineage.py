"""Typed artifact lineage for Chapter 3 derived runs."""

from __future__ import annotations

import fcntl
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from .manifest import (
    MANIFEST_FILENAME,
    RUN_KIND_DERIVED_FINETUNE,
    RUN_KIND_ROOT,
    ManifestError,
    ManifestStore,
    manifest_run_kind,
    sha256_file,
)


class LineageError(ManifestError):
    """Raised when an artifact parent cannot be resolved safely."""


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    checkpoint_path: Path
    checkpoint_sha256: str
    run_dir: Path
    manifest_path: Path
    manifest_sha256: str
    run_kind: str
    state: str
    sequence: int
    original_implementation_commit: Optional[str]

    def parent_record(self) -> Dict[str, Any]:
        if self.kind == "exported_student":
            manifest_key = "parent_manifest_path"
            manifest_hash_key = "parent_manifest_sha256"
            checkpoint_key = "checkpoint_path"
        elif self.kind == "finetune_checkpoint":
            manifest_key = "parent_finetune_manifest"
            manifest_hash_key = "parent_finetune_manifest_sha256"
            checkpoint_key = "checkpoint_path"
        else:  # pragma: no cover - guarded by constructors
            raise LineageError(f"unsupported artifact kind {self.kind!r}")
        return {
            "kind": self.kind,
            "parent_run_dir": str(self.run_dir),
            manifest_key: str(self.manifest_path),
            manifest_hash_key: self.manifest_sha256,
            checkpoint_key: str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "original_implementation_commit": self.original_implementation_commit,
            "parent_manifest_format": (
                "legacy" if self.run_kind == RUN_KIND_ROOT and not _has_snapshot(
                    ManifestStore(self.manifest_path).read()
                ) else "pinned-source"
            ),
        }


_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
def sanitize_label(label: str) -> str:
    sanitized = _LABEL_UNSAFE.sub("-", str(label).strip()).strip("-._")
    if not sanitized:
        raise LineageError("label must contain at least one safe character")
    return sanitized


def _within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _manifest_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.name == MANIFEST_FILENAME:
        return resolved
    if resolved.suffix == ".pt":
        raise LineageError(
            "raw checkpoint paths are not accepted; provide a registered run "
            "directory or manifest"
        )
    if resolved.is_dir() or not resolved.suffix:
        return resolved / MANIFEST_FILENAME
    raise LineageError(f"source is not a run directory or manifest: {resolved}")


def _has_snapshot(manifest: Mapping[str, Any]) -> bool:
    immutable = manifest.get("immutable")
    return isinstance(immutable, Mapping) and isinstance(
        immutable.get("source_snapshot"), Mapping
    )


def _implementation_commit(manifest: Mapping[str, Any]) -> Optional[str]:
    immutable = manifest.get("immutable")
    if not isinstance(immutable, Mapping):
        return None
    provenance = immutable.get("provenance")
    if isinstance(provenance, Mapping):
        implementation = provenance.get("implementation")
        if isinstance(implementation, Mapping) and implementation.get("commit"):
            return str(implementation["commit"])
    snapshot = immutable.get("source_snapshot")
    if isinstance(snapshot, Mapping) and snapshot.get("commit"):
        return str(snapshot["commit"])
    return None


def canonical_root_run(output_root: Path, experiment: str, seed: int) -> Path:
    root = (output_root / experiment / f"seed_{seed}").resolve()
    manifest = root / MANIFEST_FILENAME
    if not manifest.is_file():
        raise LineageError(f"canonical root manifest is missing: {manifest}")
    loaded = ManifestStore(manifest).read()
    if manifest_run_kind(loaded) != RUN_KIND_ROOT:
        raise LineageError(f"canonical experiment root is not a root run: {manifest}")
    return root


def _artifact_record(
    manifest_path: Path,
    *,
    artifact_kind: str,
    require_complete: bool = True,
) -> ArtifactRef:
    manifest_path = _manifest_path(manifest_path)
    store = ManifestStore(manifest_path)
    manifest = store.read()
    run_kind = manifest_run_kind(manifest)
    state = str(manifest["state"])
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LineageError(f"manifest has no artifacts object: {manifest_path}")
    key = (
        "exported_checkpoint"
        if artifact_kind == "exported_student"
        else "finetune_checkpoint"
    )
    artifact = artifacts.get(key)
    if not isinstance(artifact, Mapping):
        raise LineageError(f"manifest has no {key}: {manifest_path}")
    checkpoint = Path(str(artifact.get("path", ""))).resolve()
    if not checkpoint.is_file():
        raise LineageError(f"recorded checkpoint is missing: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    recorded_hash = str(artifact.get("sha256", ""))
    if recorded_hash and recorded_hash != actual_hash:
        raise LineageError(f"recorded checkpoint hash mismatch: {checkpoint}")
    if require_complete:
        valid_states = (
            {"exported", "finetune_running", "finetune_complete",
             "validation_complete", "selection_frozen", "test_complete"}
            if artifact_kind == "exported_student"
            else {"finetune_complete", "validation_complete",
                  "selection_frozen", "test_complete"}
        )
        if state not in valid_states:
            raise LineageError(
                f"{artifact_kind} in {manifest_path} is not complete (state={state})"
            )
    immutable = manifest["immutable"]
    lineage = immutable.get("lineage")
    sequence = (
        int(lineage.get("sequence", 0))
        if isinstance(lineage, Mapping)
        else 0
    )
    return ArtifactRef(
        kind=artifact_kind,
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_hash,
        run_dir=manifest_path.parent.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=sha256_file(manifest_path),
        run_kind=run_kind,
        state=state,
        sequence=sequence,
        original_implementation_commit=_implementation_commit(manifest),
    )


def original_export(root_run: Path) -> ArtifactRef:
    return _artifact_record(
        root_run / MANIFEST_FILENAME,
        artifact_kind="exported_student",
    )


def original_finetune(root_run: Path) -> ArtifactRef:
    return _artifact_record(
        root_run / MANIFEST_FILENAME,
        artifact_kind="finetune_checkpoint",
    )


def derived_finetunes(root_run: Path) -> tuple[ArtifactRef, ...]:
    records = []
    for path in sorted((root_run / "reruns").glob(f"*/{MANIFEST_FILENAME}")):
        try:
            manifest = ManifestStore(path).read()
            if manifest_run_kind(manifest) != RUN_KIND_DERIVED_FINETUNE:
                continue
            records.append(
                _artifact_record(path, artifact_kind="finetune_checkpoint")
            )
        except (ManifestError, OSError) as exc:
            raise LineageError(
                f"cannot inspect derived lineage manifest {path}: {exc}"
            ) from exc
    return tuple(sorted(records, key=lambda item: (item.sequence, str(item.run_dir))))


def compatible_artifacts(
    root_run: Path, artifact_kind: str
) -> tuple[ArtifactRef, ...]:
    if artifact_kind == "exported_student":
        return (original_export(root_run),)
    if artifact_kind != "finetune_checkpoint":
        raise LineageError(f"unsupported artifact role {artifact_kind!r}")
    records = []
    try:
        records.append(original_finetune(root_run))
    except LineageError:
        pass
    records.extend(derived_finetunes(root_run))
    return tuple(records)


def _explicit_artifact(
    root_run: Path, source: str, artifact_kind: str
) -> ArtifactRef:
    manifest_path = _manifest_path(Path(source))
    if not _within(manifest_path, root_run):
        raise LineageError(
            f"source manifest must remain below canonical root {root_run}: "
            f"{manifest_path}"
        )
    return _artifact_record(manifest_path, artifact_kind=artifact_kind)


def resolve_artifact(
    root_run: Path,
    source: Optional[str],
    *,
    artifact_kind: str,
) -> ArtifactRef:
    root_run = root_run.resolve()
    if source in ("original-export", "latest-export"):
        if artifact_kind != "exported_student":
            raise LineageError(f"{source} is not a fine-tuning checkpoint")
        return original_export(root_run)
    if source == "original-finetune":
        if artifact_kind != "finetune_checkpoint":
            raise LineageError("original-finetune is not an exported student")
        return original_finetune(root_run)
    if source == "latest-finetune":
        if artifact_kind != "finetune_checkpoint":
            raise LineageError("latest-finetune is not an exported student")
        candidates = compatible_artifacts(root_run, artifact_kind)
        if not candidates:
            raise LineageError("no completed fine-tuning artifact exists")
        return max(candidates, key=lambda item: (item.sequence, str(item.run_dir)))
    if source:
        return _explicit_artifact(root_run, source, artifact_kind)
    candidates = compatible_artifacts(root_run, artifact_kind)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise LineageError(f"no compatible {artifact_kind} artifact exists")
    listing = "\n".join(
        f"{index}. {item.run_dir}"
        for index, item in enumerate(candidates, start=1)
    )
    raise LineageError(
        f"Multiple compatible {artifact_kind} artifacts exist:\n{listing}\n"
        "Specify --derive-from explicitly."
    )


@contextmanager
def _allocation_lock(rerun_root: Path) -> Iterator[None]:
    rerun_root.mkdir(parents=True, exist_ok=True)
    lock = rerun_root / ".allocation.lock"
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def proposed_derived_run(
    root_run: Path,
    *,
    run_label: str,
    commit: str,
    existing_names: Optional[Sequence[str]] = None,
) -> tuple[Path, int]:
    safe_label = sanitize_label(run_label)
    short_commit = commit[:7]
    rerun_root = (root_run / "reruns").resolve()
    if not _within(rerun_root, root_run):
        raise LineageError(
            f"rerun root escapes canonical experiment run: {rerun_root}"
        )
    names = set(
        existing_names
        if existing_names is not None
        else (
            path.name
            for path in (root_run / "reruns").iterdir()
            if path.is_dir()
        )
    ) if (existing_names is not None or (root_run / "reruns").is_dir()) else set()
    sequence = 1
    while f"{safe_label}_{short_commit}_{sequence:03d}" in names:
        sequence += 1
    path = (rerun_root / f"{safe_label}_{short_commit}_{sequence:03d}").resolve()
    if not _within(path, rerun_root):
        raise LineageError(f"derived run escapes rerun root: {path}")
    return path, sequence


def allocate_derived_run(
    root_run: Path,
    *,
    run_label: str,
    commit: str,
) -> tuple[Path, int]:
    rerun_root = (root_run / "reruns").resolve()
    if not _within(rerun_root, root_run):
        raise LineageError(
            f"rerun root escapes canonical experiment run: {rerun_root}"
        )
    with _allocation_lock(rerun_root):
        path, sequence = proposed_derived_run(
            root_run,
            run_label=run_label,
            commit=commit,
        )
        if path.exists():
            raise LineageError(f"refusing to reuse derived run directory: {path}")
        path.mkdir(parents=False, exist_ok=False)
    return path, sequence


def evaluation_directory(finetune_run: Path, label: str) -> Path:
    root = (finetune_run / "evaluations").resolve()
    if not _within(root, finetune_run):
        raise LineageError(
            f"evaluation root escapes its fine-tuning run: {root}"
        )
    path = (root / sanitize_label(label)).resolve()
    if not _within(path, root) or path == root:
        raise LineageError(f"evaluation path escapes its fine-tuning run: {path}")
    return path
