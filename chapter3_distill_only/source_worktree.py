"""Commit-pinned Git worktrees for reproducible Chapter 3 runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .manifest import utc_now


SOURCE_SNAPSHOT_SCHEMA = "chapter3-source-worktree/v1"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_RUNTIME_SCHEMA = "chapter3-fairseq-runtime/v1"
_RUNTIME_MODULES = (
    "fairseq.data.data_utils_fast",
    "fairseq.data.token_block_utils_fast",
)
_RUNTIME_PROBE_MARKER = "CHAPTER3_FAIRSEQ_RUNTIME="


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_runtime_path(runtime_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(runtime_root))
    except ValueError as exc:
        raise SourceWorktreeError(
            f"Fairseq runtime artifact escapes runtime root: {resolved}"
        ) from exc


def _runtime_environment(
    worktree: Path, runtime_root: Optional[Path] = None
) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    pythonpath = []
    if runtime_root is not None:
        pythonpath.append(str(runtime_root))
    pythonpath.extend((str(worktree), str(worktree / "fairseq")))
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    environment.setdefault("MAX_JOBS", "4")
    return environment


def _probe_fairseq_runtime(
    worktree: Path,
    python_executable: Path,
    runtime_root: Optional[Path] = None,
) -> Tuple[bool, Union[Dict[str, Any], str]]:
    modules = json.dumps(_RUNTIME_MODULES)
    code = (
        "import importlib,json,pathlib,platform,sys,sysconfig;"
        f"names={modules};"
        "loaded={name:str(pathlib.Path(importlib.import_module(name).__file__).resolve())"
        " for name in names};"
        "payload={'python_executable':str(pathlib.Path(sys.executable).resolve()),"
        "'python_version':platform.python_version(),"
        "'soabi':sysconfig.get_config_var('SOABI'),"
        "'platform':platform.platform(),"
        "'modules':loaded};"
        f"print({_RUNTIME_PROBE_MARKER!r}+json.dumps(payload,sort_keys=True))"
    )
    process = subprocess.run(
        [str(python_executable), "-B", "-c", code],
        cwd=str(worktree),
        env=_runtime_environment(worktree, runtime_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        return False, detail[-4000:]
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(_RUNTIME_PROBE_MARKER):
            try:
                payload = json.loads(line[len(_RUNTIME_PROBE_MARKER) :])
            except json.JSONDecodeError as exc:
                return False, f"invalid Fairseq runtime probe output: {exc}"
            return True, payload
    return False, "Fairseq runtime probe produced no metadata marker"


def _verify_runtime_artifacts(
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    if runtime.get("schema_version") != _RUNTIME_SCHEMA:
        raise SourceWorktreeError(
            "unsupported pinned Fairseq runtime metadata schema"
        )
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SourceWorktreeError(
            "pinned Fairseq runtime metadata has no artifacts"
        )
    raw_root = runtime.get("pythonpath_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise SourceWorktreeError(
            "pinned Fairseq runtime metadata has no Python-path root"
        )
    runtime_root = Path(raw_root).resolve()
    if not runtime_root.is_dir():
        raise SourceWorktreeError(
            f"pinned Fairseq runtime root is missing: {runtime_root}"
        )
    verified = []
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise SourceWorktreeError(
                "malformed pinned Fairseq runtime artifact record"
            )
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise SourceWorktreeError(
                "pinned Fairseq runtime artifact has no relative path"
            )
        artifact = (runtime_root / relative).resolve()
        try:
            artifact.relative_to(runtime_root)
        except ValueError as exc:
            raise SourceWorktreeError(
                f"pinned Fairseq runtime artifact escapes runtime root: {relative}"
            ) from exc
        if not artifact.is_file():
            raise SourceWorktreeError(
                f"pinned Fairseq runtime artifact is missing: {artifact}"
            )
        actual_hash = _sha256_file(artifact)
        if actual_hash != expected_hash:
            raise SourceWorktreeError(
                "pinned Fairseq runtime artifact hash mismatch: "
                f"{artifact} ({actual_hash} != {expected_hash})"
            )
        verified.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "size_bytes": artifact.stat().st_size,
            }
        )
    return {
        "schema_version": _RUNTIME_SCHEMA,
        "python_executable": runtime.get("python_executable"),
        "python_version": runtime.get("python_version"),
        "soabi": runtime.get("soabi"),
        "pythonpath_root": str(runtime_root),
        "artifacts": verified,
    }


def _build_fairseq_runtime(
    worktree: Path, interpreter: Path, runtime_root: Path
) -> Tuple[Sequence[str], str]:
    fairseq_source = worktree / "fairseq"
    if not (
        fairseq_source / "fairseq" / "data" / "data_utils_fast.pyx"
    ).is_file():
        raise SourceWorktreeError(
            f"pinned Fairseq Cython sources are missing: {fairseq_source}"
        )
    builder = Path(__file__).resolve().with_name("build_fairseq_runtime.py")
    if not builder.is_file():
        raise SourceWorktreeError(
            f"Fairseq runtime builder is missing: {builder}"
        )
    command = [
        str(interpreter),
        str(builder),
        "--source-root",
        str(fairseq_source),
        "--output-root",
        str(runtime_root),
    ]
    process = subprocess.run(
        command,
        cwd=str(worktree),
        env=_runtime_environment(worktree),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = process.stdout[-8000:]
    if process.returncode:
        raise SourceWorktreeError(
            "failed to build pinned Fairseq Cython extensions "
            f"(return code {process.returncode}):\n{output}"
        )
    return command, output


def prepare_fairseq_runtime(
    snapshot: Mapping[str, Any],
    python_executable: Path,
    *,
    runtime_root: Optional[Path] = None,
    build_if_missing: bool = True,
) -> Dict[str, Any]:
    """Build, import-check, and record Fairseq extensions for a pinned source."""

    prepared = dict(snapshot)
    verify_source_worktree(prepared)
    worktree = Path(str(prepared["worktree_path"])).resolve()
    interpreter = python_executable.resolve()
    existing_runtime = prepared.get("runtime_artifacts")
    if isinstance(existing_runtime, Mapping):
        _verify_runtime_artifacts(existing_runtime)
        recorded_root = Path(str(existing_runtime["pythonpath_root"])).resolve()
        success, probe = _probe_fairseq_runtime(
            worktree, interpreter, recorded_root
        )
        if not success:
            raise SourceWorktreeError(
                "recorded pinned Fairseq runtime cannot be imported with "
                f"{interpreter}: {probe}"
            )
        assert isinstance(probe, Mapping)
        if probe.get("python_executable") != existing_runtime.get(
            "python_executable"
        ):
            raise SourceWorktreeError(
                "pinned Fairseq runtime interpreter mismatch: "
                f"{probe.get('python_executable')} != "
                f"{existing_runtime.get('python_executable')}"
            )
        if probe.get("soabi") != existing_runtime.get("soabi"):
            raise SourceWorktreeError(
                "pinned Fairseq runtime Python ABI mismatch: "
                f"{probe.get('soabi')} != {existing_runtime.get('soabi')}"
            )
        return prepared

    if runtime_root is None:
        raise SourceWorktreeError(
            "a writable runtime root is required for an unprepared source "
            "snapshot"
        )
    runtime_root = runtime_root.resolve()
    if runtime_root == worktree or worktree in runtime_root.parents:
        raise SourceWorktreeError(
            "Fairseq runtime root must be outside the read-only source worktree"
        )
    runtime_root.mkdir(parents=True, exist_ok=True)
    sitecustomize = runtime_root / "sitecustomize.py"
    sitecustomize.write_text(
        "import importlib.abc\n"
        "import importlib.machinery\n"
        "import importlib.util\n"
        "from pathlib import Path\n"
        "_ROOT = Path(__file__).resolve().parent / 'fairseq' / 'data'\n"
        "_NAMES = {\n"
        "    'fairseq.data.data_utils_fast',\n"
        "    'fairseq.data.token_block_utils_fast',\n"
        "}\n"
        "class _Chapter3FairseqRuntime(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname not in _NAMES:\n"
        "            return None\n"
        "        stem = fullname.rsplit('.', 1)[-1]\n"
        "        for suffix in importlib.machinery.EXTENSION_SUFFIXES:\n"
        "            candidate = _ROOT / (stem + suffix)\n"
        "            if candidate.is_file():\n"
        "                return importlib.util.spec_from_file_location(\n"
        "                    fullname, str(candidate)\n"
        "                )\n"
        "        return None\n"
        "import sys\n"
        "sys.meta_path.insert(0, _Chapter3FairseqRuntime())\n",
        encoding="utf-8",
    )
    success, probe = _probe_fairseq_runtime(
        worktree, interpreter, runtime_root
    )
    build_command = None
    if not success:
        if not build_if_missing:
            raise SourceWorktreeError(
                f"pinned Fairseq runtime is unavailable: {probe}"
            )
        build_command, _ = _build_fairseq_runtime(
            worktree, interpreter, runtime_root
        )
        success, probe = _probe_fairseq_runtime(
            worktree, interpreter, runtime_root
        )
        if not success:
            raise SourceWorktreeError(
                "Fairseq extensions built but import verification failed: "
                f"{probe}"
            )
    assert isinstance(probe, Mapping)
    module_paths = probe.get("modules")
    if not isinstance(module_paths, Mapping):
        raise SourceWorktreeError(
            "Fairseq runtime probe did not return module paths"
        )
    artifacts = []
    for module in _RUNTIME_MODULES:
        raw_path = module_paths.get(module)
        if not isinstance(raw_path, str):
            raise SourceWorktreeError(
                f"Fairseq runtime probe omitted required module {module}"
            )
        artifact = Path(raw_path).resolve()
        relative = _relative_runtime_path(runtime_root, artifact)
        artifacts.append(
            {
                "module": module,
                "path": relative,
                "sha256": _sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        )
    artifacts.append(
        {
            "module": "sitecustomize",
            "path": _relative_runtime_path(runtime_root, sitecustomize),
            "sha256": _sha256_file(sitecustomize),
            "size_bytes": sitecustomize.stat().st_size,
        }
    )
    prepared["runtime_artifacts"] = {
        "schema_version": _RUNTIME_SCHEMA,
        "prepared_at": utc_now(),
        "python_executable": probe.get("python_executable"),
        "python_version": probe.get("python_version"),
        "soabi": probe.get("soabi"),
        "platform": probe.get("platform"),
        "pythonpath_root": str(runtime_root),
        "build_command": build_command,
        "artifacts": artifacts,
    }
    verify_source_worktree(prepared)
    return prepared


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
    observation = {
        "verified_at": utc_now(),
        "worktree_path": str(worktree),
        "commit": actual_commit,
        "tree": actual_tree,
        "integrity_digest": expected_digest,
        "clean": True,
    }
    runtime = snapshot.get("runtime_artifacts")
    if runtime is not None:
        if not isinstance(runtime, Mapping):
            raise SourceWorktreeError(
                "malformed pinned Fairseq runtime metadata"
            )
        observation["runtime_artifacts"] = _verify_runtime_artifacts(
            runtime
        )
    return observation


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
