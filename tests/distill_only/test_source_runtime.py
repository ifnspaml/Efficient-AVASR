"""Offline tests for pinned Fairseq binary runtime preparation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chapter3_distill_only.manifest import ManifestStore
from chapter3_distill_only import source_worktree
from chapter3_distill_only.source_worktree import (
    SourceWorktreeError,
    create_source_worktree,
    prepare_fairseq_runtime,
    verify_source_worktree,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_PATH = (
    REPO_ROOT / "scripts" / "distill_only" / "prepare_source_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("chapter3_runtime_repair", REPAIR_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.stdout.strip()


class SourceRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        _git(self.repository, "init")
        _git(self.repository, "config", "user.email", "runtime@example.test")
        _git(self.repository, "config", "user.name", "Runtime Test")
        (self.repository / ".gitignore").write_text(
            "*.so\n*.cpp\nbuild/\n", encoding="utf-8"
        )
        setup = self.repository / "fairseq" / "setup.py"
        setup.parent.mkdir(parents=True)
        setup.write_text("# test setup\n", encoding="utf-8")
        (self.repository / "source.txt").write_text("source\n", encoding="utf-8")
        _git(self.repository, "add", ".")
        _git(self.repository, "commit", "-m", "runtime source")
        self.run_dir = self.root / "outputs" / "run"
        self.snapshot = create_source_worktree(
            self.repository,
            self.root / "worktrees",
            run_identifier="runtime-test",
            run_dir=self.run_dir,
        )
        self.worktree = Path(self.snapshot["worktree_path"])
        self.runtime_root = self.run_dir / "source" / "runtime" / "fairseq"
        self.modules = {
            "fairseq.data.data_utils_fast": (
                self.runtime_root
                / "fairseq"
                / "data"
                / "data_utils_fast.test.so"
            ),
            "fairseq.data.token_block_utils_fast": (
                self.runtime_root
                / "fairseq"
                / "data"
                / "token_block_utils_fast.test.so"
            ),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _probe(
        self, worktree: Path, interpreter: Path, runtime_root: Path = None
    ):
        if not all(path.is_file() for path in self.modules.values()):
            return False, "missing test extension"
        return True, {
            "python_executable": str(interpreter.resolve()),
            "python_version": "3.test",
            "soabi": "test-soabi",
            "platform": "test-platform",
            "modules": {
                module: str(path.resolve())
                for module, path in self.modules.items()
            },
        }

    def _build(
        self, worktree: Path, interpreter: Path, runtime_root: Path
    ):
        self.assertEqual(worktree, self.worktree)
        self.assertEqual(runtime_root, self.runtime_root.resolve())
        for index, path in enumerate(self.modules.values()):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"extension-{index}".encode("ascii"))
        return [str(interpreter), "setup.py", "build_ext", "--inplace"], "built"

    def _prepare(self):
        with mock.patch.object(
            source_worktree, "_probe_fairseq_runtime", side_effect=self._probe
        ), mock.patch.object(
            source_worktree, "_build_fairseq_runtime", side_effect=self._build
        ):
            return prepare_fairseq_runtime(
                self.snapshot,
                Path(sys.executable),
                runtime_root=self.runtime_root,
                build_if_missing=True,
            )

    def test_builds_records_and_verifies_required_extensions(self) -> None:
        prepared = self._prepare()
        runtime = prepared["runtime_artifacts"]
        self.assertEqual(runtime["soabi"], "test-soabi")
        self.assertEqual(
            {item["module"] for item in runtime["artifacts"]},
            {
                "fairseq.data.data_utils_fast",
                "fairseq.data.token_block_utils_fast",
                "sitecustomize",
            },
        )
        self.assertTrue(all(item["sha256"] for item in runtime["artifacts"]))
        observation = verify_source_worktree(prepared)
        self.assertEqual(
            observation["runtime_artifacts"]["soabi"], "test-soabi"
        )
        self.assertEqual(
            _git(self.worktree, "status", "--porcelain=v1"), ""
        )

    def test_runtime_artifact_mutation_and_removal_are_detected(self) -> None:
        prepared = self._prepare()
        artifact = next(iter(self.modules.values()))
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(SourceWorktreeError, "hash mismatch"):
            verify_source_worktree(prepared)
        artifact.unlink()
        with self.assertRaisesRegex(SourceWorktreeError, "is missing"):
            verify_source_worktree(prepared)

    def test_missing_runtime_is_rejected_when_build_is_disabled(self) -> None:
        with mock.patch.object(
            source_worktree, "_probe_fairseq_runtime", side_effect=self._probe
        ):
            with self.assertRaisesRegex(
                SourceWorktreeError, "runtime is unavailable"
            ):
                prepare_fairseq_runtime(
                    self.snapshot,
                    Path(sys.executable),
                    runtime_root=self.runtime_root,
                    build_if_missing=False,
                )

    def test_recorded_runtime_rejects_another_interpreter(self) -> None:
        prepared = self._prepare()
        mismatched = dict(prepared)
        runtime = dict(prepared["runtime_artifacts"])
        runtime["python_executable"] = "/different/python"
        mismatched["runtime_artifacts"] = runtime
        with mock.patch.object(
            source_worktree, "_probe_fairseq_runtime", side_effect=self._probe
        ):
            with self.assertRaisesRegex(
                SourceWorktreeError, "interpreter mismatch"
            ):
                prepare_fairseq_runtime(
                    mismatched, Path(sys.executable), build_if_missing=True
                )

    def test_repair_tool_preserves_manifest_and_writes_audit(self) -> None:
        store = ManifestStore(self.run_dir)
        store.create(
            {
                "experiment": "runtime-test",
                "source_snapshot": self.snapshot,
            }
        )
        original = store.path.read_bytes()
        with mock.patch.object(
            source_worktree, "_probe_fairseq_runtime", side_effect=self._probe
        ), mock.patch.object(
            source_worktree, "_build_fairseq_runtime", side_effect=self._build
        ), mock.patch.object(
            repair,
            "repository_snapshot",
            return_value={
                "repository_root": str(REPO_ROOT),
                "commit": "repair-commit",
                "tree": "repair-tree",
                "integrity_digest": "repair-digest",
                "status_porcelain": [],
                "clean": True,
            },
        ):
            result = repair.prepare_existing_run(
                self.run_dir,
                python_executable=Path(sys.executable),
            )
        self.assertEqual(store.path.read_bytes(), original)
        self.assertTrue(result["manifest_unchanged"])
        audit = Path(result["audit_path"])
        self.assertTrue(audit.is_file())
        recorded = json.loads(audit.read_text(encoding="utf-8"))
        self.assertEqual(recorded["source_commit"], self.snapshot["commit"])


if __name__ == "__main__":
    unittest.main()
