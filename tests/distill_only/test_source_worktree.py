"""Offline tests for commit-pinned experiment source worktrees."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from chapter3_distill_only.manifest import ManifestStore
from chapter3_distill_only.source_worktree import (
    SourceWorktreeError,
    create_source_worktree,
    default_source_worktree_root,
    render_stage_slurm_script,
    require_manifest_source_snapshot,
    verify_source_worktree,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "distill_only" / "launch.py"
SPEC = importlib.util.spec_from_file_location("chapter3_worktree_launcher", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), *arguments), text=True
    ).strip()


def _commit(repo: Path, content: str, message: str) -> str:
    (repo / "source.txt").write_text(content, encoding="utf-8")
    subprocess.run(
        ("git", "-C", str(repo), "add", "source.txt"), check=True
    )
    subprocess.run(
        ("git", "-C", str(repo), "commit", "-q", "-m", message), check=True
    )
    return _git(repo, "rev-parse", "HEAD")


class SourceWorktreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        subprocess.run(("git", "init", "-q", str(self.repository)), check=True)
        _git(self.repository, "config", "user.email", "test@example.invalid")
        _git(self.repository, "config", "user.name", "Chapter 3 Test")
        self.first_commit = _commit(
            self.repository, "first\n", "first source commit"
        )
        self.first_tree = _git(self.repository, "rev-parse", "HEAD^{tree}")
        self.worktree_root = self.root / "worktrees"
        self.run_dir = self.root / "outputs" / "experiment" / "seed_1"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self) -> dict:
        return create_source_worktree(
            self.repository,
            self.worktree_root,
            run_identifier="experiment-seed-1",
            run_dir=self.run_dir,
        )

    def test_clean_repository_creates_detached_recorded_worktree(self) -> None:
        snapshot = self.create()
        worktree = Path(snapshot["worktree_path"])
        self.assertEqual(snapshot["commit"], self.first_commit)
        self.assertEqual(snapshot["tree"], self.first_tree)
        self.assertEqual(_git(worktree, "rev-parse", "HEAD"), self.first_commit)
        self.assertEqual(_git(worktree, "branch", "--show-current"), "")
        self.assertTrue(verify_source_worktree(snapshot)["clean"])

        store = ManifestStore(self.run_dir)
        store.create({"source_snapshot": snapshot, "experiment": "test"})
        recorded = store.read()["immutable"]["source_snapshot"]
        self.assertEqual(recorded["commit"], self.first_commit)
        self.assertEqual(recorded["tree"], self.first_tree)
        self.assertEqual(recorded["worktree_path"], str(worktree))

    def test_main_checkout_can_advance_without_invalidating_run(self) -> None:
        snapshot = self.create()
        second_commit = _commit(
            self.repository, "second\n", "continued development"
        )
        self.assertNotEqual(second_commit, snapshot["commit"])
        observation = verify_source_worktree(snapshot)
        self.assertEqual(observation["commit"], self.first_commit)
        self.assertEqual(
            _git(Path(snapshot["worktree_path"]), "rev-parse", "HEAD"),
            self.first_commit,
        )

    def test_tracked_and_untracked_worktree_mutations_are_detected(self) -> None:
        snapshot = self.create()
        worktree = Path(snapshot["worktree_path"])
        tracked = worktree / "source.txt"
        tracked.write_text("mutated\n", encoding="utf-8")
        with self.assertRaisesRegex(
            SourceWorktreeError, "unexpected changes"
        ):
            verify_source_worktree(snapshot)
        tracked.write_text("first\n", encoding="utf-8")
        self.assertTrue(verify_source_worktree(snapshot)["clean"])

        unexpected = worktree / "unexpected.txt"
        unexpected.write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(
            SourceWorktreeError, "unexpected changes"
        ):
            verify_source_worktree(snapshot)
        unexpected.unlink()
        self.assertTrue(verify_source_worktree(snapshot)["clean"])

    def test_resume_reuses_original_and_rejects_commit_mismatch(self) -> None:
        snapshot = self.create()
        store = ManifestStore(self.run_dir)
        store.create(
            {
                "source_snapshot": snapshot,
                "config_digest": "stable-config",
            }
        )
        manifest = store.read()
        resumed = require_manifest_source_snapshot(manifest)
        self.assertEqual(resumed["worktree_path"], snapshot["worktree_path"])
        _commit(self.repository, "new head\n", "new main head")
        launcher._ensure_reusable_run(self.run_dir, "stable-config")
        self.assertEqual(
            verify_source_worktree(resumed)["commit"], self.first_commit
        )

        incompatible = dict(snapshot)
        incompatible["commit"] = _git(self.repository, "rev-parse", "HEAD")
        with self.assertRaisesRegex(
            SourceWorktreeError, "commit mismatch"
        ):
            verify_source_worktree(incompatible)

    def test_stage_runs_from_worktree_and_records_start_and_end(self) -> None:
        snapshot = self.create()
        store = ManifestStore(self.run_dir)
        store.create(
            {
                "source_snapshot": snapshot,
                "experiment": "test",
                "provenance": {
                    "implementation": {"commit": snapshot["commit"]}
                },
            }
        )
        observed_cwd = self.root / "observed_cwd.json"
        command = [
            sys.executable,
            "-c",
            (
                "import json,os,pathlib;"
                f"pathlib.Path({str(observed_cwd)!r}).write_text("
                "json.dumps({'cwd':os.getcwd(),"
                "'pythonpath':os.environ.get('PYTHONPATH'),"
                "'dont_write_bytecode':"
                "os.environ.get('PYTHONDONTWRITEBYTECODE')}))"
            ),
        ]
        launcher.run_command(
            command,
            stage="encoder",
            run_dir=self.run_dir,
            manifest=store,
            experiment="test",
            prepare_runtime=False,
        )
        observed = json.loads(observed_cwd.read_text(encoding="utf-8"))
        self.assertEqual(
            Path(observed["cwd"]).resolve(),
            Path(snapshot["worktree_path"]).resolve(),
        )
        self.assertTrue(
            observed["pythonpath"].startswith(snapshot["worktree_path"])
        )
        self.assertEqual(observed["dont_write_bytecode"], "1")
        stage = store.read()["runtime"]["stage_provenance"]["encoder"]
        self.assertEqual(stage["status"], "complete")
        self.assertEqual(stage["start"]["commit"], self.first_commit)
        self.assertEqual(stage["end"]["tree"], self.first_tree)

    def test_stage_detects_source_mutation_before_accepting_result(self) -> None:
        snapshot = self.create()
        store = ManifestStore(self.run_dir)
        store.create({"source_snapshot": snapshot, "experiment": "test"})
        mutation = Path(snapshot["worktree_path"]) / "unexpected.py"
        command = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(mutation)!r}).write_text('x')",
        ]
        with self.assertRaisesRegex(
            launcher.PreflightError, "integrity failed after encoder"
        ):
            launcher.run_command(
                command,
                stage="encoder",
                run_dir=self.run_dir,
                manifest=store,
                experiment="test",
                prepare_runtime=False,
            )
        stage = store.read()["runtime"]["stage_provenance"]["encoder"]
        self.assertEqual(stage["status"], "source_integrity_failed")
        self.assertFalse(stage["end"]["clean"])

    def test_rendered_slurm_script_uses_pinned_worktree(self) -> None:
        snapshot = self.create()
        worktree = Path(snapshot["worktree_path"])
        pinned_launcher = (
            worktree / "scripts" / "distill_only" / "launch.py"
        )
        script = render_stage_slurm_script(
            worktree_path=worktree,
            command=(sys.executable, str(pinned_launcher), "--stage", "encoder"),
            environment={"PYTHONPATH": str(worktree)},
        )
        self.assertIn(f"cd {worktree}", script)
        self.assertIn(str(pinned_launcher), script)
        self.assertNotIn(str(self.repository / "scripts"), script)

    def test_generated_slurm_scripts_cd_and_invoke_pinned_launcher(self) -> None:
        snapshot = self.create()
        store = ManifestStore(self.run_dir)
        store.create({"source_snapshot": snapshot, "experiment": "test"})
        args = Namespace(
            experiment="c2_t6_historical_heads",
            seed=1337,
            run_dir=self.run_dir,
            output_root=self.root / "outputs",
            source_worktree_root=self.worktree_root,
            teacher=self.root / "teacher.pt",
            data=self.root / "data",
            tokenizer=self.root / "tokenizer.model",
            noise_root=self.root / "noise",
            evaluation_protocol=self.root / "protocol.yaml",
            evaluation_seed=1337,
            final_selection=self.root / "selection.json",
            gpus=1,
            workers=4,
            max_tokens=4000,
            update_freq=4,
            finetune_update_freq=8,
            fairseq_train="fairseq-hydra-train",
            from_selection=None,
            override=[],
        )
        paths = launcher._write_pinned_slurm_scripts(
            args, store, snapshot
        )
        self.assertEqual(
            set(paths), {"encoder", "export", "finetune", "validate"}
        )
        pinned = Path(snapshot["worktree_path"]).resolve()
        for stage, raw_path in paths.items():
            script = Path(raw_path).read_text(encoding="utf-8")
            with self.subTest(stage=stage):
                self.assertIn(f"cd {pinned}", script)
                self.assertIn(
                    str(pinned / "scripts" / "distill_only" / "launch.py"),
                    script,
                )
                self.assertIn("export PYTHONDONTWRITEBYTECODE=1", script)

    def test_dirty_initial_checkout_is_rejected(self) -> None:
        (self.repository / "source.txt").write_text(
            "dirty\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(SourceWorktreeError, "must be clean"):
            self.create()

    def test_legacy_manifest_is_not_rebound(self) -> None:
        legacy = {
            "immutable": {
                "provenance": {
                    "implementation": {"commit": self.first_commit}
                }
            }
        }
        with self.assertRaisesRegex(
            SourceWorktreeError,
            "will not be silently rebound.*Recorded source commit",
        ):
            require_manifest_source_snapshot(legacy)

    def test_default_root_is_beside_repository(self) -> None:
        resolved = self.repository.resolve()
        expected = (
            resolved.parent
            / f"{resolved.name}_worktrees"
            / "chapter3_distill_only"
        )
        self.assertEqual(default_source_worktree_root(self.repository), expected)


if __name__ == "__main__":
    unittest.main()
