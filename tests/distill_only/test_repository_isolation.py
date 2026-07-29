"""Regression checks for protected DP code and removed infrastructure."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "4502130ce4470fe8b0456fd5b466bef043f383f9"
PROTECTED = (
    "avhubert/hubert_distill.py",
    "avhubert/hubert_distill_criterion.py",
    "avhubert/prune.py",
    "avhubert/merge.py",
    "avhubert/save_final_ckpt.py",
    "scripts/run_pruning.sh",
    "scripts/run_pruning_merging.sh",
    "scripts/run_merging.sh",
)
REMOVED_MODULES = (
    "build_fairseq_runtime.py",
    "checkpoint_audit.py",
    "evaluation.py",
    "evaluation_runs.py",
    "lineage.py",
    "manifest.py",
    "profiling.py",
    "selection.py",
    "source_worktree.py",
)


def git(*arguments: str) -> bytes:
    return subprocess.check_output(("git", *arguments), cwd=ROOT)


class RepositoryIsolationTest(unittest.TestCase):
    def test_baseline_ancestor_and_branch(self) -> None:
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", BASELINE, "HEAD"),
            cwd=ROOT,
            check=True,
        )
        self.assertEqual(git("branch", "--show-current").decode().strip(), "ch3-distill-only-v2")

    def test_protected_joint_dp_files_and_configs_match_baseline(self) -> None:
        paths = list(PROTECTED)
        paths.extend(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "avhubert" / "conf" / "distill").glob("*.yaml")
        )
        for relative in paths:
            with self.subTest(path=relative):
                self.assertEqual(
                    (ROOT / relative).read_bytes(),
                    git("show", f"{BASELINE}:{relative}"),
                )

    def test_noise_utils_and_gitignore_are_restored_to_baseline(self) -> None:
        for relative in (".gitignore", "avhubert/noise_utils.py"):
            self.assertEqual(
                (ROOT / relative).read_bytes(),
                git("show", f"{BASELINE}:{relative}"),
            )

    def test_legacy_modules_and_launchers_are_absent(self) -> None:
        package = ROOT / "chapter3_distill_only"
        for name in REMOVED_MODULES:
            self.assertFalse((package / name).exists(), name)
        for relative in (
            "scripts/distill_only/launch.py",
            "scripts/distill_only/launch.sh",
            "scripts/distill_only/record_provenance.py",
            "scripts/distill_only/run_evaluation.py",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_retained_runtime_has_no_dead_imports(self) -> None:
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "chapter3_distill_only").glob("*.py")
        )
        for stem in (Path(name).stem for name in REMOVED_MODULES):
            self.assertNotIn(f".{stem} import", text)


if __name__ == "__main__":
    unittest.main()
