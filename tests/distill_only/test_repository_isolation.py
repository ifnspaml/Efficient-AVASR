"""Regression checks for the Chapter 3 distillation-only namespace."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "4502130ce4470fe8b0456fd5b466bef043f383f9"

# Existing joint-DP implementation and entry points that must remain unchanged.
PROTECTED_PATHS = (
    "avhubert/hubert_distill.py",
    "avhubert/hubert_distill_criterion.py",
    "avhubert/prune.py",
    "avhubert/merge.py",
    "avhubert/save_final_ckpt.py",
    "scripts/run_pruning.sh",
    "scripts/run_pruning_merging.sh",
    "scripts/run_merging.sh",
)
ALLOWED_NEW_PREFIXES = (
    "chapter3_distill_only/",
    "avhubert/conf/distill_only/",
    "scripts/distill_only/",
    "tests/distill_only/",
    "docs/chapter3_distill_only_",
)


def _git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT)


class RepositoryIsolationTest(unittest.TestCase):
    def test_baseline_is_an_ancestor(self) -> None:
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"),
            cwd=REPO_ROOT,
            check=True,
        )

    def test_not_on_forbidden_branch(self) -> None:
        branch = _git("branch", "--show-current").decode().strip()
        self.assertNotEqual(branch, "dev_li")

    def test_protected_joint_dp_files_match_baseline(self) -> None:
        for relative_path in PROTECTED_PATHS:
            with self.subTest(path=relative_path):
                baseline = _git("show", f"{BASELINE_COMMIT}:{relative_path}")
                current = (REPO_ROOT / relative_path).read_bytes()
                self.assertEqual(current, baseline)

    def test_existing_joint_dp_configs_match_baseline(self) -> None:
        config_dir = REPO_ROOT / "avhubert" / "conf" / "distill"
        for path in sorted(config_dir.glob("*.yaml")):
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            with self.subTest(path=relative_path):
                baseline = _git("show", f"{BASELINE_COMMIT}:{relative_path}")
                self.assertEqual(path.read_bytes(), baseline)

    def test_branch_changes_are_confined_to_isolated_namespaces(self) -> None:
        committed = set(
            _git("diff", "--name-only", f"{BASELINE_COMMIT}..HEAD")
            .decode()
            .splitlines()
        )
        working = set(
            _git("diff", "--name-only", "HEAD").decode().splitlines()
        )
        untracked = set(
            _git("ls-files", "--others", "--exclude-standard")
            .decode()
            .splitlines()
        )
        for path in sorted(committed | working | untracked):
            with self.subTest(path=path):
                self.assertTrue(
                    path.startswith(ALLOWED_NEW_PREFIXES),
                    f"non-isolated repository change: {path}",
                )


if __name__ == "__main__":
    unittest.main()
