"""Regression checks for protected DP code and removed infrastructure."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "4502130ce4470fe8b0456fd5b466bef043f383f9"
PROTECTED = (
    "avhubert/hubert_distill_criterion.py",
    "avhubert/prune.py",
    "avhubert/merge.py",
    "avhubert/save_final_ckpt.py",
    "scripts/run_pruning.sh",
    "scripts/run_pruning_merging.sh",
    "scripts/run_merging.sh",
)
V3_BASELINE = "38a22cec4c7000cba0de0feb6863ac0fc20cc5c9"
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
REPLACED_STAGE_J_STUBS = {
    f"avhubert/conf/distill_v3/joint_dp_stage{stage}/{name}.yaml"
    for stage in (1, 2)
    for name in (
        "j1_transformer_tau65",
        "j2_hybrid_tau70",
        "j3_hybrid_tau80",
    )
}


def git(*arguments: str) -> bytes:
    return subprocess.check_output(("git", *arguments), cwd=ROOT)


class RepositoryIsolationTest(unittest.TestCase):
    def test_baseline_ancestor_and_branch(self) -> None:
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", BASELINE, "HEAD"),
            cwd=ROOT,
            check=True,
        )
        self.assertEqual(git("branch", "--show-current").decode().strip(), "ch3-distill-only-v3")

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

    def test_noise_utils_is_restored_to_baseline(self) -> None:
        relative = "avhubert/noise_utils.py"
        self.assertEqual(
            (ROOT / relative).read_bytes(),
            git("show", f"{BASELINE}:{relative}"),
        )

    def test_completed_a_l_and_existing_v3_joint_files_are_unchanged(self) -> None:
        protected = []
        for directory in (
            "avhubert/conf/distill_v3/distill_only",
            "avhubert/conf/distill_v3/joint_dp_stage1",
            "avhubert/conf/distill_v3/joint_dp_stage2",
        ):
            protected.extend(
                git("ls-tree", "-r", "--name-only", V3_BASELINE, directory)
                .decode()
                .splitlines()
            )
        protected.extend(
            (
                "scripts_v3/run_ch3_distill_only.sh",
                "scripts_v3/run_ch3_distill_only_48gb.sh",
                "scripts_v3/run_ch3_joint_dp.sh",
                "scripts_v3/run_ch3_joint_dp_48gb.sh",
                "scripts/run_distill_only.sh",
                "scripts/run_distill_only_fine-tuning.sh",
                "scripts/run_distill_only_inference.sh",
            )
        )
        for relative in protected:
            if relative in REPLACED_STAGE_J_STUBS:
                continue
            with self.subTest(path=relative):
                self.assertEqual(
                    (ROOT / relative).read_bytes(),
                    git("show", f"{V3_BASELINE}:{relative}"),
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
