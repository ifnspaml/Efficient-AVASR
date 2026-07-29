"""CPU-only interface tests for the direct shell runner."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_distill_only.sh"
ENV_PYTHON = Path(
    "/home/zhengyangli/work_fast/envs/dpavhubert_pro6000/bin"
)


def run(arguments, *, project=ROOT):
    environment = os.environ.copy()
    environment["CH3_PROJECT_PATH"] = str(project)
    environment["PATH"] = f"{ENV_PYTHON}:{environment['PATH']}"
    return subprocess.run(
        ("bash", str(RUNNER), *arguments),
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class SimpleRunnerTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(("bash", "-n", str(RUNNER)), check=True)

    def test_finetune_dry_run_is_exact_and_creates_nothing(self) -> None:
        label = f"dry-{uuid.uuid4().hex}"
        target = (
            ROOT
            / "exp/chapter3_distill_only/c3a_t12_historical_heads/seed_999991"
            / "finetune_runs"
            / label
        )
        self.assertFalse(target.exists())
        result = run(
            (
                "--exp-name",
                "c3a_t12_historical_heads",
                "--stage",
                "finetune",
                "--seed",
                "999991",
                "--run-name",
                label,
                "--input-checkpoint",
                "/tmp/nonexistent-for-dry-run.pt",
                "--dry-run",
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model.w2v_path=/tmp/nonexistent-for-dry-run.pt", result.stdout)
        self.assertIn(
            f"common.user_dir={ROOT}/chapter3_distill_only", result.stdout
        )
        self.assertIn("optimization.update_freq=\\[8\\]", result.stdout)
        self.assertNotIn("override.noise_method=itut", result.stdout)
        self.assertFalse(target.exists())

    def test_evaluation_dry_run_expands_clean_and_itut_noise(self) -> None:
        label = f"eval-{uuid.uuid4().hex}"
        result = run(
            (
                "--exp-name",
                "c2_t6_historical_heads",
                "--stage",
                "evaluate",
                "--seed",
                "999992",
                "--run-name",
                label,
                "--input-checkpoint",
                "/tmp/nonexistent-eval-dry.pt",
                "--eval-name",
                "screening",
                "--evaluation-phase",
                "screening",
                "--eval-subsets",
                "valid",
                "--dry-run",
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith("$")]
        self.assertEqual(len(commands), 3)
        self.assertNotIn("override.noise_prob", commands[0])
        self.assertIn("override.noise_prob=1", "\n".join(commands[1:]))
        self.assertIn("override.noise_method=itut", "\n".join(commands[1:]))

    def test_stage_validation_and_s2_overrides(self) -> None:
        bad = run(("--exp-name", "../bad", "--stage", "encoder", "--dry-run"))
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unsafe", bad.stderr)
        invalid = run(
            (
                "--exp-name",
                "c2_t6_historical_heads",
                "--stage",
                "encoder",
                "--input-checkpoint",
                "/tmp/x.pt",
                "--dry-run",
            )
        )
        self.assertNotEqual(invalid.returncode, 0)
        s2 = run(
            (
                "--exp-name",
                "s2_optional_two_stage",
                "--stage",
                "stage2",
                "--seed",
                "999993",
                "--dry-run",
            )
        )
        self.assertEqual(s2.returncode, 0, s2.stderr)
        self.assertIn("optimization.max_update=25000", s2.stdout)
        self.assertIn("optimization.lr=\\[0.0001\\]", s2.stdout)
        self.assertIn("model.initialization_policy=warm_start_distilled", s2.stdout)

    def test_nonempty_and_resume_checkpoint_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "avhubert/conf/distill_only").mkdir(parents=True)
            (project / "scripts/distill_only").mkdir(parents=True)
            (project / "avhubert/conf/distill_only/example.yaml").write_text(
                "model: {}\\n", encoding="utf-8"
            )
            (project / "scripts/distill_only/evaluation_protocol_itut.yaml").write_text(
                "schema_version: chapter3-evaluation/v1\\n", encoding="utf-8"
            )
            subprocess.run(("git", "init", "-q"), cwd=project, check=True)
            subprocess.run(("git", "add", "."), cwd=project, check=True)
            subprocess.run(
                (
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ),
                cwd=project,
                check=True,
            )
            output = (
                project
                / "exp/chapter3_distill_only/example/seed_7"
                / "finetune_runs/run"
            )
            output.mkdir(parents=True)
            (output / "unexpected").write_text("x", encoding="utf-8")
            arguments = (
                "--exp-name",
                "example",
                "--stage",
                "finetune",
                "--seed",
                "7",
                "--run-name",
                "run",
                "--input-checkpoint",
                "/tmp/dry.pt",
                "--dry-run",
            )
            first = run(arguments, project=project)
            self.assertIn("output is nonempty", first.stderr)
            resumed = run((*arguments, "--resume"), project=project)
            self.assertIn("checkpoint_last.pt", resumed.stderr)
            checkpoint = output / "checkpoints/checkpoint_last.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"x")
            resumed = run((*arguments, "--resume"), project=project)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)


if __name__ == "__main__":
    unittest.main()
