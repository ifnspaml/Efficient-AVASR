"""Validation for the Chapter 3 Stage-K joint-DP redesign."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "avhubert/conf/distill_v3"
RUNNER = ROOT / "scripts_v3/run_ch3_joint_dp_stage_k.sh"
ENV_PYTHON = Path("/home/zhengyangli/work_fast/envs/dpavhubert_pro6000/bin")
MATRIX = {
    "k-ref": ("kref_l2l_t04812_raw_l1_0p1", "kref_l2l_t04812_raw_l1_0p1", "layer2layer", "0.4,8,12", "raw", 0.1, 0.1),
    "k-t": ("kt_l2l_t812_raw_l1_0p1", "kt_l2l_t812_raw_l1_0p1", "layer2layer", "8,12", "raw", 0.1, 0.1),
    "k0": ("k0_pred_t812_raw_l1_0p1", "k0_pred_t812_raw_l1_0p1", "historical_pred_heads", "8,12", "raw", 0.1, 0.1),
    "k1": ("k1_pred_t812_log_sig_l1_0p1", "k1_pred_t812_log_sig_l1_0p1", "historical_pred_heads", "8,12", "log_sig", 0.1, 0.1),
    "k2": ("k2_pred_t812_raw_l1_0p1", "k2_pred_t812_raw_l1_1p0", "historical_pred_heads", "8,12", "raw", 0.1, 1.0),
    "k3": ("k3_pred_t812_log_sig_l1_0p1", "k3_pred_t812_log_sig_l1_1p0", "historical_pred_heads", "8,12", "log_sig", 0.1, 1.0),
}


def load(directory: str, name: str) -> dict:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str((CONFIG_ROOT / directory).resolve())):
        value = compose(config_name=name)
    return OmegaConf.to_container(value, resolve=True)


def changed_paths(left, right, prefix="") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(changed_paths(left[key], right[key], path))
        return result
    return set() if left == right else {prefix}


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CH3_PROJECT_PATH"] = str(ROOT)
    environment["PATH"] = f"{ENV_PYTHON}:{environment['PATH']}"
    return subprocess.run(
        ("bash", str(RUNNER), *arguments),
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageKConfigTest(unittest.TestCase):
    def test_exact_matrix_and_fixed_protocol(self) -> None:
        for experiment, expected in MATRIX.items():
            stage1_name, stage2_name, matching, layers, cosine, l1_first, l1_second = expected
            with self.subTest(experiment=experiment):
                first = load("joint_dp_stage1", stage1_name)
                second = load("joint_dp_stage2", stage2_name)
                self.assertEqual(first["model"]["distill_mode"], matching)
                self.assertEqual(second["model"]["distill_mode"], matching)
                self.assertEqual(first["model"]["distill_layers"], layers)
                self.assertEqual(second["model"]["distill_layers"], layers)
                self.assertEqual(first["criterion"]["cos_type"], cosine)
                self.assertEqual(second["criterion"]["cos_type"], cosine)
                self.assertEqual(first["criterion"]["l1_weight"], l1_first)
                self.assertEqual(second["criterion"]["l1_weight"], l1_second)
                self.assertEqual(first["model"]["pruning_units"], "conv,head,interm")
                self.assertEqual(first["criterion"]["target_sparsity"], 0.70)
                self.assertEqual(first["optimization"]["max_update"], 50000)
                self.assertEqual(second["optimization"]["max_update"], 25000)
                for cfg in (first, second):
                    self.assertEqual(cfg["common"]["seed"], 1337)
                    self.assertEqual(cfg["task"]["noise_prob"], 0.25)
                    self.assertEqual(cfg["task"]["noise_snr"], 0)
                    self.assertEqual(cfg["optimization"]["clip_norm"], 10.0)
                    self.assertEqual(cfg["optimization"]["update_freq"], [4])

    def test_controlled_comparisons_change_only_declared_fields(self) -> None:
        configs = {
            experiment: (load("joint_dp_stage1", values[0]), load("joint_dp_stage2", values[1]))
            for experiment, values in MATRIX.items()
        }
        self.assertEqual(changed_paths(configs["k-ref"][0], configs["k-t"][0]), {"model.distill_layers"})
        self.assertEqual(changed_paths(configs["k-ref"][1], configs["k-t"][1]), {"model.distill_layers"})
        self.assertEqual(changed_paths(configs["k-t"][0], configs["k0"][0]), {"model.distill_mode"})
        self.assertEqual(changed_paths(configs["k-t"][1], configs["k0"][1]), {"model.distill_mode"})
        for stage in (0, 1):
            self.assertEqual(changed_paths(configs["k0"][stage], configs["k1"][stage]), {"criterion.cos_type"})
            self.assertEqual(changed_paths(configs["k2"][stage], configs["k3"][stage]), {"criterion.cos_type"})
        self.assertEqual(changed_paths(configs["k0"][0], configs["k2"][0]), set())
        self.assertEqual(changed_paths(configs["k0"][1], configs["k2"][1]), {"criterion.l1_weight"})
        self.assertEqual(changed_paths(configs["k1"][0], configs["k3"][0]), set())
        self.assertEqual(changed_paths(configs["k1"][1], configs["k3"][1]), {"criterion.l1_weight"})


class PredictionHeadAuditTest(unittest.TestCase):
    def test_joint_head_is_numerically_equivalent_to_stage_l_head(self) -> None:
        stage_l = module_from(ROOT / "chapter3_distill_only/heads.py", "stage_l_heads")
        joint = module_from(ROOT / "avhubert/historical_prediction_heads.py", "joint_heads")
        torch.manual_seed(91)
        expected = stage_l.HistoricalPredictionHeads(6, 8, (8, 12))
        torch.manual_seed(91)
        actual = joint.HistoricalPredictionHeads(6, 8, (8, 12))
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
        source = torch.randn(2, 5, 6)
        torch.testing.assert_close(expected(source), actual(source), rtol=0, atol=0)
        self.assertEqual(tuple(actual(source).shape), (2, 2, 5, 8))

        production_head = joint.HistoricalPredictionHeads(768, 768, (8, 12))
        production_output = production_head(torch.randn(1, 3, 768))
        self.assertEqual(tuple(production_output.shape), (1, 2, 3, 768))

    def test_existing_layer2layer_and_predlayer_forward_paths_are_unchanged(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "avhubert"))
        from hubert_distill import AVHubertDistill

        teacher_values = [torch.randn(2, 4, 3) for _ in range(3)]
        student_values = [torch.randn(2, 4, 3) for _ in range(3)]

        class Teacher(nn.Module):
            def extract_intermediate_features(self, **unused):
                return teacher_values

        class Student(nn.Module):
            def extract_intermediate_features(self, **unused):
                return student_values

            def extract_features(self, **unused):
                raise AssertionError("legacy modes must not call extract_features")

        source = {"audio": torch.empty(0), "video": torch.empty(0)}
        kwargs = {"source": source, "padding_mask": None}
        identities = nn.ModuleList([nn.Identity(), nn.Identity()])
        l2l = AVHubertDistill(Teacher(), Student(), [0, 2], identities, SimpleNamespace(distill_mode="layer2layer"))
        torch.testing.assert_close(
            l2l(**kwargs)["student_hiddens"],
            torch.stack((student_values[0], student_values[2]), dim=1),
        )
        pred_heads = nn.ModuleList((nn.Identity(), nn.Identity()))
        pred = AVHubertDistill(Teacher(), Student(), [0, 2], pred_heads, SimpleNamespace(distill_mode="predlayer"))
        torch.testing.assert_close(
            pred(**kwargs)["student_hiddens"],
            torch.stack((student_values[-1], student_values[-1]), dim=1),
        )


class StageKLauncherTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(("bash", "-n", str(RUNNER)), cwd=ROOT, check=True)

    def test_all_six_dry_runs_are_auditable_and_create_nothing(self) -> None:
        for experiment, expected in MATRIX.items():
            output_id = "k_ref" if experiment == "k-ref" else ("k_t" if experiment == "k-t" else experiment)
            target = ROOT / f"exp/chapter3_joint_dp/{output_id}/seed_1337"
            self.assertFalse(target.exists())
            result = run("--experiment", experiment, "--stage", "all", "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            stage1_name, stage2_name, matching, layers, cosine, l1_first, l1_second = expected
            targets = "0,4,8,12" if layers == "0.4,8,12" else layers
            for text in (
                f"Stage-1 config: {CONFIG_ROOT}/joint_dp_stage1/{stage1_name}.yaml",
                f"Stage-2 config: {CONFIG_ROOT}/joint_dp_stage2/{stage2_name}.yaml",
                f"Matching: {matching}",
                f"Teacher targets: {{{targets}}}",
                f"Cosine: {cosine}",
                f"L1: Stage 1={l1_first} Stage 2={l1_second}",
                "Pruning units: conv,head,interm",
                "Target sparsity: 0.70",
                "updates=50000 clip_norm=10.0 update_freq=4",
                "updates=25000 clip_norm=10.0 update_freq=4",
                "updates=60000 freeze=48000 lr=0.0005 clip_norm=0.0 update_freq=8",
                f"Output root: {target}",
            ):
                self.assertIn(text, result.stdout)
            self.assertEqual(result.stdout.count("avhubert/infer_s2s.py"), 3)
            self.assertFalse(target.exists())

    def test_final_evaluation_defaults_to_full_valid_and_test_grid(self) -> None:
        result = run(
            "--experiment", "k0", "--stage", "evaluate",
            "--evaluation-phase", "final", "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("subsets=valid,test", result.stdout)
        self.assertIn("test WER is reporting-only", result.stdout)
        self.assertEqual(result.stdout.count("avhubert/infer_s2s.py"), 32)

    def test_screening_rejects_test_subset(self) -> None:
        result = run(
            "--experiment", "k0", "--stage", "evaluate",
            "--evaluation-phase", "screening",
            "--evaluation-subsets", "test", "--dry-run",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation-only", result.stderr)


class StageKArtifactTest(unittest.TestCase):
    def test_prune_and_export_preserve_training_head_state(self) -> None:
        artifacts = module_from(
            ROOT / "scripts_v3/joint_dp/stage_k_artifacts.py",
            "stage_k_artifacts",
        )
        student = MagicMock()
        student.prune.return_value = (
            [1, 2], [True], [True], [6], [1024]
        )
        student.state_dict.return_value = {"student.weight": torch.ones(1)}
        wrapper = SimpleNamespace(
            student=student,
            distill_linear_projs=nn.Linear(2, 2),
        )
        load = MagicMock(return_value=([wrapper], None))
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.pt"
            torch.save(
                {
                    "cfg": {"model": {}},
                    "model": {"old": torch.zeros(1)},
                    "extra_state": {},
                },
                original,
            )
            pruned = artifacts.prune(load, Path("wrapper.pt"), original)
            self.assertEqual(pruned["model"].keys(), {"student.weight"})
            self.assertEqual(
                pruned["cfg"]["model"]["encoder_attention_heads_detailed"],
                [6],
            )
            self.assertIn("distill_linear_projs", pruned["extra_state"])
            exported = artifacts.export(load, Path("stage2.pt"), original)
            self.assertEqual(exported["model"].keys(), {"student.weight"})
            self.assertIn("distill_linear_projs", exported["extra_state"])


if __name__ == "__main__":
    unittest.main()
