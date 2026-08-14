"""CPU-only validation of the isolated Chapter 3 v3 experiment suite."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf

from chapter3_distill_only import exporter
from scripts_v3.distill_only.stage_x_tools import relative_targets


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "avhubert/conf/distill_v3"
DISTILL_RUNNER = ROOT / "scripts_v3/run_ch3_distill_only.sh"
JOINT_RUNNER = ROOT / "scripts_v3/run_ch3_joint_dp.sh"
FINAL_RUNNER = ROOT / "scripts_v3/run_ch3_final_evaluation.sh"
ENV_PYTHON = Path("/home/zhengyangli/work_fast/envs/dpavhubert_pro6000/bin")
A_CONFIGS = {
    "a0_t2_teacher_init_noisy": (2, 768, 3072, "teacher_sequence_teacher_frontend"),
    "a1_t2_random_sequence_noisy": (2, 768, 3072, "random_sequence_teacher_frontend"),
    "a2_t6_historical_heads_noisy": (6, 384, 3200, "random_sequence_teacher_frontend"),
    "a3_t12_historical_heads_noisy": (12, 384, 1024, "random_sequence_teacher_frontend"),
    "a4_t12_historical_heads_noisy_student_only": (12, 384, 1024, "random_sequence_teacher_frontend"),
}
L_CONFIGS = {
    "l0_t12_historical_heads_noisy": ("historical_pred_heads", "0,4,8,12", "", 0.25),
    "l1_t12_historical_heads_noisy_targets4812": ("historical_pred_heads", "4,8,12", "", 0.25),
    "l2_t12_historical_heads_noisy_targets812": ("historical_pred_heads", "8,12", "", 0.25),
    "l3_t12_historical_heads_noisy_target12": ("historical_pred_heads", "12", "", 0.25),
    "l4_t12_layer_to_layer_noisy": ("layer_to_layer", "0,4,8,12", "0,4,8,12", 0.25),
    "l5_t12_layer_to_layer_noisy_target12": ("layer_to_layer", "12", "12", 0.25),
    "l6_t12_historical_heads_clean": ("historical_pred_heads", "0,4,8,12", "", 0.0),
}
K_CONFIGS = {
    "k0_hybrid_tau70_raw_cos_l1_0p1": ("raw", 0.1),
    "k1_hybrid_tau70_log_sig_cos_l1_0p1": ("log_sig", 0.1),
    "k2_hybrid_tau70_raw_cos_l1_0p0": ("raw", 0.0),
    "k2_hybrid_tau70_log_sig_cos_l1_0p0": ("log_sig", 0.0),
    "k3_hybrid_tau70_raw_cos_l1_1p0": ("raw", 1.0),
    "k3_hybrid_tau70_log_sig_cos_l1_1p0": ("log_sig", 1.0),
}


def load(directory: str, name: str, overrides: list[str] | None = None) -> dict:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str((CONFIG_ROOT / directory).resolve())):
        value = compose(config_name=name, overrides=overrides or [])
    return OmegaConf.to_container(value, resolve=True)


def run(script: Path, arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CH3_PROJECT_PATH"] = str(ROOT)
    environment["PATH"] = f"{ENV_PYTHON}:{environment['PATH']}"
    return subprocess.run(
        ("bash", str(script), *arguments),
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class V3ConfigTest(unittest.TestCase):
    def test_a_matrix_and_fixed_protocol(self) -> None:
        noise_root = "/beegfs/data/shared/lrs3/noise/musan/tsv/all"
        for name, architecture in A_CONFIGS.items():
            with self.subTest(name=name):
                cfg = load("distill_only", name)
                model = cfg["model"]
                self.assertEqual(
                    (model["student_depth"], model["student_embed_dim"], model["student_ffn_dim"], model["initialization_policy"]),
                    architecture,
                )
                self.assertEqual(model["student_attention_heads"], 12)
                self.assertEqual(model["teacher_target_layers"], "0,4,8,12")
                self.assertEqual(model["student_match_layers"], "")
                self.assertEqual(cfg["optimization"]["max_update"], 75000)
                self.assertEqual(cfg["optimization"]["lr"], [0.002])
                self.assertEqual(cfg["optimization"]["clip_norm"], 10.0)
                self.assertEqual(cfg["optimization"]["update_freq"], [4])
                self.assertEqual(cfg["lr_scheduler"]["warmup_updates"], 15000)
                self.assertEqual(cfg["common"]["seed"], 1337)
                for section in ("model", "task"):
                    self.assertEqual(cfg[section]["distillation_noise_prob"], 0.25)
                    self.assertEqual(cfg[section]["distillation_noise_snr"], "0")
                    self.assertEqual(cfg[section]["distillation_noise_method"], "rms")
                    self.assertEqual(cfg[section]["distillation_noise_manifest_root"], noise_root)
                    self.assertIs(cfg[section]["distillation_noise_train_only"], True)
                    self.assertIs(
                        cfg[section]["distillation_noise_student_only"],
                        name == "a4_t12_historical_heads_noisy_student_only",
                    )
                self.assertEqual(cfg["task"]["noise_prob"], 0.0)
                self.assertIsNone(cfg["task"]["noise_wav"])

    def test_l_matrix_changes_only_supervision_and_encoder_noise(self) -> None:
        for name, expected in L_CONFIGS.items():
            with self.subTest(name=name):
                cfg = load("distill_only", name)
                model = cfg["model"]
                self.assertEqual(model["student_depth"], 12)
                self.assertEqual(model["student_embed_dim"], 384)
                self.assertEqual(model["student_ffn_dim"], 1024)
                self.assertEqual(
                    (model["distill_head_mode"], model["teacher_target_layers"], model["student_match_layers"], model["distillation_noise_prob"]),
                    expected,
                )
                self.assertEqual(cfg["task"]["distillation_noise_prob"], expected[3])
                self.assertEqual(model["l1_weight"], cfg["criterion"]["l1_weight"])
                self.assertEqual(model["cosine_type"], cfg["criterion"]["cosine_type"])

    def test_k_pairs_are_loss_homogeneous(self) -> None:
        for name, loss in K_CONFIGS.items():
            with self.subTest(name=name):
                first = load("joint_dp_stage1", name)
                second = load("joint_dp_stage2", name)
                self.assertEqual((first["criterion"]["cos_type"], first["criterion"]["l1_weight"]), loss)
                self.assertEqual((second["criterion"]["cos_type"], second["criterion"]["l1_weight"]), loss)
                self.assertEqual(first["optimization"]["max_update"], 50000)
                self.assertEqual(second["optimization"]["max_update"], 25000)
                self.assertIs(first["criterion"]["use_reg"], True)
                self.assertIs(second["criterion"]["use_reg"], False)
                self.assertEqual(first["criterion"]["target_sparsity"], 0.70)
                self.assertEqual(first["model"]["pruning_units"], "conv,head,interm")

    def test_unrun_stage_j_placeholders_were_retired(self) -> None:
        for directory in ("joint_dp_stage1", "joint_dp_stage2"):
            for name in (
                "j1_transformer_tau65",
                "j2_hybrid_tau70",
                "j3_hybrid_tau80",
            ):
                self.assertFalse((CONFIG_ROOT / directory / f"{name}.yaml").exists())


class V3LauncherTest(unittest.TestCase):
    def test_shell_syntax_and_forbidden_provenance(self) -> None:
        scripts = list((ROOT / "scripts_v3").rglob("*.sh"))
        subprocess.run(("bash", "-n", *(str(path) for path in scripts)), check=True)
        forbidden = ("git rev-parse", "git status", "sha256sum", "source_snapshot", "immutable_sha256", "commit mismatch", "tree mismatch")
        text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "scripts_v3").rglob("*.*") if path.is_file())
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_every_a_and_l_encoder_dry_run_uses_v3_and_creates_nothing(self) -> None:
        for name in (*A_CONFIGS, *L_CONFIGS):
            seed = str(900000 + len(name))
            target = ROOT / f"exp/chapter3_distill_only/{name}/seed_{seed}"
            self.assertFalse(target.exists())
            result = run(DISTILL_RUNNER, ("--exp-name", name, "--stage", "encoder", "--seed", seed, "--dry-run"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("avhubert/conf/distill_v3/distill_only", result.stdout)
            self.assertNotIn("chapter3_distill_only_preflight", result.stdout)
            self.assertFalse(target.exists())

    def test_centralized_finetuning_contract_is_used_by_both_launchers(self) -> None:
        required = (
            "optimization.max_update=60000",
            "model.freeze_finetune_updates=48000",
            "optimization.lr=\\[0.0005\\]",
            "optimization.clip_norm=0.0",
            "optimization.update_freq=\\[8\\]",
            "task.noise_prob=0.25",
            "task.noise_snr=0",
        )
        distill = run(DISTILL_RUNNER, ("--exp-name", "a3_t12_historical_heads_noisy", "--stage", "finetune", "--input-checkpoint", "/tmp/v3-dry.pt", "--dry-run"))
        joint = run(JOINT_RUNNER, ("--exp-name", "k0_hybrid_tau70_raw_cos_l1_0p1", "--stage", "finetune", "--input-checkpoint", "/tmp/v3-dry.pt", "--dry-run"))
        self.assertEqual(distill.returncode, 0, distill.stderr)
        self.assertEqual(joint.returncode, 0, joint.stderr)
        self.assertIn("scripts_v3/distill_only/run_finetune.sh", distill.stdout)
        self.assertIn("scripts_v3/distill_only/run_finetune.sh", joint.stdout)
        for token in required:
            self.assertIn(token, distill.stdout)
            self.assertIn(token, joint.stdout)

    def test_legacy_launcher_does_not_accept_stage_j(self) -> None:
        mismatch = run(JOINT_RUNNER, ("--exp-name", "k0_hybrid_tau70_raw_cos_l1_0p1", "--stage", "joint_dp", "--cosine-type", "log_sig", "--dry-run"))
        self.assertNotEqual(mismatch.returncode, 0)
        for name in ("j1_transformer_tau65", "j2_hybrid_tau70", "j3_hybrid_tau80"):
            result = run(JOINT_RUNNER, ("--exp-name", name, "--stage", "joint_dp", "--dry-run"))
            self.assertNotEqual(result.returncode, 0)

    def test_final_registry_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fixture")
            registry = root / "models.yaml"
            registry.write_text(f"f1_best_manual:\n  checkpoint: {checkpoint}\n", encoding="utf-8")
            result = run(FINAL_RUNNER, ("--registry", str(registry), "--models", "f1_best_manual", "--dry-run"))
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [line for line in result.stdout.splitlines() if line.startswith("$")]
            self.assertEqual(len(commands), 32)
            self.assertTrue(all("chapter3_distill_only/stage_f/" in command for command in commands))


class V3ExporterAndExtensionTest(unittest.TestCase):
    def test_no_digest_export_never_calls_digest_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            output = Path(directory) / "student.pt"
            source.write_bytes(b"source")
            wrapper = MagicMock()
            wrapper.get_export_state.return_value = {"model": {}, "extra_state": {}}
            wrapper.get_student_num_params.return_value = 1
            wrapper.get_prediction_head_num_params.return_value = 2
            wrapper.get_teacher_num_params.return_value = 3
            wrapper.student.cfg.student_arch = "transformer"
            wrapper.student.encoder.layers = [object()]
            wrapper.student.encoder_embed_dim = 384
            with patch.object(exporter, "_load_wrapper", return_value=wrapper), patch.object(exporter, "verify_export", return_value={"verified": True}), patch.object(exporter, "_sha256_file", side_effect=AssertionError("digest called")):
                result = exporter.export_checkpoint(source, output, include_digests=False)
            self.assertNotIn("source_sha256", result)
            self.assertNotIn("checkpoint_sha256", result)
            metadata = torch.load(output, map_location="cpu", weights_only=False)["extra_state"]["chapter3_distill_only"]
            self.assertNotIn("source_distillation_checkpoint_sha256", metadata)

    def test_relative_large_teacher_targets(self) -> None:
        self.assertEqual(relative_targets(12), (0, 4, 8, 12))
        self.assertEqual(relative_targets(24), (0, 8, 16, 24))
        with self.assertRaises(ValueError):
            relative_targets(2)


if __name__ == "__main__":
    unittest.main()
