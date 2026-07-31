"""Direct composition and scientific-setting checks for public configs."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "avhubert" / "conf" / "distill_only"
EXPECTED = {
    "b1_t2_teacher_init": ("transformer", 2, 768, 3072, "historical_pred_heads"),
    "b2_c1_t2_random_sequence": (
        "transformer",
        2,
        768,
        3072,
        "historical_pred_heads",
    ),
    "c2_t6_historical_heads": (
        "transformer",
        6,
        384,
        3200,
        "historical_pred_heads",
    ),
    "c3a_t12_historical_heads": (
        "transformer",
        12,
        384,
        1024,
        "historical_pred_heads",
    ),
    "c3b_t12_layer_to_layer": (
        "transformer",
        12,
        384,
        1024,
        "layer_to_layer",
    ),
    "c3c_t12_historical_heads_noisy": (
        "transformer",
        12,
        384,
        1024,
        "historical_pred_heads",
    ),
    "d1_selected_transformer": (
        "transformer",
        6,
        384,
        3200,
        "historical_pred_heads",
    ),
    "d2_selected_conformer": (
        "conformer",
        6,
        384,
        1536,
        "historical_pred_heads",
    ),
    "e1_selected_clean": (
        "transformer",
        6,
        384,
        3200,
        "historical_pred_heads",
    ),
    "e2_selected_noisy": (
        "transformer",
        6,
        384,
        3200,
        "historical_pred_heads",
    ),
    "s1_selected_main": (
        "transformer",
        6,
        384,
        3200,
        "historical_pred_heads",
    ),
    "s2_optional_two_stage": (
        "transformer",
        6,
        384,
        3200,
        "historical_pred_heads",
    ),
}


def load(name: str) -> dict:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT.resolve())):
        value = compose(config_name=name)
    return OmegaConf.to_container(value, resolve=True)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


class ConfigMatrixTest(unittest.TestCase):
    def test_every_public_config_resolves_exact_architecture(self) -> None:
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                config = load(name)
                model = config["model"]
                actual = (
                    model["student_arch"],
                    model["student_depth"],
                    model["student_embed_dim"],
                    model["student_ffn_dim"],
                    model["distill_head_mode"],
                )
                self.assertEqual(actual, expected)
                self.assertEqual(model["teacher_target_layers"], "0,4,8,12")
                self.assertEqual(config["optimizer"]["_name"], "adam")
                forbidden = (
                    "pruning",
                    "log_alpha",
                    "lagrange",
                    "l0",
                    "merge",
                )
                keys = " ".join(key.lower() for key, _ in flatten(config))
                self.assertFalse(any(item in keys for item in forbidden))

    def test_initialization_and_layer_mapping(self) -> None:
        self.assertEqual(
            load("b1_t2_teacher_init")["model"]["initialization_policy"],
            "teacher_sequence_teacher_frontend",
        )
        self.assertEqual(
            load("b2_c1_t2_random_sequence")["model"]["initialization_policy"],
            "random_sequence_teacher_frontend",
        )
        for name in ("c2_t6_historical_heads", "c3a_t12_historical_heads"):
            self.assertEqual(load(name)["model"]["student_match_layers"], "")
        self.assertEqual(
            load("c3b_t12_layer_to_layer")["model"]["student_match_layers"],
            "0,4,8,12",
        )

    def test_main_optimizer_and_optional_schedule(self) -> None:
        main = load("c2_t6_historical_heads")
        self.assertEqual(main["optimization"]["max_update"], 75000)
        self.assertEqual(main["optimization"]["lr"], [0.002])
        self.assertEqual(main["optimization"]["clip_norm"], 10.0)
        self.assertEqual(main["lr_scheduler"]["warmup_updates"], 15000)
        self.assertEqual(main["lr_scheduler"]["total_num_update"], 75000)
        optional = load("s2_optional_two_stage")
        self.assertEqual(optional["optimization"]["max_update"], 50000)
        self.assertEqual(optional["model"]["schedule_mode"], "two_stage_50k_25k")

    def test_distillation_noise_is_enabled_only_for_e2_and_c3c(self) -> None:
        for name in EXPECTED:
            config = load(name)
            expected = (
                0.25
                if name in {"e2_selected_noisy", "c3c_t12_historical_heads_noisy"}
                else 0.0
            )
            self.assertEqual(config["task"]["distillation_noise_prob"], expected)
            self.assertEqual(config["model"]["distillation_noise_prob"], expected)

    def test_c3c_differs_from_c3a_only_by_dedicated_noise(self) -> None:
        clean = load("c3a_t12_historical_heads")
        noisy = load("c3c_t12_historical_heads_noisy")
        root = "/beegfs/data/shared/lrs3/noise/musan/tsv/all"
        for section in ("model", "task"):
            self.assertEqual(noisy[section]["distillation_noise_prob"], 0.25)
            self.assertEqual(noisy[section]["distillation_noise_snr"], "0")
            self.assertEqual(noisy[section]["distillation_noise_method"], "rms")
            self.assertEqual(
                noisy[section]["distillation_noise_manifest_root"], root
            )
            self.assertIs(noisy[section]["distillation_noise_train_only"], True)
            noisy[section]["distillation_noise_prob"] = clean[section][
                "distillation_noise_prob"
            ]
            noisy[section]["distillation_noise_manifest_root"] = clean[section][
                "distillation_noise_manifest_root"
            ]
        self.assertEqual(noisy["task"]["noise_prob"], 0.0)
        self.assertIsNone(noisy["task"]["noise_wav"])
        self.assertEqual(noisy, clean)

    def test_no_matrix_directory_or_self_defaults(self) -> None:
        self.assertFalse((CONFIG_ROOT / "_matrix").exists())
        for path in CONFIG_ROOT.glob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("_matrix/", text)
            self.assertNotIn("_self_", text)


if __name__ == "__main__":
    unittest.main()
