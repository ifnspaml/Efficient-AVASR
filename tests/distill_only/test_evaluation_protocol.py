"""Tests for the versioned ITU-T post-fine-tuning evaluation protocol."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import yaml

from avhubert import noise_utils
from chapter3_distill_only.evaluation import (
    DEFAULT_PROTOCOL_PATH,
    EvaluationProtocolError,
    condition_name,
    load_evaluation_protocol,
    measurement_metadata,
    snr_condition_suffix,
    validate_protocol_artifacts,
)
from chapter3_distill_only.manifest import (
    LIFECYCLE,
    ManifestError,
    ManifestStore,
    sha256_file,
)
from chapter3_distill_only.selection import (
    create_selection,
    require_final_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_PATH = REPO_ROOT / "scripts" / "distill_only" / "launch.py"
SPEC = importlib.util.spec_from_file_location("chapter3_eval_launcher", LAUNCH_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _advance_to_validation(store: ManifestStore) -> None:
    measurements = {}
    for condition, value in (("clean", 10.0), ("babble_0db", 20.0)):
        artifact = store.path.parent / f"wer.{condition}"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"WER: {value}\n", encoding="utf-8")
        measurements[condition] = {
            "value": value,
            "artifact": str(artifact.resolve()),
            "artifact_sha256": sha256_file(artifact),
        }
    store.update({"measurements": {"wer": {"validation": measurements}}})
    for state in LIFECYCLE[1 : LIFECYCLE.index("validation_complete") + 1]:
        store.transition(state)


def _write_protocol(directory: Path, **overrides) -> Path:
    payload = yaml.safe_load(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload.update(overrides)
    if "noise_roots" in overrides:
        payload["noise_roots"] = overrides["noise_roots"]
    path = directory / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _args_with_protocol(protocol, run_dir: Path) -> Namespace:
    return Namespace(
        run_dir=run_dir,
        evaluation_protocol_obj=protocol,
        data=Path("/tmp/data"),
        tokenizer=Path("/tmp/tok"),
        teacher=Path("/tmp/teacher"),
        noise_root=Path("/tmp/noise"),
        seed=1337,
        gpus=1,
        workers=4,
        finetune_update_freq=8,
        fairseq_train="fairseq-hydra-train",
        experiment="c2_t6_historical_heads",
    )


class EvaluationProtocolTest(unittest.TestCase):
    def test_default_protocol_expands_screening_and_final_matrix(self) -> None:
        protocol = load_evaluation_protocol(DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol.schema_version, "chapter3-evaluation/v1")
        self.assertEqual(protocol.noise_method, "itut")
        self.assertEqual(protocol.evaluation_seed, 1337)
        self.assertEqual(protocol.speech_level_dbov, -26)
        self.assertEqual(
            [item.name for item in protocol.screening],
            ["clean", "babble_0db", "speech_0db"],
        )
        self.assertEqual(len(protocol.final), 32)
        self.assertEqual(protocol.expected_final_condition_count, 32)
        for subset in ("valid", "test"):
            subset_items = [item for item in protocol.final if item.subset == subset]
            self.assertEqual(len(subset_items), 16)
            noisy = [item for item in subset_items if item.noise_type is not None]
            self.assertEqual(len(noisy), 15)
            self.assertEqual(
                sorted({item.snr_db for item in noisy}),
                [-10, -5, 0, 5, 10],
            )
            for noise_type in ("babble", "music", "speech"):
                typed = [item for item in noisy if item.noise_type == noise_type]
                self.assertEqual(len(typed), 5)
            speech_roots = {
                item.noise_root
                for item in noisy
                if item.noise_type == "speech"
            }
            self.assertEqual(
                speech_roots,
                {Path("/beegfs/data/shared/lrs3/noise/speech")},
            )

    def test_negative_snr_names_are_deterministic(self) -> None:
        self.assertEqual(snr_condition_suffix(-10), "m10db")
        self.assertEqual(snr_condition_suffix(-5), "m5db")
        self.assertEqual(snr_condition_suffix(0), "0db")
        self.assertEqual(snr_condition_suffix(5), "p5db")
        self.assertEqual(condition_name("babble", -10), "babble_m10db")
        self.assertEqual(condition_name("music", 10), "music_p10db")

    def test_unsupported_evaluation_method_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_protocol(Path(directory), noise_method="rms")
            with self.assertRaises(EvaluationProtocolError):
                load_evaluation_protocol(path)

    def test_clean_and_noisy_decode_commands(self) -> None:
        protocol = load_evaluation_protocol(DEFAULT_PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            args = _args_with_protocol(protocol, run_dir)
            commands = launcher._decode_commands(
                args,
                phase="screening",
                require_checkpoint=False,
                validate_artifacts=False,
            )
            self.assertEqual(len(commands), 3)
            clean = next(item for item in commands if item[0].name == "clean")
            clean_cmd = " ".join(map(str, clean[1]))
            for key in launcher.NOISE_OVERRIDE_KEYS:
                self.assertNotIn(f"{key}=", clean_cmd)
            self.assertIn("common.seed=1337", clean_cmd)
            self.assertIn("/evaluation/screening/itut/clean/valid", str(clean[2]))

            babble = next(item for item in commands if item[0].name == "babble_0db")
            babble_cmd = " ".join(map(str, babble[1]))
            self.assertIn("override.noise_prob=1", babble_cmd)
            self.assertIn("override.noise_method=itut", babble_cmd)
            self.assertIn("override.noise_snr=0", babble_cmd)
            self.assertIn(
                "override.noise_wav=/beegfs/data/shared/lrs3/noise/musan/tsv/babble",
                babble_cmd,
            )
            self.assertIn("common.seed=1337", babble_cmd)
            self.assertIn("/evaluation/screening/itut/babble/0/valid", str(babble[2]))

            speech = next(item for item in commands if item[0].name == "speech_0db")
            speech_cmd = " ".join(map(str, speech[1]))
            self.assertIn(
                "override.noise_wav=/beegfs/data/shared/lrs3/noise/speech",
                speech_cmd,
            )
            self.assertIn("override.noise_method=itut", speech_cmd)

    def test_final_decode_expands_to_thirty_two_commands(self) -> None:
        protocol = load_evaluation_protocol(DEFAULT_PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            args = _args_with_protocol(protocol, Path(directory))
            commands = launcher._decode_commands(
                args,
                phase="final",
                require_checkpoint=False,
                validate_artifacts=False,
            )
            self.assertEqual(len(commands), 32)
            valid = [item for item in commands if item[0].subset == "valid"]
            test = [item for item in commands if item[0].subset == "test"]
            self.assertEqual(len(valid), 16)
            self.assertEqual(len(test), 16)
            snrs = sorted(
                {
                    item[0].snr_db
                    for item in commands
                    if item[0].noise_type is not None
                }
            )
            self.assertEqual(snrs, [-10, -5, 0, 5, 10])
            for condition, command, output in commands:
                joined = " ".join(map(str, command))
                self.assertIn("common.seed=1337", joined)
                if condition.noise_type is None:
                    for key in launcher.NOISE_OVERRIDE_KEYS:
                        self.assertNotIn(f"{key}=", joined)
                    self.assertIn("/evaluation/final/itut/clean/", str(output))
                else:
                    self.assertIn("override.noise_method=itut", joined)
                    self.assertIn(f"override.noise_snr={condition.snr_db}", joined)
                    self.assertIn(
                        f"/evaluation/final/itut/{condition.noise_type}/"
                        f"{condition.snr_db}/{condition.subset}",
                        str(output),
                    )
                    if condition.snr_db < 0:
                        self.assertIn("_m", condition.name)
                        self.assertIn(f"/{condition.snr_db}/", str(output))

    def test_training_commands_remain_rms_or_clean(self) -> None:
        e2 = launcher.load_composed_config(
            launcher.CONFIG_ROOT / "e2_selected_noisy.yaml"
        )
        launcher.validate_safe_config(e2)
        self.assertEqual(e2["task"]["noise_method"], "rms")
        for name in (
            "b1_t2_teacher_init",
            "b2_c1_t2_random_sequence",
            "c2_t6_historical_heads",
            "c3a_t12_historical_heads",
            "c3b_t12_layer_to_layer",
        ):
            config = launcher.load_composed_config(
                launcher.CONFIG_ROOT / f"{name}.yaml"
            )
            self.assertEqual(float(config["task"]["noise_prob"]), 0.0)
            launcher.validate_safe_config(config)

        with tempfile.TemporaryDirectory() as directory:
            student = Path(directory) / "export" / "student.pt"
            student.parent.mkdir(parents=True)
            student.write_bytes(b"x")
            args = Namespace(
                run_dir=Path(directory),
                data=Path("/tmp/data"),
                tokenizer=Path("/tmp/tok"),
                noise_root=Path("/tmp/noise"),
                seed=7,
                gpus=1,
                workers=1,
                finetune_update_freq=8,
                fairseq_train="fairseq-hydra-train",
            )
            command, _ = launcher._finetune_command(args)
            joined = " ".join(map(str, command))
            self.assertIn("task.noise_prob=0.25", joined)
            self.assertIn("task.noise_snr=0", joined)
            self.assertIn("task.noise_wav=", joined)
            # Match DP finetune: only wav/prob/snr; noise_num/method use schema defaults (rms).
            self.assertNotIn("task.noise_num=", joined)
            self.assertNotIn("task.noise_method=", joined)
            self.assertNotIn("task.noise_method=itut", joined)

        e2_args = Namespace(
            experiment="e2_selected_noisy",
            data=Path("/tmp/data"),
            tokenizer=Path("/tmp/tok"),
            teacher=Path("/tmp/teacher"),
            noise_root=Path("/tmp/noise"),
            seed=1337,
            gpus=1,
            workers=4,
            max_tokens=4000,
            update_freq=4,
            override=[],
        )
        overrides = launcher._base_overrides(e2_args, Path("/tmp/run"))
        joined = " ".join(overrides)
        self.assertIn("task.noise_method=rms", joined)
        self.assertNotIn("task.noise_method=itut", joined)

        bad = dict(e2)
        bad["task"] = dict(e2["task"])
        bad["task"]["noise_method"] = "itut"
        bad["task"]["distillation_noise_method"] = "itut"
        bad["model"] = dict(e2["model"])
        bad["model"]["distillation_noise_method"] = "itut"
        with self.assertRaises(launcher.PreflightError):
            launcher.validate_safe_config(bad)

    def test_selection_still_uses_clean_and_babble_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root / "run")
            store.create({"config_digest": "abc", "resolved_config": {}})
            _advance_to_validation(store)
            speech_artifact = root / "run" / "wer.speech_0db"
            speech_artifact.write_text("WER: 30.0\n", encoding="utf-8")
            store.update(
                {
                    "measurements": {
                        "wer": {
                            "validation": {
                                "speech_0db": {
                                    "value": 30.0,
                                    "artifact": str(speech_artifact.resolve()),
                                    "artifact_sha256": sha256_file(speech_artifact),
                                }
                            }
                        }
                    }
                }
            )
            selection = create_selection(
                root / "c_to_d.json",
                kind="final",
                candidates=[store.path],
                selected=store.path,
                metric="validation_babble_0db_wer",
                rationale="speech_0db must remain unused",
                approver="tester",
            )
            self.assertEqual(selection["metric_condition"], "babble_0db")
            self.assertIn("clean", selection["candidates"][0]["validation_wer"])
            self.assertIn("babble_0db", selection["candidates"][0]["validation_wer"])
            self.assertNotIn(
                "speech_0db", selection["candidates"][0]["validation_wer"]
            )
            with self.assertRaises(ManifestError):
                require_final_selection(
                    root / "missing_final.json",
                    current_manifest=store.path,
                )

    def test_final_evaluation_blocked_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory) / "run")
            store.create({"config_digest": "abc"})
            _advance_to_validation(store)
            lock = Path(directory) / "final.json"
            create_selection(
                lock,
                kind="final",
                candidates=[store.path],
                selected=store.path,
                metric="validation_babble_0db_wer",
                rationale="freeze later",
                approver="tester",
            )
            other = ManifestStore(Path(directory) / "other")
            other.create({"config_digest": "other"})
            _advance_to_validation(other)
            with self.assertRaises(Exception):
                require_final_selection(
                    lock,
                    current_manifest=other.path,
                )

    def test_itut_preflight_and_mixing_dispatch(self) -> None:
        with patch.object(
            noise_utils,
            "_get_itut_noise_tools",
            side_effect=ImportError("missing itut"),
        ):
            with self.assertRaises(ImportError):
                noise_utils.require_noise_method_available("itut")

        called = {}

        def fake_itut(clean_wav, noise_wavs, noise_snr=0, speech_level_dbov=-26):
            called["itut"] = True
            return clean_wav.astype("int16")

        def fake_rms(clean_wav, noise_wavs, noise_snr=0):
            called["rms"] = True
            return clean_wav.astype("int16")

        with patch.object(noise_utils, "add_noise_itut", side_effect=fake_itut), patch.object(
            noise_utils, "add_noise_rms", side_effect=fake_rms
        ), patch.object(
            noise_utils,
            "_get_itut_noise_tools",
            return_value=(object, object),
        ):
            noise_utils.require_noise_method_available("itut")
            import numpy as np

            noise_utils.add_noise(
                np.zeros(8, dtype=np.float32),
                ["unused.wav"],
                noise_snr=0,
                noise_method="itut",
            )
        self.assertTrue(called.get("itut"))
        self.assertFalse(called.get("rms"))

        with self.assertRaises(ValueError):
            noise_utils.require_noise_method_available("unknown")

    def test_manifest_metadata_and_path_isolation(self) -> None:
        protocol = load_evaluation_protocol(DEFAULT_PROTOCOL_PATH)
        self.assertEqual(protocol.as_manifest_dict()["expected_final_condition_count"], 32)
        self.assertEqual(protocol.as_manifest_dict()["final_snrs_db"], [-10, -5, 0, 5, 10])
        self.assertIn("sha256", protocol.as_manifest_dict())

        screening_babble = next(
            item for item in protocol.screening if item.name == "babble_0db"
        )
        final_babble = next(
            item
            for item in protocol.final
            if item.name == "babble_0db" and item.subset == "valid"
        )
        screening_path = screening_babble.output_relative(
            protocol_noise_method=protocol.noise_method
        )
        final_path = final_babble.output_relative(
            protocol_noise_method=protocol.noise_method
        )
        self.assertNotEqual(screening_path, final_path)
        self.assertTrue(str(screening_path).startswith("screening/itut/"))
        self.assertTrue(str(final_path).startswith("final/itut/"))

        rms_like = Path("evaluation/babble_0db/valid")
        self.assertNotEqual(
            Path("evaluation") / screening_path,
            rms_like,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "wer"
            artifact.write_text("WER: 12.5\n", encoding="utf-8")
            metadata = measurement_metadata(
                protocol,
                final_babble,
                value=12.5,
                artifact=artifact,
                artifact_sha256=sha256_file(artifact),
            )
            self.assertEqual(metadata["noise_method"], "itut")
            self.assertEqual(metadata["noise_type"], "babble")
            self.assertEqual(metadata["snr_db"], 0)
            self.assertEqual(metadata["subset"], "valid")
            self.assertEqual(metadata["evaluation_phase"], "final")
            self.assertEqual(metadata["evaluation_seed"], 1337)
            self.assertEqual(metadata["speech_level_dbov"], -26)
            self.assertIn("babble/valid.tsv", metadata["noise_manifest"])

            clean = next(item for item in protocol.screening if item.name == "clean")
            clean_meta = measurement_metadata(
                protocol,
                clean,
                value=1.0,
                artifact=artifact,
                artifact_sha256=sha256_file(artifact),
            )
            self.assertIsNone(clean_meta["noise_method"])
            self.assertIsNone(clean_meta["noise_type"])
            self.assertIsNone(clean_meta["snr_db"])

    def test_validate_protocol_artifacts_requires_manifests(self) -> None:
        protocol = load_evaluation_protocol(DEFAULT_PROTOCOL_PATH)
        with patch(
            "avhubert.noise_utils.require_noise_method_available",
            return_value=None,
        ), patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(EvaluationProtocolError):
                validate_protocol_artifacts(protocol, phases=("screening",))


if __name__ == "__main__":
    unittest.main()
