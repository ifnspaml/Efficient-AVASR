"""Tests for immutable run manifests and scientific selection gates."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from typing import Optional

from chapter3_distill_only.manifest import (
    LIFECYCLE,
    ManifestError,
    ManifestStore,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from chapter3_distill_only.profiling import (
    profile_architecture_fields,
    select_conformer_ffn_match,
    validate_config_profile,
)
from chapter3_distill_only.selection import (
    SelectionError,
    create_selection,
    freeze_selected_manifest,
    inherited_hydra_overrides,
    require_final_selection,
    read_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MATCH_TOOL = REPO_ROOT / "scripts" / "distill_only" / "match_conformer.py"
MATCH_SPEC = importlib.util.spec_from_file_location(
    "chapter3_test_match_conformer", MATCH_TOOL
)
assert MATCH_SPEC is not None and MATCH_SPEC.loader is not None
match_tool = importlib.util.module_from_spec(MATCH_SPEC)
MATCH_SPEC.loader.exec_module(match_tool)
_candidate = match_tool._candidate


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


def _resolved_profile_config(
    *,
    architecture: str,
    ffn_dim: int,
    depth: int = 6,
    embed_dim: int = 384,
    heads: int = 12,
) -> dict:
    return {
        "model": {
            "student_arch": architecture,
            "student_depth": depth,
            "student_embed_dim": embed_dim,
            "student_ffn_dim": ffn_dim,
            "student_attention_heads": heads,
            "student_conformer_kernel": 31,
            "student_conformer_attention_type": "original",
            "student_conformer_position_type": "abs",
            "distill_head_mode": "historical_pred_heads",
            "prediction_head_hidden_dim": -1,
            "teacher_target_layers": "0,4,8,12",
            "student_match_layers": "",
            "initialization_policy": "random_sequence_teacher_frontend",
            "distill_loss_type": "historical",
            "l1_weight": 1.0,
            "l2_weight": 0.0,
            "cosine_weight": 1.0,
            "cosine_type": "log_sig",
            "feature_penalty_weight": 0.0,
        },
        "criterion": {
            "distill_loss_type": "historical",
            "l1_weight": 1.0,
            "l2_weight": 0.0,
            "cosine_weight": 1.0,
            "cosine_type": "log_sig",
            "feature_penalty_weight": 0.0,
        },
        "dataset": {"max_tokens": 4000},
        "optimization": {
            "max_update": 75000,
            "clip_norm": 10.0,
            "update_freq": [4],
            "lr": [0.002],
        },
        "optimizer": {
            "_name": "adam",
            "adam_betas": "(0.9,0.999)",
            "adam_eps": 1e-8,
            "weight_decay": 0.0,
        },
        "lr_scheduler": {
            "_name": "polynomial_decay",
            "power": 1,
            "warmup_updates": 15000,
            "total_num_update": 75000,
        },
    }


def _profile(
    source: ManifestStore,
    *,
    architecture: str,
    ffn_dim: int,
    parameters: int,
    flops: int,
    depth: int = 6,
    embed_dim: int = 384,
    heads: int = 12,
) -> dict:
    manifest = source.read()
    return {
        "schema_version": "chapter3-config-profile/v1",
        "source_manifest": {
            "path": str(source.path.resolve()),
            "immutable_sha256": manifest["immutable_sha256"],
            "config_digest": manifest["immutable"].get("config_digest"),
        },
        "resolved_config": _resolved_profile_config(
            architecture=architecture,
            ffn_dim=ffn_dim,
            depth=depth,
            embed_dim=embed_dim,
            heads=heads,
        ),
        "student_arch": architecture,
        "student_depth": depth,
        "student_embed_dim": embed_dim,
        "student_ffn_dim": ffn_dim,
        "student_attention_heads": heads,
        "deployed_backbone_parameters": parameters,
        "flops": flops,
    }


def _write_match(
    root: Path,
    source: ManifestStore,
    *,
    suffix: str = "",
    source_override: Optional[ManifestStore] = None,
    candidate_depth: int = 6,
) -> tuple[Path, Path, Path]:
    profile_source = source_override or source
    target_path = root / f"transformer{suffix}.json"
    conformer_path = root / f"conformer_1536{suffix}.json"
    target = _profile(
        profile_source,
        architecture="transformer",
        ffn_dim=3200,
        parameters=1000,
        flops=2000,
    )
    conformer = _profile(
        profile_source,
        architecture="conformer",
        ffn_dim=1536,
        parameters=1010,
        flops=1990,
        depth=candidate_depth,
    )
    atomic_write_json(target_path, target)
    atomic_write_json(conformer_path, conformer)
    candidate = {
        **conformer,
        "profile_path": str(conformer_path.resolve()),
        "profile_sha256": sha256_file(conformer_path),
    }
    match = select_conformer_ffn_match(target, (candidate,))
    match.update(
        {
            "schema_version": "chapter3-conformer-match/v1",
            "created_at": utc_now(),
            "target_profile_path": str(target_path.resolve()),
            "target_profile_sha256": sha256_file(target_path),
        }
    )
    match_path = root / f"conformer_match{suffix}.json"
    atomic_write_json(match_path, match)
    return match_path, target_path, conformer_path


class ManifestStoreTest(unittest.TestCase):
    def test_lifecycle_and_immutable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory) / "run")
            created = store.create({"experiment": "C2", "seed": 1337})
            self.assertEqual(created["state"], "prepared")
            self.assertEqual(
                store.create({"experiment": "C2", "seed": 1337})["state"],
                "prepared",
            )
            with self.assertRaises(ManifestError):
                store.create({"experiment": "C3", "seed": 1337})
            with self.assertRaises(ManifestError):
                store.transition("encoder_complete")
            store.transition("encoder_running")
            updated = store.update(
                {"measurements": {"runtime_seconds": 1.5}},
                expected_state="encoder_running",
            )
            self.assertEqual(
                updated["measurements"]["runtime_seconds"], 1.5
            )


class SelectionGateTest(unittest.TestCase):
    def test_selection_is_write_once_and_gates_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ManifestStore(root / "c2")
            second = ManifestStore(root / "c3")
            first.create({"experiment": "C2"})
            second.create({"experiment": "C3"})
            _advance_to_validation(first)
            _advance_to_validation(second)

            lock = root / "selection" / "final.json"
            create_selection(
                lock,
                kind="final",
                candidates=(first.path, second.path),
                selected=first.path,
                metric="valid_babble_0db_wer",
                rationale="controlled test decision",
                approver="unit-test",
            )
            require_final_selection(lock, current_manifest=first.path)
            with self.assertRaises(SelectionError):
                require_final_selection(lock, current_manifest=second.path)
            with self.assertRaises(SelectionError):
                create_selection(
                    lock,
                    kind="final",
                    candidates=(first.path, second.path),
                    selected=second.path,
                    metric="valid_babble_0db_wer",
                    rationale="attempted mutation",
                    approver="unit-test",
                )
            frozen = freeze_selected_manifest(lock)
            self.assertEqual(frozen["state"], "selection_frozen")

    def test_selection_requires_and_seals_validation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = ManifestStore(root / "missing")
            missing.create({"experiment": "missing"})
            for state in LIFECYCLE[
                1 : LIFECYCLE.index("validation_complete") + 1
            ]:
                missing.transition(state)
            with self.assertRaises(SelectionError):
                create_selection(
                    root / "missing.json",
                    kind="final",
                    candidates=(missing.path,),
                    selected=missing.path,
                    metric="validation_clean_wer",
                    rationale="must fail",
                    approver="unit-test",
                )

            valid = ManifestStore(root / "valid")
            valid.create({"experiment": "valid"})
            _advance_to_validation(valid)
            with self.assertRaises(SelectionError):
                create_selection(
                    root / "c_to_d.json",
                    kind="c_to_d",
                    candidates=(valid.path,),
                    selected=valid.path,
                    metric="validation_babble_0db_wer",
                    rationale="missing match evidence",
                    approver="unit-test",
                )
            lock = root / "final.json"
            create_selection(
                lock,
                kind="final",
                candidates=(valid.path,),
                selected=valid.path,
                metric="validation_clean_wer",
                rationale="sealed evidence",
                approver="unit-test",
            )
            artifact = valid.path.parent / "wer.clean"
            artifact.write_text("WER: 99.0\n", encoding="utf-8")
            with self.assertRaises(SelectionError):
                read_selection(lock)

    def test_c_to_d_freezes_conformer_match_profile_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_manifest = ManifestStore(root / "c2")
            candidate_manifest.create(
                {
                    "experiment": "C2",
                    "config_digest": "c2-config",
                    "resolved_config": _resolved_profile_config(
                        architecture="transformer",
                        ffn_dim=3200,
                    ),
                }
            )
            _advance_to_validation(candidate_manifest)
            match_path, _, conformer_path = _write_match(
                root, candidate_manifest
            )

            selection_path = root / "c_to_d.json"
            record = create_selection(
                selection_path,
                kind="c_to_d",
                candidates=(candidate_manifest.path,),
                selected=candidate_manifest.path,
                metric="validation_babble_0db_wer",
                rationale="profile-matched control",
                approver="unit-test",
                conformer_match=match_path,
            )
            self.assertEqual(
                record["conformer_match"]["selected_student_ffn_dim"], 1536
            )
            identical = create_selection(
                selection_path,
                kind="c_to_d",
                candidates=(candidate_manifest.path,),
                selected=candidate_manifest.path,
                metric="validation_babble_0db_wer",
                rationale="profile-matched control",
                approver="unit-test",
                conformer_match=match_path,
            )
            self.assertEqual(
                identical["selection_sha256"], record["selection_sha256"]
            )
            alternate_match = root / "alternate_match.json"
            alternate_match.write_bytes(match_path.read_bytes())
            with self.assertRaises(SelectionError):
                create_selection(
                    selection_path,
                    kind="c_to_d",
                    candidates=(candidate_manifest.path,),
                    selected=candidate_manifest.path,
                    metric="validation_babble_0db_wer",
                    rationale="profile-matched control",
                    approver="unit-test",
                    conformer_match=alternate_match,
                )
            conformer_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SelectionError):
                read_selection(selection_path)

    def test_conformer_match_rejects_wrong_source_and_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = ManifestStore(root / "selected")
            selected.create(
                {
                    "config_digest": "selected",
                    "resolved_config": _resolved_profile_config(
                        architecture="transformer", ffn_dim=3200
                    ),
                }
            )
            other = ManifestStore(root / "other")
            other.create(
                {
                    "config_digest": "other",
                    "resolved_config": _resolved_profile_config(
                        architecture="transformer", ffn_dim=3200
                    ),
                }
            )
            _advance_to_validation(selected)
            match_path, _, _ = _write_match(
                root, selected, source_override=other
            )
            with self.assertRaisesRegex(
                SelectionError, "not bound to the selected C manifest"
            ):
                create_selection(
                    root / "wrong_source.json",
                    kind="c_to_d",
                    candidates=(selected.path,),
                    selected=selected.path,
                    metric="validation_babble_0db_wer",
                    rationale="wrong source",
                    approver="unit-test",
                    conformer_match=match_path,
                )

            target = _profile(
                selected,
                architecture="transformer",
                ffn_dim=3200,
                parameters=1000,
                flops=2000,
            )
            changed_depth = _profile(
                selected,
                architecture="conformer",
                ffn_dim=1536,
                parameters=1010,
                flops=1990,
                depth=12,
            )
            with self.assertRaisesRegex(ValueError, "student_depth"):
                select_conformer_ffn_match(target, (changed_depth,))

    def test_match_cli_ffn_must_equal_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ManifestStore(root / "source")
            source.create(
                {
                    "config_digest": "source",
                    "resolved_config": _resolved_profile_config(
                        architecture="transformer", ffn_dim=3200
                    ),
                }
            )
            profile_path = root / "conformer.json"
            atomic_write_json(
                profile_path,
                _profile(
                    source,
                    architecture="conformer",
                    ffn_dim=1536,
                    parameters=1000,
                    flops=2000,
                ),
            )
            with self.assertRaises(ArgumentTypeError):
                _candidate(f"2048={profile_path}")

    def test_profile_schema_emits_resolved_architecture_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = ManifestStore(Path(directory) / "source")
            source.create({"config_digest": "source"})
            config = _resolved_profile_config(
                architecture="conformer", ffn_dim=1536
            )
            fields = profile_architecture_fields(config["model"])
            self.assertEqual(
                fields,
                {
                    "student_arch": "conformer",
                    "student_depth": 6,
                    "student_embed_dim": 384,
                    "student_ffn_dim": 1536,
                    "student_attention_heads": 12,
                },
            )
            profile = {
                **_profile(
                    source,
                    architecture="conformer",
                    ffn_dim=1536,
                    parameters=1000,
                    flops=2000,
                ),
                **fields,
            }
            validated = validate_config_profile(
                profile, expected_arch="conformer"
            )
            self.assertEqual(validated["student_ffn_dim"], 1536)

    def test_selected_controls_inherit_batch_but_e2_owns_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory) / "selected")
            store.create(
                {
                    "resolved_config": {
                        "model": {
                            "student_arch": "transformer",
                            "student_depth": 6,
                            "student_embed_dim": 384,
                            "student_ffn_dim": 3200,
                            "student_attention_heads": 12,
                            "distillation_noise_prob": 0.0,
                        },
                        "criterion": {"distill_loss_type": "historical"},
                        "task": {
                            "data": "/fixed/data",
                            "noise_prob": 0.0,
                            "distillation_noise_prob": 0.0,
                        },
                        "dataset": {"max_tokens": 2000},
                        "optimization": {
                            "max_update": 75000,
                            "update_freq": [8],
                        },
                        "lr_scheduler": {
                            "_name": "polynomial_decay",
                            "warmup_updates": 15000,
                            "total_num_update": 75000,
                        },
                    }
                }
            )
            selection = {
                "selected_manifest": str(store.path),
                "materialized_overrides": [],
                "overrides_by_experiment": {},
            }
            e2 = inherited_hydra_overrides(
                selection, destination_experiment="e2_selected_noisy"
            )
            self.assertIn("dataset.max_tokens=2000", e2)
            self.assertIn("optimization.update_freq=[8]", e2)
            self.assertNotIn("task.noise_prob=0.0", e2)
            s2 = inherited_hydra_overrides(
                selection, destination_experiment="s2_optional_two_stage"
            )
            self.assertIn("task.noise_prob=0.0", s2)
            self.assertNotIn("optimization.max_update=75000", s2)


if __name__ == "__main__":
    unittest.main()
