"""Tests for immutable run manifests and scientific selection gates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chapter3_distill_only.manifest import (
    LIFECYCLE,
    ManifestError,
    ManifestStore,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from chapter3_distill_only.profiling import select_conformer_ffn_match
from chapter3_distill_only.selection import (
    SelectionError,
    create_selection,
    freeze_selected_manifest,
    inherited_hydra_overrides,
    require_final_selection,
    read_selection,
)


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
            candidate_manifest.create({"experiment": "C2"})
            _advance_to_validation(candidate_manifest)

            target_path = root / "transformer.json"
            conformer_path = root / "conformer_1536.json"
            atomic_write_json(
                target_path,
                {
                    "deployed_backbone_parameters": 1000,
                    "flops": 2000,
                },
            )
            atomic_write_json(
                conformer_path,
                {
                    "deployed_backbone_parameters": 1010,
                    "flops": 1990,
                },
            )
            candidate_profile = {
                "deployed_backbone_parameters": 1010,
                "flops": 1990,
                "student_ffn_dim": 1536,
                "profile_path": str(conformer_path.resolve()),
                "profile_sha256": sha256_file(conformer_path),
            }
            match = select_conformer_ffn_match(
                {
                    "deployed_backbone_parameters": 1000,
                    "flops": 2000,
                },
                (candidate_profile,),
            )
            match.update(
                {
                    "schema_version": "chapter3-conformer-match/v1",
                    "created_at": utc_now(),
                    "target_profile_path": str(target_path.resolve()),
                    "target_profile_sha256": sha256_file(target_path),
                }
            )
            match_path = root / "conformer_match.json"
            atomic_write_json(match_path, match)

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
            conformer_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SelectionError):
                read_selection(selection_path)

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
