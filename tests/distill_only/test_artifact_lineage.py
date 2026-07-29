"""CPU-only tests for derived fine-tuning and evaluation lineage."""

from __future__ import annotations

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from chapter3_distill_only.evaluation import load_evaluation_protocol
from chapter3_distill_only.evaluation_runs import (
    evaluation_immutable,
    execute_evaluation_conditions,
    parse_eval_subsets,
    prepare_evaluation_store,
    resolve_evaluation_phase,
    selected_conditions,
)
from chapter3_distill_only.lineage import (
    LineageError,
    allocate_derived_run,
    derived_finetunes,
    evaluation_directory,
    proposed_derived_run,
    resolve_artifact,
    sanitize_label,
)
from chapter3_distill_only.manifest import (
    RUN_KIND_DERIVED_FINETUNE,
    RUN_KIND_EVALUATION,
    ManifestError,
    ManifestStore,
    manifest_run_kind,
    sha256_file,
)
from chapter3_distill_only.selection import (
    create_selection,
    require_final_selection_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    REPO_ROOT / "scripts" / "distill_only" / "evaluation_protocol_itut.yaml"
)


class ArtifactFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.export = self.root / "export" / "student.pt"
        self.export.parent.mkdir()
        self.export.write_bytes(b"export")
        self.finetune = (
            self.root / "finetune" / "checkpoints" / "checkpoint_best.pt"
        )
        self.finetune.parent.mkdir(parents=True)
        self.finetune.write_bytes(b"finetune")
        self.store = ManifestStore(self.root)
        self.store.create(
            {
                "experiment": "fixture",
                "provenance": {
                    "implementation": {"commit": "a" * 40}
                },
            }
        )
        self.store.transition("encoder_running")
        self.store.transition("encoder_complete")
        self.store.transition(
            "exported",
            values={
                "artifacts": {
                    "exported_checkpoint": {
                        "path": str(self.export),
                        "sha256": sha256_file(self.export),
                    }
                }
            },
        )
        self.store.transition("finetune_running")
        self.store.transition(
            "finetune_complete",
            values={
                "artifacts": {
                    "finetune_checkpoint": {
                        "path": str(self.finetune),
                        "sha256": sha256_file(self.finetune),
                    }
                }
            },
        )

    def add_derived(self, sequence: int = 1) -> tuple[Path, ManifestStore]:
        run = self.root / "reruns" / f"projection-fix_deadbee_{sequence:03d}"
        checkpoint = run / "finetune" / "checkpoints" / "checkpoint_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"derived-{sequence}".encode())
        store = ManifestStore(run)
        store.create(
            {
                "run_kind": RUN_KIND_DERIVED_FINETUNE,
                "lineage": {"sequence": sequence},
                "parent_artifact": {
                    "parent_run_dir": str(self.root),
                },
            }
        )
        store.transition("finetune_running")
        store.transition(
            "finetune_complete",
            values={
                "artifacts": {
                    "finetune_checkpoint": {
                        "path": str(checkpoint),
                        "sha256": sha256_file(checkpoint),
                    }
                }
            },
        )
        return run, store


class ManifestKindTest(unittest.TestCase):
    def test_legacy_root_is_read_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary) / "root")
            before = fixture.store.path.read_bytes()
            manifest = fixture.store.read()
            self.assertEqual(manifest_run_kind(manifest), "root_pipeline")
            self.assertEqual(fixture.store.path.read_bytes(), before)

    def test_kind_specific_lifecycles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = ManifestStore(root / "derived")
            created = derived.create(
                {"run_kind": RUN_KIND_DERIVED_FINETUNE}
            )
            self.assertEqual(created["state"], "exported")
            derived.transition("finetune_running")
            derived.transition("finetune_complete")
            with self.assertRaises(ManifestError):
                derived.transition("validation_complete")

            evaluation = ManifestStore(root / "evaluation")
            self.assertEqual(
                evaluation.create(
                    {"run_kind": RUN_KIND_EVALUATION}
                )["state"],
                "prepared",
            )
            evaluation.transition("evaluation_running")
            evaluation.transition("evaluation_complete")
            with self.assertRaises(ManifestError):
                evaluation.transition("test_complete")


class LineageResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ArtifactFixture(
            Path(self.temporary.name) / "experiment" / "seed_1337"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_original_and_latest_artifacts(self) -> None:
        original_export = resolve_artifact(
            self.fixture.root,
            "original-export",
            artifact_kind="exported_student",
        )
        self.assertEqual(
            original_export.checkpoint_path, self.fixture.export.resolve()
        )
        self.assertEqual(
            original_export.parent_record()["parent_manifest_format"],
            "legacy",
        )
        first, _ = self.fixture.add_derived(1)
        second, _ = self.fixture.add_derived(2)
        latest = resolve_artifact(
            self.fixture.root,
            "latest-finetune",
            artifact_kind="finetune_checkpoint",
        )
        self.assertEqual(latest.run_dir, second.resolve())
        os.utime(first, (2_000_000_000, 2_000_000_000))
        os.utime(second, (1, 1))
        latest_after_mtime_change = resolve_artifact(
            self.fixture.root,
            "latest-finetune",
            artifact_kind="finetune_checkpoint",
        )
        self.assertEqual(latest_after_mtime_change.run_dir, second.resolve())
        self.assertEqual(len(derived_finetunes(self.fixture.root)), 2)

    def test_explicit_ambiguity_type_and_hash_failures(self) -> None:
        self.fixture.add_derived(1)
        with self.assertRaisesRegex(LineageError, "Multiple compatible"):
            resolve_artifact(
                self.fixture.root,
                None,
                artifact_kind="finetune_checkpoint",
            )
        with self.assertRaises(LineageError):
            resolve_artifact(
                self.fixture.root,
                str(self.fixture.export),
                artifact_kind="exported_student",
            )
        with self.assertRaisesRegex(LineageError, "not a fine-tuning"):
            resolve_artifact(
                self.fixture.root,
                "original-export",
                artifact_kind="finetune_checkpoint",
            )
        self.fixture.export.write_bytes(b"changed")
        with self.assertRaisesRegex(LineageError, "hash mismatch"):
            resolve_artifact(
                self.fixture.root,
                "original-export",
                artifact_kind="exported_student",
            )

    def test_locked_allocation_and_path_safety(self) -> None:
        commit = "deadbeef" * 5
        first, first_sequence = allocate_derived_run(
            self.fixture.root,
            run_label="projection fix",
            commit=commit,
        )
        second, second_sequence = allocate_derived_run(
            self.fixture.root,
            run_label="projection fix",
            commit=commit,
        )
        self.assertEqual(first.name, "projection-fix_deadbee_001")
        self.assertEqual(second.name, "projection-fix_deadbee_002")
        self.assertEqual((first_sequence, second_sequence), (1, 2))
        proposed, sequence = proposed_derived_run(
            self.fixture.root,
            run_label="projection fix",
            commit=commit,
        )
        self.assertEqual(sequence, 3)
        self.assertEqual(proposed.name, "projection-fix_deadbee_003")
        self.assertEqual(sanitize_label("../../unsafe name"), "unsafe-name")
        evaluation = evaluation_directory(first, "../../screening")
        self.assertEqual(evaluation.name, "screening")

    def test_concurrent_allocation_is_unique(self) -> None:
        commit = "cafebabe" * 5
        with ThreadPoolExecutor(max_workers=4) as executor:
            allocated = list(
                executor.map(
                    lambda _: allocate_derived_run(
                        self.fixture.root,
                        run_label="parallel",
                        commit=commit,
                    )[0],
                    range(8),
                )
            )
        self.assertEqual(len({path.name for path in allocated}), 8)

    def test_symlinked_output_roots_cannot_escape(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        reruns = self.fixture.root / "reruns"
        if reruns.exists():
            for path in reruns.iterdir():
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
            reruns.rmdir()
        reruns.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(LineageError, "escapes"):
            allocate_derived_run(
                self.fixture.root,
                run_label="unsafe",
                commit="a" * 40,
            )


class EvaluationSelectionTest(unittest.TestCase):
    def test_phase_and_subset_matrix(self) -> None:
        protocol = load_evaluation_protocol(PROTOCOL)
        valid = parse_eval_subsets("valid")
        test = parse_eval_subsets("test")
        both = parse_eval_subsets("valid,test")
        self.assertEqual(resolve_evaluation_phase("auto", valid), "screening")
        self.assertEqual(resolve_evaluation_phase("auto", test), "final")
        self.assertEqual(resolve_evaluation_phase("auto", both), "final")
        self.assertTrue(
            all(
                item.subset == "valid"
                for item in selected_conditions(
                    protocol, phase="final", subsets=valid
                )
            )
        )
        self.assertEqual(
            {item.subset for item in selected_conditions(
                protocol, phase="final", subsets=both
            )},
            {"valid", "test"},
        )
        with self.assertRaises(LineageError):
            resolve_evaluation_phase("screening", test)
        for invalid in ("", "valid,", "valid,valid", "dev"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(LineageError):
                    parse_eval_subsets(invalid)

    def test_multiple_evaluations_and_label_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ArtifactFixture(Path(temporary) / "root")
            run, store = fixture.add_derived(1)
            parent = resolve_artifact(
                fixture.root,
                str(run),
                artifact_kind="finetune_checkpoint",
            )
            parent_before = store.path.read_bytes()
            protocol = load_evaluation_protocol(PROTOCOL)
            conditions = selected_conditions(
                protocol, phase="screening", subsets=("valid",)
            )
            snapshot = {
                "schema_version": "chapter3-source-worktree/v1",
                "repository_root": str(REPO_ROOT),
                "worktree_root": str(REPO_ROOT.parent),
                "worktree_path": str(REPO_ROOT),
                "commit": "a" * 40,
                "tree": "b" * 40,
                "integrity_digest": "c" * 64,
                "created_at": "test",
            }
            first = evaluation_immutable(
                experiment="fixture",
                seed=1337,
                label="screening",
                parent=parent,
                protocol=protocol,
                phase="screening",
                subsets=("valid",),
                conditions=conditions,
                evaluation_seed=1337,
                source_snapshot=snapshot,
            )
            first_store, created = prepare_evaluation_store(
                run,
                label="screening",
                immutable=first,
                allow_existing_source=False,
            )
            self.assertTrue(created)
            repeated, created_again = prepare_evaluation_store(
                run,
                label="screening",
                immutable=first,
                allow_existing_source=False,
            )
            self.assertFalse(created_again)
            self.assertEqual(repeated.path, first_store.path)
            second_store, second_created = prepare_evaluation_store(
                run,
                label="full-valid",
                immutable={
                    **first,
                    "evaluation": {
                        **first["evaluation"],
                        "label": "full-valid",
                    },
                },
                allow_existing_source=False,
            )
            self.assertTrue(second_created)
            self.assertNotEqual(second_store.path, first_store.path)
            conflicting = {
                **first,
                "evaluation": {
                    **first["evaluation"],
                    "subsets": ["test"],
                    "configuration_digest": "different",
                },
            }
            with self.assertRaisesRegex(ManifestError, "different"):
                prepare_evaluation_store(
                    run,
                    label="screening",
                    immutable=conflicting,
                    allow_existing_source=False,
                )
            self.assertEqual(store.path.read_bytes(), parent_before)

            calls = []
            condition = conditions[0]
            result_dir = first_store.path.parent / "results"
            artifact = result_dir / "wer.test"

            def run_condition(item, command, output):
                calls.append((item.name, tuple(command), output))
                output.mkdir(parents=True, exist_ok=True)
                artifact.write_text("WER: 1.0\n", encoding="utf-8")

            def measure_condition(item, output):
                self.assertEqual(item, condition)
                self.assertEqual(output, result_dir)
                return {
                    "artifact": str(artifact),
                    "artifact_sha256": sha256_file(artifact),
                    "value": 1.0,
                }

            execute_evaluation_conditions(
                store=first_store,
                commands=((condition, ("decode",), result_dir),),
                run_condition=run_condition,
                measure_condition=measure_condition,
            )
            self.assertEqual(len(calls), 1)
            babble = result_dir / "wer.babble"
            babble.write_text("WER: 2.0\n", encoding="utf-8")
            first_store.update(
                {
                    "measurements": {
                        "wer": {
                            "valid": {
                                "babble_0db": {
                                    "artifact": str(babble),
                                    "artifact_sha256": sha256_file(babble),
                                    "value": 2.0,
                                }
                            }
                        }
                    }
                }
            )
            selection_path = Path(temporary) / "final.json"
            selection = create_selection(
                selection_path,
                kind="final",
                candidates=(first_store.path,),
                selected=first_store.path,
                metric="validation_clean",
                rationale="test",
                approver="unit-test",
            )
            identity = selection["selected_finetune_artifact"]
            self.assertEqual(identity["manifest"], str(store.path.resolve()))
            require_final_selection_artifact(
                selection_path,
                finetune_manifest=store.path,
                checkpoint_sha256=parent.checkpoint_sha256,
            )
            self.assertEqual(store.path.read_bytes(), parent_before)
            execute_evaluation_conditions(
                store=first_store,
                commands=((condition, ("decode",), result_dir),),
                run_condition=run_condition,
                measure_condition=measure_condition,
            )
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
