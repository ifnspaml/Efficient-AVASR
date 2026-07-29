"""Focused launcher tests for artifact-derived execution."""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from chapter3_distill_only.selection import (
    SelectionError,
    require_final_selection_artifact,
)
from chapter3_distill_only.evaluation import load_evaluation_protocol
from chapter3_distill_only.lineage import ArtifactRef
from chapter3_distill_only.manifest import ManifestStore, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "distill_only" / "launch.py"
SPEC = importlib.util.spec_from_file_location("artifact_launcher", LAUNCHER)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class StageExpressionTest(unittest.TestCase):
    def test_supported_stage_expressions(self) -> None:
        for value in (
            "prepare",
            "encoder",
            "stage1",
            "stage2",
            "export",
            "finetune",
            "validate",
            "test",
            "evaluate",
            "all",
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    launcher.parse_stage_expression(value), (value,)
                )
        self.assertEqual(
            launcher.parse_stage_expression("finetune,evaluate"),
            ("finetune", "evaluate"),
        )

    def test_invalid_expressions_are_rejected(self) -> None:
        for value in (
            "evaluate,finetune",
            "finetune,finetune",
            "export,evaluate",
            "test,finetune",
            "unknown",
            "finetune,",
            ",evaluate",
        ):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    launcher.parse_stage_expression(value)


class ExplicitCommandInputTest(unittest.TestCase):
    def _args(self, root: Path) -> Namespace:
        return Namespace(
            run_dir=root / "derived",
            data=root / "data",
            tokenizer=root / "tokenizer.model",
            noise_root=root / "noise",
            gpus=1,
            finetune_update_freq=8,
            workers=24,
            seed=1337,
            fairseq_train="fairseq-hydra-train",
            source_root=root / "source",
        )

    def test_finetune_command_uses_explicit_parent_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            config = source / "avhubert" / "conf" / "av-finetune"
            config.mkdir(parents=True)
            student = root / "legacy" / "export" / "student.pt"
            student.parent.mkdir(parents=True)
            student.write_bytes(b"student")
            output = root / "derived" / "finetune"
            args = self._args(root)
            command, returned_output = launcher._finetune_command(
                args,
                exported_checkpoint=student,
                output_dir=output,
                source_root=source,
            )
            self.assertEqual(returned_output, output.resolve())
            self.assertIn(f"model.w2v_path={student.resolve()}", command)
            self.assertNotIn(
                f"model.w2v_path={args.run_dir / 'export' / 'student.pt'}",
                command,
            )

    def test_root_finetune_keeps_existing_export_convention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            student = args.run_dir / "export" / "student.pt"
            student.parent.mkdir(parents=True)
            student.write_bytes(b"student")
            with mock.patch.object(
                launcher, "_source_root", return_value=root / "source"
            ):
                command, output = launcher._finetune_command(args)
            self.assertIn(f"model.w2v_path={student}", command)
            self.assertEqual(output, args.run_dir / "finetune")

    def test_pinned_launcher_preserves_artifact_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            worktree = root / "worktree"
            protocol = development / "scripts" / "protocol.yaml"
            args = Namespace(
                experiment="c3a_t12_historical_heads",
                seed=1337,
                run_dir=root / "derived",
                output_root=root / "outputs",
                source_worktree_root=root / "worktrees",
                teacher=root / "teacher.pt",
                data=root / "data",
                tokenizer=root / "tokenizer.model",
                noise_root=root / "noise",
                evaluation_protocol=protocol,
                evaluation_seed=1337,
                final_selection=root / "final.json",
                gpus=1,
                workers=24,
                max_tokens=4000,
                update_freq=4,
                finetune_update_freq=8,
                fairseq_train="fairseq-hydra-train",
                from_selection=None,
                derive_from="original-export",
                run_label="projection-fix",
                evaluation_label="itut-screening",
                eval_subsets="valid",
                evaluation_phase="auto",
                override=[],
            )
            snapshot = {
                "repository_root": str(development),
                "worktree_path": str(worktree),
            }
            command = launcher._pinned_launcher_command(
                args,
                stage="finetune,evaluate",
                source_snapshot=snapshot,
            )
            joined = " ".join(map(str, command))
            self.assertIn("--derive-from original-export", joined)
            self.assertIn("--run-label projection-fix", joined)
            self.assertIn("--evaluation-label itut-screening", joined)
            self.assertIn("--eval-subsets valid", joined)
            self.assertIn(
                str((worktree / "scripts" / "protocol.yaml").resolve()),
                command,
            )

    def test_runtime_overlay_precedes_pinned_source_on_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worktree = root / "worktree"
            runtime = root / "run" / "source" / "runtime" / "fairseq"
            snapshot = {
                "worktree_path": str(worktree),
                "runtime_artifacts": {
                    "pythonpath_root": str(runtime),
                },
            }
            self.assertEqual(
                launcher._snapshot_pythonpath(snapshot).split(":"),
                [
                    str(runtime.resolve()),
                    str(worktree.resolve()),
                    str((worktree / "fairseq").resolve()),
                ],
            )


class SelectionIdentityTest(unittest.TestCase):
    def test_final_selection_binds_manifest_and_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "rerun" / "manifest.v1.json"
            manifest.parent.mkdir(parents=True)
            record = {
                "kind": "final",
                "selected_finetune_artifact": {
                    "manifest": str(manifest),
                    "checkpoint_sha256": "abc",
                },
            }
            with mock.patch(
                "chapter3_distill_only.selection.read_selection",
                return_value=record,
            ):
                accepted = require_final_selection_artifact(
                    root / "final.json",
                    finetune_manifest=manifest,
                    checkpoint_sha256="abc",
                )
                self.assertIs(accepted, record)
                with self.assertRaises(SelectionError):
                    require_final_selection_artifact(
                        root / "final.json",
                        finetune_manifest=manifest,
                        checkpoint_sha256="different",
                    )
                with self.assertRaises(SelectionError):
                    require_final_selection_artifact(
                        root / "final.json",
                        finetune_manifest=root / "another",
                        checkpoint_sha256="abc",
                    )


class ArtifactDryRunTest(unittest.TestCase):
    def test_dry_run_plans_without_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "root" / "export" / "student.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"student")
            manifest = root / "root" / "manifest.v1.json"
            ManifestStore(manifest).create({"experiment": "fixture"})
            parent = ArtifactRef(
                kind="exported_student",
                checkpoint_path=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                run_dir=checkpoint.parents[1],
                manifest_path=manifest,
                manifest_sha256=sha256_file(manifest),
                run_kind="root_pipeline",
                state="exported",
                sequence=0,
                original_implementation_commit="a" * 40,
            )
            derived = root / "root" / "reruns" / "projection-fix_deadbee_001"
            args = Namespace(
                experiment="c3a_t12_historical_heads",
                stage_sequence=("finetune", "evaluate"),
                run_label="projection-fix",
                evaluation_label="screening",
                evaluation_phase_resolved="screening",
                eval_subsets_tuple=("valid",),
                evaluation_protocol_obj=load_evaluation_protocol(
                    REPO_ROOT
                    / "scripts"
                    / "distill_only"
                    / "evaluation_protocol_itut.yaml"
                ),
                source_worktree_root=root / "worktrees",
                run_dir=derived,
                data=root / "data",
                tokenizer=root / "tokenizer.model",
                noise_root=root / "noise",
                gpus=1,
                finetune_update_freq=8,
                workers=24,
                seed=1337,
                fairseq_train="fairseq-hydra-train",
                source_root=REPO_ROOT,
            )
            source = {
                "commit": "deadbee" + "0" * 33,
                "tree": "b" * 40,
            }
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = launcher._dry_run_artifact_request(
                    args,
                    root_run=parent.run_dir,
                    parent=parent,
                    derived_run=derived,
                    source=source,
                )
            self.assertEqual(result, 0)
            self.assertIn('"creates_files": false', output.getvalue())
            self.assertIn(str(checkpoint), output.getvalue())
            self.assertFalse(derived.exists())


if __name__ == "__main__":
    unittest.main()
