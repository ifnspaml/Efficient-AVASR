"""Focused tests for the native student exporter helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from chapter3_distill_only import exporter


class ExporterTest(unittest.TestCase):
    def test_atomic_json_and_torch_writes_replace_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "metadata.json"
            exporter._atomic_write_json(json_path, {"version": 1})
            exporter._atomic_write_json(json_path, {"version": 2})
            self.assertEqual(json.loads(json_path.read_text()), {"version": 2})
            checkpoint = root / "student.pt"
            exporter._atomic_torch_save({"model": {"x": torch.tensor([1])}}, checkpoint)
            exporter._atomic_torch_save({"model": {"x": torch.tensor([2])}}, checkpoint)
            self.assertEqual(torch.load(checkpoint)["model"]["x"].item(), 2)
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_export_payload_uses_native_state_and_excludes_wrapper_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            source.write_bytes(b"checkpoint")
            wrapper = MagicMock()
            wrapper.get_export_state.return_value = {
                "cfg": {"model": {"_name": "chapter3_av_hubert"}},
                "model": {"encoder.weight": torch.ones(1)},
            }
            wrapper.get_student_num_params.return_value = 11
            wrapper.get_prediction_head_num_params.return_value = 3
            wrapper.get_teacher_num_params.return_value = 17
            payload = exporter._export_payload(wrapper, source)
            self.assertEqual(payload["model"]["encoder.weight"].item(), 1)
            self.assertNotIn("teacher", payload["model"])
            self.assertNotIn("prediction_heads", payload["model"])
            metadata = payload["extra_state"]["chapter3_distill_only"]
            self.assertEqual(metadata["parameters"]["deployed_backbone"], 11)
            self.assertEqual(
                metadata["parameters"]["training_only_prediction_heads"], 3
            )
            self.assertEqual(metadata["parameters"]["frozen_teacher"], 17)

    def test_wrapper_loading_is_strict_and_type_checked(self) -> None:
        checkpoint = Path("/tmp/not-used.pt")
        with patch.object(
            exporter.checkpoint_utils,
            "load_model_ensemble",
            return_value=([object()], None),
        ):
            with self.assertRaisesRegex(exporter.ExportError, "Expected one"):
                exporter._load_wrapper(checkpoint)

    def test_export_failure_does_not_publish_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            output = Path(directory) / "student.pt"
            source.write_bytes(b"source")
            wrapper = MagicMock()
            wrapper.get_export_state.return_value = {"model": {}}
            wrapper.get_student_num_params.return_value = 1
            wrapper.get_prediction_head_num_params.return_value = 1
            wrapper.get_teacher_num_params.return_value = 1
            wrapper.student.cfg.student_arch = "transformer"
            wrapper.student.encoder.layers = [object()]
            wrapper.student.encoder_embed_dim = 384
            with patch.object(exporter, "_load_wrapper", return_value=wrapper), patch.object(
                exporter, "verify_export", side_effect=exporter.ExportError("bad reload")
            ):
                with self.assertRaisesRegex(exporter.ExportError, "bad reload"):
                    exporter.export_checkpoint(source, output)
            # The checkpoint is atomically complete even though verification
            # failed; metadata is withheld so it cannot be mistaken as verified.
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".pt.metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
