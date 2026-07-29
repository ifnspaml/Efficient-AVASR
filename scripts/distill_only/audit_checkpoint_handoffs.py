#!/usr/bin/env python3
"""Audit distillation -> export -> fine-tuning checkpoint handoffs."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
FAIRSEQ_ROOT = REPO_ROOT / "fairseq"
for source_root in (REPO_ROOT, FAIRSEQ_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

# Import Fairseq before torch.load so pickle can resolve Fairseq classes without
# triggering the package's historical circular import path.
import fairseq  # noqa: E402,F401
import torch  # noqa: E402

from chapter3_distill_only.checkpoint_audit import (  # noqa: E402
    ENCODER_STUDENT_PREFIX,
    FINETUNE_BACKBONE_PREFIX,
    checkpoint_num_updates,
    compare_mapped_tensors,
    nested_get,
    projection_tensors,
    summarize_training_tail,
    summarize_validation_log,
)
from chapter3_distill_only.manifest import atomic_write_json, utc_now  # noqa: E402


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _git_show(commit: str, relative_path: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative_path}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.stdout if process.returncode == 0 else ""


def _manifest_wer(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    validation = nested_get(
        manifest, "measurements", "wer", "validation", default={}
    )
    output = {}
    for name, measurement in validation.items():
        if isinstance(measurement, Mapping):
            output[name] = measurement.get("value")
    return output


def _freeze_updates(checkpoint: Mapping[str, Any]) -> int:
    cfg = checkpoint.get("cfg")
    if isinstance(cfg, Mapping):
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, Mapping):
            return int(model_cfg.get("freeze_finetune_updates", 0))
    return 0


def audit_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoder_path = run_dir / "encoder" / "checkpoints" / "checkpoint_last.pt"
    export_path = run_dir / "export" / "student.pt"
    finetune_path = (
        run_dir / "finetune" / "checkpoints" / "checkpoint_best.pt"
    )
    finetune_log = run_dir / "logs" / "finetune.log"

    encoder = _load_checkpoint(encoder_path)
    exported = _load_checkpoint(export_path)
    exported_names = sorted(exported["model"])
    distill_to_export = compare_mapped_tensors(
        encoder["model"],
        exported["model"],
        source_prefix=ENCODER_STUDENT_PREFIX,
        names=exported_names,
    )
    encoder_update = checkpoint_num_updates(encoder)
    del encoder
    gc.collect()

    finetune = _load_checkpoint(finetune_path)
    best_update = checkpoint_num_updates(finetune)
    freeze_updates = _freeze_updates(finetune)
    export_to_finetune = compare_mapped_tensors(
        exported["model"],
        finetune["model"],
        target_prefix=FINETUNE_BACKBONE_PREFIX,
        names=exported_names,
        # Fine-tuning intentionally reconstructs mask_emb rather than loading
        # it; extract_finetune does not use it in this seq2seq path.
        excluded_names=("mask_emb",),
    )
    projection = projection_tensors(finetune["model"])
    del finetune
    del exported
    gc.collect()

    implementation_commit = nested_get(
        manifest,
        "immutable",
        "provenance",
        "implementation",
        "commit",
        default="",
    )
    recorded_source = _git_show(
        implementation_commit, "avhubert/hubert_asr.py"
    )
    recorded_wrapper_has_projection = (
        "HubertEncoderWrapper(encoder_, cfg)" in recorded_source
    )
    best_before_unfreeze = (
        best_update is not None and best_update < freeze_updates
    )

    return {
        "experiment": nested_get(
            manifest, "immutable", "experiment", default=run_dir.parent.name
        ),
        "run_directory": str(run_dir),
        "manifest_state": manifest.get("state"),
        "recorded_implementation_commit": implementation_commit,
        "encoder_final_update": encoder_update,
        "finetune_best_update": best_update,
        "freeze_finetune_updates": freeze_updates,
        "best_checkpoint_precedes_unfreeze": best_before_unfreeze,
        "distillation_to_export": distill_to_export,
        "export_to_finetune_best": export_to_finetune,
        "fine_tuning_projection": {
            "tensors": projection,
            "present": bool(projection),
            "present_in_recorded_implementation": recorded_wrapper_has_projection,
            "stage_provenance_mismatch_proven": bool(projection)
            and not recorded_wrapper_has_projection,
        },
        "fine_tuning_validation": summarize_validation_log(finetune_log),
        "fine_tuning_tail": summarize_training_tail(finetune_log),
        "wer": _manifest_wer(manifest),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "schema_version": "chapter3-checkpoint-handoff-audit/v1",
        "created_at": utc_now(),
        "repository": str(REPO_ROOT),
        "runs": [audit_run(run_dir) for run_dir in args.run_dir],
    }
    report["all_distillation_exports_exact"] = all(
        run["distillation_to_export"]["all_tensors_exact"]
        for run in report["runs"]
    )
    frozen_runs = [
        run for run in report["runs"] if run["best_checkpoint_precedes_unfreeze"]
    ]
    report["all_pre_unfreeze_backbone_parameters_close"] = all(
        run["export_to_finetune_best"]["parameters_close"]
        for run in frozen_runs
    )
    report["pre_unfreeze_run_count"] = len(frozen_runs)
    report["all_pre_unfreeze_batchnorm_buffers_exact"] = all(
        not run["export_to_finetune_best"]["mutable_buffer_mismatches"]
        for run in frozen_runs
    )
    report["all_frozen_backbone_parameters_exact"] = all(
        run["export_to_finetune_best"]["parameters_exact"]
        for run in frozen_runs
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
