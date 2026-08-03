"""Export a distillation wrapper checkpoint as a native AV-HuBERT student."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from fairseq import checkpoint_utils

from .model import AVHubertDistillOnly
from .student import Chapter3AVHubertModel


class ExportError(RuntimeError):
    """Raised when a checkpoint cannot be exported or verified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(payload: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_wrapper(checkpoint: Path) -> AVHubertDistillOnly:
    models, _ = checkpoint_utils.load_model_ensemble(
        [str(checkpoint)], strict=True
    )
    if len(models) != 1 or not isinstance(models[0], AVHubertDistillOnly):
        names = [type(model).__name__ for model in models]
        raise ExportError(
            f"Expected one AVHubertDistillOnly in {checkpoint}, got {names}"
        )
    return models[0]


def _parameter_counts(wrapper: AVHubertDistillOnly) -> Dict[str, int]:
    student = wrapper.get_student_num_params()
    heads = wrapper.get_prediction_head_num_params()
    return {
        "deployed_backbone": student,
        "training_only_prediction_heads": heads,
        "trainable_student_and_heads": student + heads,
        "frozen_teacher": wrapper.get_teacher_num_params(),
    }


def _expected_student(wrapper: AVHubertDistillOnly) -> Dict[str, Any]:
    return {
        "student_arch": wrapper.student.cfg.student_arch,
        "encoder_depth": len(wrapper.student.encoder.layers),
        "encoder_embed_dim": wrapper.student.encoder_embed_dim,
    }


def _export_payload(
    wrapper: AVHubertDistillOnly,
    source_checkpoint: Path,
    include_digests: bool = True,
) -> Dict[str, Any]:
    payload = copy.deepcopy(wrapper.get_export_state())
    payload.setdefault("criterion", None)
    payload.setdefault("optimizer_history", [])
    payload.setdefault("last_optimizer_state", None)
    payload.setdefault("args", None)
    metadata = payload.setdefault("extra_state", {}).setdefault(
        "chapter3_distill_only", {}
    )
    metadata.update(
        {
            "exported_at": _utc_now(),
            "source_distillation_checkpoint": str(source_checkpoint.resolve()),
            "parameters": _parameter_counts(wrapper),
        }
    )
    if include_digests:
        metadata["source_distillation_checkpoint_sha256"] = _sha256_file(
            source_checkpoint
        )
    return payload


def verify_export(
    checkpoint: Path,
    expected: Optional[Mapping[str, Any]] = None,
    include_digest: bool = True,
) -> Dict[str, Any]:
    state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint))
    models, _ = checkpoint_utils.load_model_ensemble(
        [str(checkpoint)], strict=True
    )
    if len(models) != 1 or not isinstance(models[0], Chapter3AVHubertModel):
        names = [type(model).__name__ for model in models]
        raise ExportError(
            f"Export did not reload as Chapter3AVHubertModel: {names}"
        )
    model = models[0]
    report = {
        "verified": True,
        "model_class": type(model).__name__,
        "student_arch": model.cfg.student_arch,
        "encoder_depth": len(model.encoder.layers),
        "encoder_embed_dim": model.encoder_embed_dim,
    }
    if include_digest:
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
    if expected is not None:
        for key in ("student_arch", "encoder_depth", "encoder_embed_dim"):
            if report[key] != expected[key]:
                raise ExportError(
                    f"Exported {key} mismatch: expected {expected[key]!r}, "
                    f"got {report[key]!r}"
                )

    from avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoder

    task_cfg = state["cfg"].task
    downstream_cfg = AVHubertAsrConfig(
        w2v_path=str(checkpoint),
        normalize=bool(task_cfg.normalize),
        data=str(task_cfg.data),
        apply_mask=False,
        no_pretrained_weights=False,
    )
    downstream_model = HubertEncoder(downstream_cfg).w2v_model
    if not isinstance(downstream_model, Chapter3AVHubertModel):
        raise ExportError(
            "Existing HubertEncoder did not load the native student: "
            f"{type(downstream_model).__name__}"
        )
    if len(downstream_model.encoder.layers) != report["encoder_depth"]:
        raise ExportError("HubertEncoder reload changed the student depth")
    if downstream_model.encoder_embed_dim != report["encoder_embed_dim"]:
        raise ExportError("HubertEncoder reload changed the student dimension")
    report["hubert_encoder_reload_verified"] = True
    return report


def export_checkpoint(
    source_checkpoint: Path,
    output: Path,
    include_digests: bool = True,
) -> Dict[str, Any]:
    if not source_checkpoint.is_file():
        raise ExportError(
            f"Distillation checkpoint does not exist: {source_checkpoint}"
        )
    wrapper = _load_wrapper(source_checkpoint)
    expected = _expected_student(wrapper)
    parameters = _parameter_counts(wrapper)
    _atomic_torch_save(
        _export_payload(wrapper, source_checkpoint, include_digests), output
    )
    result = {
        "schema_version": "chapter3-student-export/v1",
        "exported_at": _utc_now(),
        "source": str(source_checkpoint.resolve()),
        "output": str(output.resolve()),
        "parameters": parameters,
        "verification": verify_export(
            output, expected, include_digest=include_digests
        ),
    }
    if include_digests:
        result["source_sha256"] = _sha256_file(source_checkpoint)
        result["checkpoint_sha256"] = _sha256_file(output)
    _atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--distilled-checkpoint", type=Path)
    source.add_argument("--verify-existing", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-digest-metadata",
        action="store_true",
        help="perform structural verification without calculating SHA metadata",
    )
    args = parser.parse_args()
    if args.distilled_checkpoint is not None and args.output is None:
        parser.error("--output is required with --distilled-checkpoint")
    if args.verify_existing is not None and args.output is not None:
        parser.error("--output cannot be used with --verify-existing")
    return args


def main() -> None:
    args = _parse_args()
    if args.verify_existing is not None:
        result = verify_export(
            args.verify_existing, include_digest=not args.no_digest_metadata
        )
    else:
        result = export_checkpoint(
            args.distilled_checkpoint,
            args.output,
            include_digests=not args.no_digest_metadata,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
