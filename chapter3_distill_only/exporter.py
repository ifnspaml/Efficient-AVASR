"""Export a distillation wrapper checkpoint as a standalone student checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
from fairseq import checkpoint_utils

from .manifest import atomic_write_json, sha256_file, utc_now
from .profiling import parameter_partitions
from .model import AVHubertDistillOnly
from .student import Chapter3AVHubertModel


class ExportError(RuntimeError):
    """Raised when a checkpoint cannot be exported or verified."""


def _load_wrapper(checkpoint: Path) -> AVHubertDistillOnly:
    models, _ = checkpoint_utils.load_model_ensemble(
        [str(checkpoint)], strict=True
    )
    if len(models) != 1 or not isinstance(models[0], AVHubertDistillOnly):
        names = [type(model).__name__ for model in models]
        raise ExportError(
            "Expected one AVHubertDistillOnly model in "
            f"{checkpoint}, received {names}"
        )
    return models[0]


def _complete_payload(
    wrapper: AVHubertDistillOnly, source_checkpoint: Path
) -> Dict[str, Any]:
    payload = copy.deepcopy(wrapper.get_export_state())
    payload.setdefault("criterion", None)
    payload.setdefault("optimizer_history", [])
    payload.setdefault("last_optimizer_state", None)
    payload.setdefault("args", None)
    payload["extra_state"].setdefault("chapter3_distill_only", {})
    payload["extra_state"]["chapter3_distill_only"].update(
        {
            "exported_at": utc_now(),
            "source_distillation_checkpoint": str(
                source_checkpoint.resolve()
            ),
            "source_distillation_checkpoint_sha256": sha256_file(
                source_checkpoint
            ),
            "initialization_report": copy.deepcopy(
                wrapper.initialization_report
            ),
            "parameter_partitions": parameter_partitions(wrapper),
        }
    )
    return payload


def _atomic_torch_save(payload: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def verify_export(checkpoint: Path) -> Dict[str, Any]:
    state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint))
    models, _ = checkpoint_utils.load_model_ensemble(
        [str(checkpoint)], strict=True
    )
    if len(models) != 1 or not isinstance(models[0], Chapter3AVHubertModel):
        names = [type(model).__name__ for model in models]
        raise ExportError(
            "Export did not reload as Chapter3AVHubertModel: "
            f"{names}"
        )
    model = models[0]
    from avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoder

    task_cfg = state["cfg"].task
    downstream_cfg = AVHubertAsrConfig(
        w2v_path=str(checkpoint),
        normalize=bool(task_cfg.normalize),
        data=str(task_cfg.data),
        apply_mask=False,
        no_pretrained_weights=False,
    )
    downstream_encoder = HubertEncoder(downstream_cfg)
    downstream_model = downstream_encoder.w2v_model
    if not isinstance(downstream_model, Chapter3AVHubertModel):
        raise ExportError(
            "Existing HubertEncoder did not load the native student: "
            f"{type(downstream_model).__name__}"
        )
    return {
        "verified": True,
        "model_class": type(model).__name__,
        "student_arch": model.cfg.student_arch,
        "encoder_depth": len(model.encoder.layers),
        "encoder_embed_dim": model.encoder_embed_dim,
        "checkpoint_sha256": sha256_file(checkpoint),
        "deployed_backbone_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "hubert_encoder_reload_verified": True,
        "hubert_encoder_model_class": type(downstream_model).__name__,
        "hubert_encoder_depth": len(downstream_model.encoder.layers),
    }


def export_checkpoint(
    source_checkpoint: Path, output: Path, *, verify: bool = True
) -> Dict[str, Any]:
    if not source_checkpoint.is_file():
        raise ExportError(
            f"Distillation checkpoint does not exist: {source_checkpoint}"
        )
    wrapper = _load_wrapper(source_checkpoint)
    payload = _complete_payload(wrapper, source_checkpoint)
    _atomic_torch_save(payload, output)
    result = {
        "source": str(source_checkpoint.resolve()),
        "output": str(output.resolve()),
        "checkpoint_sha256": sha256_file(output),
        "parameter_partitions": parameter_partitions(wrapper),
        "initialization_report": wrapper.initialization_report,
    }
    if verify:
        result["verification"] = verify_export(output)
    atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distilled-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = export_checkpoint(
        args.distilled_checkpoint, args.output, verify=args.verify
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
