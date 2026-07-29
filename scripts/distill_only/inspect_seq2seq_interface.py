#!/usr/bin/env python3
"""Inspect exported students and validate their seq2seq interface dimensions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "fairseq"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from avhubert.hubert_asr import (  # noqa: E402
    AVHubertSeq2SeqConfig,
    HubertEncoderWrapper,
)


class _InspectionBackbone(nn.Module):
    def __init__(self, encoder_dim: int) -> None:
        super().__init__()
        self.encoder = SimpleNamespace(embedding_dim=encoder_dim)

    def extract_finetune(self, **kwargs):  # pragma: no cover - not executed
        raise RuntimeError("inspection does not execute the backbone")


def _seed_run(experiment_dir: Path) -> Path:
    if (experiment_dir / "manifest.v1.json").is_file():
        return experiment_dir
    candidates = sorted(
        path
        for path in experiment_dir.glob("seed_*")
        if (path / "manifest.v1.json").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one manifest-bearing run below {experiment_dir}"
        )
    return candidates[0]


def inspect_run(experiment_dir: Path) -> Dict[str, Any]:
    run_dir = _seed_run(experiment_dir.resolve())
    manifest = json.loads(
        (run_dir / "manifest.v1.json").read_text(encoding="utf-8")
    )
    metadata_path = run_dir / "export" / "student.pt.metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    encoder_dim = (
        metadata.get("verification", {}).get("encoder_embed_dim")
        or manifest["immutable"]["resolved_config"]["model"].get(
            "student_embed_dim"
        )
    )
    if encoder_dim is None:
        raise ValueError(f"cannot infer encoder output dimension for {run_dir}")
    encoder_dim = int(encoder_dim)

    finetune_config = run_dir / "finetune" / ".hydra" / "config.yaml"
    decoder_dim = None
    if finetune_config.is_file():
        resolved = yaml.safe_load(finetune_config.read_text(encoding="utf-8"))
        decoder_dim = (resolved or {}).get("model", {}).get(
            "decoder_embed_dim"
        )
    if decoder_dim is None:
        decoder_dim = AVHubertSeq2SeqConfig.__dataclass_fields__[
            "decoder_embed_dim"
        ].default
    decoder_dim = int(decoder_dim)

    wrapper = HubertEncoderWrapper(
        _InspectionBackbone(encoder_dim),
        SimpleNamespace(decoder_embed_dim=decoder_dim),
    )
    projection_needed = encoder_dim != decoder_dim
    if projection_needed != (wrapper.proj is not None):
        raise AssertionError("constructed interface disagrees with dimensions")
    projection_parameters = (
        sum(parameter.numel() for parameter in wrapper.proj.parameters())
        if wrapper.proj is not None
        else 0
    )
    return {
        "experiment": manifest["immutable"]["experiment"],
        "run_directory": str(run_dir),
        "encoder_output_dim": encoder_dim,
        "decoder_embedding_dim": decoder_dim,
        "projection_needed": projection_needed,
        "projection_weight_shape": (
            list(wrapper.proj.weight.shape)
            if wrapper.proj is not None
            else None
        ),
        "projection_parameter_count": projection_parameters,
        "supported": True,
        "evidence": {
            "manifest": str(run_dir / "manifest.v1.json"),
            "export_metadata": (
                str(metadata_path) if metadata_path.is_file() else None
            ),
            "finetune_config": (
                str(finetune_config) if finetune_config.is_file() else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "schema_version": "chapter3-seq2seq-interface-inspection/v1",
        "runs": [inspect_run(path) for path in args.experiment_dir],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
