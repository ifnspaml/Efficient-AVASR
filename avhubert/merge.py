import logging
import sys
from argparse import ArgumentParser
from typing import Any, Dict, Optional
from pathlib import Path

import torch

from fairseq.checkpoint_utils import load_model_ensemble

from hubert import AVHubertModel
from hubert_distill import AVHubertDistill


_logger = logging.getLogger(__name__)


def merge_from_ckpt(distilled_ckpt: Optional[str], original_ckpt: str, merge_type: str) -> Dict[str, Any]:
    ckpt = original_ckpt if distilled_ckpt is None else distilled_ckpt
    ensemble, _ = load_model_ensemble([ckpt], strict=False)
    model = ensemble[0] if distilled_ckpt is None else ensemble[0].student

    model.merge(1.0, merge_type)

    # Update original checkpoint state with merged information
    original_state = torch.load(original_ckpt, map_location=torch.device("cpu"))
    original_state["cfg"]["model"]["encoder_merge_type"] = str(merge_type)
    original_state["model"] = model.state_dict()
    for i in range (0, 12):
        if "encoder_use_attention" in original_state["cfg"]["model"] and not original_state["cfg"]["model"]["encoder_use_attention"][i]:
            continue
        if merge_type != "KV": # QKV QK QV
            original_state["model"][f"encoder.layers.{i}.self_attn.h_proj.weight"] = original_state["model"][f"encoder.layers.{i}.self_attn.q_proj.weight"]
            original_state["model"][f"encoder.layers.{i}.self_attn.h_proj.bias"] = original_state["model"][f"encoder.layers.{i}.self_attn.q_proj.bias"]
        else: # KV
            original_state["model"][f"encoder.layers.{i}.self_attn.h_proj.weight"] = original_state["model"][f"encoder.layers.{i}.self_attn.k_proj.weight"]
            original_state["model"][f"encoder.layers.{i}.self_attn.h_proj.bias"] = original_state["model"][f"encoder.layers.{i}.self_attn.k_proj.bias"]

        if merge_type != "KV":
            del original_state["model"][f"encoder.layers.{i}.self_attn.q_proj.weight"]
            del original_state["model"][f"encoder.layers.{i}.self_attn.q_proj.bias"]
        if merge_type != "QK":
            del original_state["model"][f"encoder.layers.{i}.self_attn.v_proj.weight"]
            del original_state["model"][f"encoder.layers.{i}.self_attn.v_proj.bias"]
        if merge_type != "QV":
            del original_state["model"][f"encoder.layers.{i}.self_attn.k_proj.weight"]
            del original_state["model"][f"encoder.layers.{i}.self_attn.k_proj.bias"]

    if distilled_ckpt is not None:
        original_state["extra_state"]["distill_linear_projs"] = ensemble[0].distill_linear_projs.state_dict()

    return original_state


def _load_merged_model(merged_ckpt: str) -> AVHubertModel:
    esemble, _ = load_model_ensemble([merged_ckpt], strict=False)
    return esemble[0]


def _parse_args():
    parser = ArgumentParser(description="Merge and save model.")
    parser.add_argument(
        "--distilled_ckpt",
        type=str,
        default=None,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--original_ckpt",
        type=str,
        help="Path to the original checkpoint."
    )
    parser.add_argument(
        "--merge_type",
        type=str,
        help="Type of attention head merging. Choose from [QKV, QK, QV, KV]."
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="WARNING",
        help="Set log level"
    )
    return parser.parse_args()


def _init_logger(log_level: str) -> None:
    try:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=log_level,
        )
    except ValueError:
        _logger.error(f"Invalid log level: {args.log_level}")
        sys.exit(1)


if __name__ == "__main__":
    args = _parse_args()
    _init_logger(args.log_level)

    if args.distilled_ckpt is None:
        out_path = Path(args.original_ckpt).parent / f"merged_{Path(args.original_ckpt).stem}.pt"
    else:
        out_path = Path(args.distilled_ckpt).parent / f"merged_{Path(args.distilled_ckpt).stem}.pt"
    
    torch.save(
        merge_from_ckpt(args.distilled_ckpt, args.original_ckpt, args.merge_type),
        out_path,
    )

    # Check if loading from ckpt works
    model = _load_merged_model(str(out_path))

    _logger.info(f"Model:\n{model}")
    _logger.info(f"Number of trainable parameters: {model.get_num_params()}")
    _logger.info(f"Successfully saved merged model weights and config to: {out_path}")
