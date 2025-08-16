import logging
import sys
from argparse import ArgumentParser
from typing import Dict, Any
from pathlib import Path

import torch

from fairseq.checkpoint_utils import load_model_ensemble

from hubert import AVHubertModel
from hubert_distill import AVHubertDistill


_logger = logging.getLogger(__name__)


def prune_from_ckpt(distilled_ckpt: str, original_ckpt: str) -> Dict[str, Any]:
    ensemble, _ = load_model_ensemble([distilled_ckpt])
    distill_model = ensemble[0]
    model = distill_model.student

    resnet_config, encoder_use_attention, encoder_use_ffn, encoder_attention_heads, encoder_ffn_embed_dim = model.prune()

    pruning_cfg = {
      "resnet_prune_conv_channels": False,
      "encoder_prune_attention_heads": False,
      "encoder_prune_attention_layer": False,
      "encoder_prune_ffn_intermediate": False,
      "encoder_prune_ffn_layer": False,
      "encoder_use_attention": encoder_use_attention,
      "encoder_attention_heads_detailed": encoder_attention_heads,
      "encoder_use_ffn": encoder_use_ffn,
      "encoder_ffn_embed_dim_detailed": encoder_ffn_embed_dim,
      "resnet_layers_detailed": resnet_config,
    }
    #Update original checkpoint state with pruned information
    original_state = torch.load(original_ckpt, map_location=torch.device("cpu"))
    original_state["model"] = model.state_dict()
    original_state["cfg"]["model"].update(pruning_cfg)
    original_state["extra_state"]["distill_linear_projs"] = distill_model.distill_linear_projs.state_dict()

    return original_state


def _load_pruned_model(pruned_ckpt: str) -> AVHubertModel:
    esemble, _ = load_model_ensemble([pruned_ckpt], strict=False)
    return esemble[0]


def _parse_args():
    parser = ArgumentParser(description="Prune and save distilled model.")
    parser.add_argument(
        "--distilled_ckpt",
        type=str,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--original_ckpt",
        type=str,
        help="Path to the original checkpoint."
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
    
    out_path = Path(args.distilled_ckpt).parent / f"pruned_{Path(args.distilled_ckpt).stem}.pt"
    torch.save(prune_from_ckpt(args.distilled_ckpt, args.original_ckpt), out_path)

    # Check if loading from ckpt works
    model = _load_pruned_model(str(out_path))

    _logger.info(f"Pruned model:\n{model}")
    _logger.info(f"Number of trainable parameters: {model.get_num_params()}")
    _logger.info(f"Successfully saved pruned model weights and config to: {out_path}")
