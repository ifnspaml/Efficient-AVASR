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


def save_from_ckpt(distilled_ckpt: str, original_ckpt: str) -> Dict[str, Any]:
    ensemble, _ = load_model_ensemble([distilled_ckpt])
    distill_model = ensemble[0]
    model = distill_model.student

    # Update original checkpoint state
    original_state = torch.load(original_ckpt, map_location=torch.device("cpu"))
    original_state["model"] = model.state_dict()
    original_state["extra_state"]["distill_linear_projs"] = distill_model.distill_linear_projs.state_dict()

    return original_state


def _load_saved_model(ckpt: str) -> AVHubertModel:
    esemble, _ = load_model_ensemble([ckpt], strict=False)
    return esemble[0]


def _parse_args():
    parser = ArgumentParser(description="Save distilled model.")
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
    
    out_path = Path(args.distilled_ckpt).parent / f"{Path(args.original_ckpt).stem}_final.pt"
    torch.save(
        save_from_ckpt(
            distilled_ckpt=args.distilled_ckpt,
            original_ckpt=args.original_ckpt,
        ),
        out_path
    )

    # Check if loading from ckpt works
    model = _load_saved_model(str(out_path))

    _logger.info(f"Model:\n{model}")
    _logger.info(f"Number of trainable parameters: {model.get_num_params()}")
    _logger.info(f"Successfully saved model checkpoint to: {out_path}")
    