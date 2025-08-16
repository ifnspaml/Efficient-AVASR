import logging
import sys
from argparse import ArgumentParser

import numpy as np

import torch
from ptflops import get_model_complexity_info

from fairseq import tasks, checkpoint_utils
import hubert_pretraining, hubert


_logger = logging.getLogger(__name__)


def _parse_args():
    parser = ArgumentParser(description="Evaluate model FLOPs.")
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint."
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
    np.random.seed(1337)

    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt)
    model_cfg = state.get("cfg", None)
    task_pretrain = tasks.setup_task(model_cfg.task)
    task_pretrain.load_state_dict(state['task_state'])
    model = task_pretrain.build_model(model_cfg.model)

    del state['model']['mask_emb']
    model.load_state_dict(state["model"], strict=False)
    model.remove_pretraining_modules()
    model.eval()

    data = lambda x : {
        "source": {"audio": torch.ones((1, 104, 100)), "video": torch.ones((1, 1, 100, 88, 88))},
        "features_only": True,
        "mask":False
    }

    macs, params = get_model_complexity_info(model, (1,768), input_constructor=data, as_strings=True, backend='aten', print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
