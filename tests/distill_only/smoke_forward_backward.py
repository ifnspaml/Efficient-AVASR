"""Opt-in end-to-end smoke test for a real Chapter 3 student.

This script loads the frozen teacher checkpoint, performs an optimizer update,
saves the complete model/optimizer state, reconstructs the model, resumes that
state strictly, and performs one further update.  It is intentionally excluded
from unit-test discovery because each invocation loads the large teacher.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
from fairseq import tasks, utils
from hydra.experimental import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "avhubert" / "conf" / "distill_only"


def compose_config(config_name: str):
    overrides = [
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        "task.data=/beegfs/data/shared/lrs3/433h_data_avhubert",
        "task.label_dir=/beegfs/data/shared/lrs3/433h_data_avhubert",
        "task.tokenizer_bpe_model=/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model",
        f"hydra.run.dir={REPO_ROOT / 'exp' / 'chapter3_distill_only' / 'smoke'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        return compose(config_name=config_name, overrides=overrides)


def build(config_name: str, device: torch.device):
    cfg = compose_config(config_name)
    utils.import_user_module(cfg.common)
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model).to(device)
    criterion = task.build_criterion(cfg.criterion).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(cfg.optimization.lr[0]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    return cfg, task, model, criterion, optimizer


def sample(device: torch.device, frames: int) -> dict:
    return {
        "id": torch.tensor([0], device=device),
        "net_input": {
            "source": {
                "audio": torch.randn(1, 104, frames, device=device),
                "video": torch.randn(1, 1, frames, 88, 88, device=device),
            },
            "padding_mask": None,
        },
    }


def update(model, criterion, optimizer, batch) -> tuple[float, tuple[int, ...]]:
    # Evaluation mode keeps the tiny, batch-one smoke deterministic while
    # retaining gradients through every student and head parameter.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    captured = {}

    def capture_shape(_module, _inputs, output):
        captured["shape"] = tuple(output.shape)

    hook = model.distillation_head.register_forward_hook(capture_shape)
    try:
        loss, sample_size, _ = criterion(model, batch)
    finally:
        hook.remove()
    if sample_size != 1 or not torch.isfinite(loss):
        raise AssertionError(f"invalid smoke loss/sample size: {loss}, {sample_size}")
    loss.backward()
    if any(parameter.grad is not None for parameter in model.teacher.parameters()):
        raise AssertionError("the frozen teacher received gradients")
    if not any(
        parameter.grad is not None for parameter in model.student.parameters()
    ):
        raise AssertionError("the student received no gradients")
    optimizer.step()
    return float(loss.detach().cpu()), captured["shape"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default="b2_c1_t2_random_sequence",
        choices=(
            "b2_c1_t2_random_sequence",
            "c2_t6_historical_heads",
            "c3a_t12_historical_heads",
            "c3b_t12_layer_to_layer",
            "d2_selected_conformer",
        ),
    )
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    _, _, model, criterion, optimizer = build(args.config_name, device)
    batch = sample(device, args.frames)
    first_loss, first_shape = update(
        model, criterion, optimizer, batch
    )
    probe_name, probe = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    saved_probe = probe.detach().cpu().clone()

    with tempfile.TemporaryDirectory(prefix="chapter3-distill-smoke-") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_updates": 1,
            },
            checkpoint,
        )
        del batch, criterion, optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _, _, resumed_model, resumed_criterion, resumed_optimizer = build(
            args.config_name, device
        )
        state = torch.load(checkpoint, map_location=device)
        resumed_model.load_state_dict(state["model"], strict=True)
        resumed_optimizer.load_state_dict(state["optimizer"])
        resumed_model.set_num_updates(int(state["num_updates"]))
        resumed_probe = dict(resumed_model.named_parameters())[probe_name]
        if not torch.equal(resumed_probe.detach().cpu(), saved_probe):
            raise AssertionError("strict resume did not restore model weights exactly")
        if resumed_model.num_updates != 1:
            raise AssertionError("strict resume did not restore the update count")
        second_batch = sample(device, args.frames)
        second_loss, second_shape = update(
            resumed_model,
            resumed_criterion,
            resumed_optimizer,
            second_batch,
        )

    result = {
        "config_name": args.config_name,
        "device": str(device),
        "first_loss": first_loss,
        "second_loss": second_loss,
        "first_output_shape": first_shape,
        "second_output_shape": second_shape,
        "checkpoint_round_trip": "strict",
        "resumed_update": 1,
        "teacher_gradients": False,
    }
    if device.type == "cuda":
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
