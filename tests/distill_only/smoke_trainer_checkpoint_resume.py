"""Opt-in Fairseq Trainer checkpoint/resume smoke for Chapter 3.

Unlike ``smoke_forward_backward.py``, this script exercises Fairseq's actual
``Trainer`` and ``checkpoint_utils`` save/load paths. It verifies:

* exact within-stage restoration of model, optimizer, LR scheduler/update,
  iterator epoch/offset, and a deterministic meter;
* a successful optimizer update after the in-stage resume; and
* S2 ``finetune_from_model`` semantics: stage-1 model/head weights are kept,
  while optimizer, scheduler, update count, iterator, and meters start fresh.

It is intentionally excluded from unit-test discovery. Every invocation loads
the real frozen teacher three times and reads the real LRS3 train manifest, so
run it only as an explicit CPU/GPU integration check.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.logging import metrics
from fairseq.trainer import Trainer
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "avhubert" / "conf" / "distill_only"
DEFAULT_DATA = Path("/beegfs/data/shared/lrs3/433h_data_avhubert")
DEFAULT_TOKENIZER = Path(
    "/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model"
)
DEFAULT_TEACHER = Path(
    "/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/"
    "base_vox_iter5.pt"
)
WITHIN_STAGE_METER = "chapter3_within_stage_resume_probe"
STAGE2_METER = "chapter3_s2_fresh_meter_probe"

HELPERS_PATH = Path(__file__).with_name("trainer_resume_helpers.py")
SPEC = importlib.util.spec_from_file_location(
    "chapter3_trainer_resume_helpers", HELPERS_PATH
)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def _require_path(path: Path, label: str, *, directory: bool) -> None:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")


def _compose_trainer_config(
    *,
    config_name: str,
    save_dir: Path,
    data: Path,
    label_dir: Path,
    tokenizer: Path,
    teacher: Path,
    device: str,
    seed: int,
    iterator_frames: int,
    stage: str,
    finetune_from_model: Path | None = None,
):
    overrides = [
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        f"task.data={data}",
        f"task.label_dir={label_dir}",
        f"task.tokenizer_bpe_model={tokenizer}",
        f"model.teacher_path={teacher}",
        f"hydra.run.dir={save_dir.parent}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        experiment_cfg = compose(
            config_name=config_name,
            overrides=overrides,
        )

    cfg = OmegaConf.merge(OmegaConf.structured(FairseqConfig), experiment_cfg)
    # Registration must happen before add_defaults can resolve the new model,
    # task, and criterion dataclasses.
    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"unsupported smoke stage {stage!r}")
    with open_dict(cfg):
        cfg.common.cpu = device == "cpu"
        cfg.common.amp = device == "cuda"
        cfg.common.fp16 = False
        cfg.common.bf16 = False
        cfg.common.seed = seed
        cfg.common.log_interval = 1
        cfg.distributed_training.distributed_world_size = 1
        cfg.distributed_training.nprocs_per_node = 1

        # The real dataset is loaded so Fairseq creates/restores its genuine
        # EpochBatchIterator. Keep its first materialized batch tiny; training
        # itself uses a synthetic audiovisual sample of ``--frames`` frames.
        cfg.dataset.num_workers = 0
        cfg.dataset.max_tokens = max(iterator_frames, 4)
        cfg.dataset.batch_size = 1
        cfg.dataset.required_batch_size_multiple = 1
        cfg.dataset.data_buffer_size = 0
        cfg.optimization.update_freq = [1]
        cfg.task.max_sample_size = iterator_frames
        cfg.task.image_aug = False

        cfg.checkpoint.save_dir = str(save_dir)
        cfg.checkpoint.restore_file = "checkpoint_last.pt"
        cfg.checkpoint.finetune_from_model = (
            str(finetune_from_model) if finetune_from_model is not None else None
        )
        cfg.checkpoint.no_save = False
        cfg.checkpoint.no_last_checkpoints = False
        cfg.checkpoint.no_save_optimizer_state = False
        cfg.checkpoint.write_checkpoints_asynchronously = False

        # Reproduce the optional S2 schedule boundaries without running their
        # full update budgets.
        cfg.model.schedule_mode = "two_stage_50k_25k"
        if stage == "stage1":
            cfg.optimization.max_update = 50000
            cfg.optimization.lr = [0.002]
            cfg.lr_scheduler.warmup_updates = 15000
            cfg.lr_scheduler.total_num_update = 50000
            cfg.model.max_update = 50000
            cfg.model.warmup_updates = 15000
        else:
            if finetune_from_model is None:
                raise ValueError("stage2 requires a stage-1 checkpoint")
            cfg.optimization.max_update = 25000
            cfg.optimization.lr = [0.0001]
            cfg.lr_scheduler.warmup_updates = 5000
            cfg.lr_scheduler.total_num_update = 25000
            cfg.model.max_update = 25000
            cfg.model.warmup_updates = 5000
            cfg.model.initialization_policy = "warm_start_distilled"

    # This repository pins an OmegaConf release predating
    # ``OmegaConf.resolve``. Round-trip through a resolved container, matching
    # Fairseq's own hydra_train entry point.
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


def _build_trainer(cfg) -> tuple[Any, Trainer]:
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    return task, Trainer(cfg, task, model, criterion)


def _synthetic_sample(frames: int) -> dict[str, Any]:
    return {
        "id": torch.tensor([0]),
        "net_input": {
            "source": {
                "audio": torch.randn(1, 104, frames),
                "video": torch.randn(1, 1, frames, 88, 88),
            },
            "padding_mask": None,
        },
    }


def _train_one_update(trainer: Trainer, frames: int) -> dict[str, Any]:
    before = trainer.get_num_updates()
    logging_output = trainer.train_step([_synthetic_sample(frames)])
    if logging_output is None:
        raise AssertionError("Fairseq Trainer returned no logging output")
    if trainer.get_num_updates() != before + 1:
        raise AssertionError(
            "Trainer update count did not increment exactly once: "
            f"{before} -> {trainer.get_num_updates()}"
        )
    model = trainer.get_model()
    if any(parameter.grad is not None for parameter in model.teacher.parameters()):
        raise AssertionError("the frozen teacher received gradients")
    return {
        "before": before,
        "after": trainer.get_num_updates(),
        "learning_rate": trainer.get_lr(),
    }


def _activate_iterator(epoch_itr, *, consume: int) -> dict[str, Any]:
    iterator = epoch_itr.next_epoch_itr(shuffle=False)
    for _ in range(consume):
        batch = next(iterator)
        del batch
    state = epoch_itr.state_dict()
    helpers.iterator_position(state)
    return state


def _meter_state(key: str) -> dict[str, Any] | None:
    meter = metrics.get_meter("default", key)
    return None if meter is None else meter.state_dict()


def _snapshot(
    trainer: Trainer,
    *,
    iterator_state: dict[str, Any],
    meter_key: str,
) -> dict[str, Any]:
    meter_state = _meter_state(meter_key)
    if meter_state is None:
        raise AssertionError(f"required Fairseq meter {meter_key!r} is absent")
    return {
        # Store digests rather than live tensor mappings so the large teacher
        # can be released before the next Trainer is constructed.
        "model_state": helpers.state_digest(trainer.model.state_dict()),
        "optimizer_state": helpers.state_digest(trainer.optimizer.state_dict()),
        "lr_scheduler_state": trainer.lr_scheduler.state_dict(),
        "num_updates": trainer.get_num_updates(),
        "learning_rate": trainer.get_lr(),
        "iterator_state": iterator_state,
        "meter_state": meter_state,
    }


def _run(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    stage1_dir = work_dir / "stage1"
    stage2_dir = work_dir / "stage2"
    stage1_cfg = _compose_trainer_config(
        config_name=args.config_name,
        save_dir=stage1_dir,
        data=args.data,
        label_dir=args.label_dir,
        tokenizer=args.tokenizer,
        teacher=args.teacher,
        device=args.device,
        seed=args.seed,
        iterator_frames=args.iterator_frames,
        stage="stage1",
    )

    metrics.reset()
    task, trainer = _build_trainer(stage1_cfg)
    epoch_itr = trainer.get_train_iterator(epoch=1, load_dataset=True)
    stage1_iterator_state = _activate_iterator(epoch_itr, consume=1)
    if helpers.iterator_position(stage1_iterator_state) != (1, 1):
        raise AssertionError(
            "stage-1 iterator must be advanced exactly one batch, got "
            f"{helpers.iterator_position(stage1_iterator_state)}"
        )
    first_update = _train_one_update(trainer, args.frames)
    metrics.log_scalar(WITHIN_STAGE_METER, 123.5, weight=2.0)
    saved = _snapshot(
        trainer,
        iterator_state=stage1_iterator_state,
        meter_key=WITHIN_STAGE_METER,
    )
    optimizer_slot_count = len(trainer.optimizer.state_dict().get("state", {}))
    if optimizer_slot_count == 0:
        raise AssertionError(
            "stage-1 Adam optimizer has no slot state after an update"
        )

    checkpoint_utils.save_checkpoint(
        stage1_cfg.checkpoint,
        trainer,
        epoch_itr,
        val_loss=17.0,
    )
    checkpoint = stage1_dir / "checkpoint_last.pt"
    _require_path(checkpoint, "stage-1 smoke checkpoint", directory=False)

    del epoch_itr, trainer, task
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Normal interruption resume: same stage directory, no reset flags.
    metrics.reset()
    resumed_task, resumed_trainer = _build_trainer(stage1_cfg)
    extra_state, resumed_epoch_itr = checkpoint_utils.load_checkpoint(
        stage1_cfg.checkpoint,
        resumed_trainer,
        disable_iterator_cache=resumed_task.has_sharded_data("train"),
    )
    if extra_state is None:
        raise AssertionError("within-stage checkpoint was not loaded")
    resumed_iterator_state = _activate_iterator(
        resumed_epoch_itr,
        consume=0,
    )
    resumed = _snapshot(
        resumed_trainer,
        iterator_state=resumed_iterator_state,
        meter_key=WITHIN_STAGE_METER,
    )
    helpers.require_within_stage_resume(expected=saved, actual=resumed)
    resumed_update = _train_one_update(resumed_trainer, args.frames)

    del resumed_epoch_itr, resumed_trainer, resumed_task
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # S2 warm start: the empty stage2 save directory makes Fairseq select
    # finetune_from_model, whose internal reset flags are the semantics under
    # test.
    stage2_cfg = _compose_trainer_config(
        config_name=args.config_name,
        save_dir=stage2_dir,
        data=args.data,
        label_dir=args.label_dir,
        tokenizer=args.tokenizer,
        teacher=args.teacher,
        device=args.device,
        seed=args.seed,
        iterator_frames=args.iterator_frames,
        stage="stage2",
        finetune_from_model=checkpoint,
    )
    metrics.reset()
    stage2_task, stage2_trainer = _build_trainer(stage2_cfg)
    # Force creation before load and snapshot the pristine optimizer/scheduler.
    metrics.log_scalar(STAGE2_METER, 9.25, weight=4.0)
    fresh_stage2 = {
        "optimizer_state": helpers.state_digest(
            stage2_trainer.optimizer.state_dict()
        ),
        "lr_scheduler_state": stage2_trainer.lr_scheduler.state_dict(),
        "learning_rate": stage2_trainer.get_lr(),
        "meter_state": _meter_state(STAGE2_METER),
    }
    fresh_optimizer_slots = len(
        stage2_trainer.optimizer.state_dict().get("state", {})
    )
    if fresh_optimizer_slots != 0:
        raise AssertionError(
            "fresh S2 stage-2 Adam unexpectedly has optimizer slot state"
        )

    s2_extra_state, stage2_epoch_itr = checkpoint_utils.load_checkpoint(
        stage2_cfg.checkpoint,
        stage2_trainer,
        disable_iterator_cache=stage2_task.has_sharded_data("train"),
    )
    if s2_extra_state is None:
        raise AssertionError("S2 stage 2 did not load the stage-1 checkpoint")
    if _meter_state(WITHIN_STAGE_METER) is not None:
        raise AssertionError("S2 stage 2 restored stage-1 meters")
    stage2_iterator_state = _activate_iterator(stage2_epoch_itr, consume=0)
    loaded_stage2 = _snapshot(
        stage2_trainer,
        iterator_state=stage2_iterator_state,
        meter_key=STAGE2_METER,
    )
    helpers.require_s2_warm_start(
        stage1_model_state=saved["model_state"],
        stage1_iterator_state=saved["iterator_state"],
        fresh_stage2=fresh_stage2,
        loaded_stage2=loaded_stage2,
    )

    result = {
        "schema_version": "chapter3-fairseq-trainer-resume-smoke/v1",
        "config_name": args.config_name,
        "device": args.device,
        "seed": args.seed,
        "teacher_checkpoint": str(args.teacher.resolve()),
        "data_root": str(args.data.resolve()),
        "fairseq_apis": {
            "save": "checkpoint_utils.save_checkpoint",
            "within_stage_load": "checkpoint_utils.load_checkpoint",
            "s2_load": "checkpoint.finetune_from_model",
        },
        "within_stage": {
            "model_state_sha256": saved["model_state"],
            "optimizer_state_sha256": saved["optimizer_state"],
            "lr_scheduler_state_sha256": helpers.state_digest(
                saved["lr_scheduler_state"]
            ),
            "meter_state_sha256": helpers.state_digest(saved["meter_state"]),
            "num_updates": saved["num_updates"],
            "learning_rate": saved["learning_rate"],
            "iterator_position": helpers.iterator_position(
                saved["iterator_state"]
            ),
            "optimizer_slot_count": optimizer_slot_count,
            "post_resume_update": resumed_update,
            "exact_restore": True,
        },
        "s2_stage2": {
            "model_state_sha256": loaded_stage2["model_state"],
            "fresh_optimizer_state_sha256": fresh_stage2["optimizer_state"],
            "loaded_optimizer_state_sha256": loaded_stage2["optimizer_state"],
            "fresh_lr_scheduler_state_sha256": helpers.state_digest(
                fresh_stage2["lr_scheduler_state"]
            ),
            "loaded_lr_scheduler_state_sha256": helpers.state_digest(
                loaded_stage2["lr_scheduler_state"]
            ),
            "num_updates": loaded_stage2["num_updates"],
            "learning_rate": loaded_stage2["learning_rate"],
            "iterator_position": helpers.iterator_position(
                loaded_stage2["iterator_state"]
            ),
            "optimizer_slot_count": fresh_optimizer_slots,
            "stage1_meter_absent": True,
            "weights_retained_training_state_reset": True,
        },
        "first_update": first_update,
    }
    if args.device == "cuda":
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default="b2_c1_t2_random_sequence",
        help="public distill_only config whose architecture is exercised",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--iterator-frames",
        type=int,
        default=16,
        help="tiny real-LRS3 batch length used only for iterator state",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="optional empty directory in which checkpoints are retained",
    )
    args = parser.parse_args()

    if args.frames <= 0 or args.iterator_frames <= 0:
        parser.error("--frames and --iterator-frames must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    for path, label, directory in (
        (args.data, "LRS3 data", True),
        (args.label_dir, "LRS3 label", True),
        (args.tokenizer, "tokenizer", False),
        (args.teacher, "teacher checkpoint", False),
    ):
        _require_path(path, label, directory=directory)

    if args.work_dir is not None:
        work_dir = args.work_dir.resolve()
        if work_dir.exists() and any(work_dir.iterdir()):
            parser.error(f"--work-dir must be empty: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        result = _run(args, work_dir)
        result["work_dir"] = str(work_dir)
        result["work_dir_retained"] = True
    else:
        with tempfile.TemporaryDirectory(
            prefix="chapter3-fairseq-trainer-resume-"
        ) as directory:
            result = _run(args, Path(directory))
        result["work_dir_retained"] = False
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
