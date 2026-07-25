"""Build one real teacher/student wrapper from a matrix config.

This is intentionally an opt-in smoke script rather than a unit test because
it loads the approximately 1.2 GB frozen teacher checkpoint.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from fairseq import tasks, utils
from hydra.experimental import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "avhubert" / "conf" / "distill_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", default="b2_c1_t2_random_sequence"
    )
    parser.add_argument(
        "--verify-export",
        action="store_true",
        help="also save the student-native payload and load it through Fairseq and HubertEncoder",
    )
    args = parser.parse_args()
    overrides = [
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        "task.data=/beegfs/data/shared/lrs3/433h_data_avhubert",
        "task.label_dir=/beegfs/data/shared/lrs3/433h_data_avhubert",
        "task.tokenizer_bpe_model=/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model",
        f"hydra.run.dir={REPO_ROOT / 'exp' / 'chapter3_distill_only' / 'smoke'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    utils.import_user_module(cfg.common)
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    output = {
        "model": type(model).__name__,
        "teacher_depth": len(model.teacher.encoder.layers),
        "student_depth": len(model.student.encoder.layers),
        "student_dim": model.student.encoder_embed_dim,
        "targets": list(model.teacher_target_layers),
        "mode": model.distill_head_mode,
        "initialization": model.initialization_report,
        "student_parameters": model.get_student_num_params(),
        "head_parameters": model.get_prediction_head_num_params(),
        "teacher_parameters": model.get_teacher_num_params(),
    }
    if args.verify_export:
        from chapter3_distill_only.exporter import verify_export

        with tempfile.TemporaryDirectory(
            prefix="chapter3-distill-export-smoke-"
        ) as directory:
            checkpoint = Path(directory) / "student.pt"
            torch.save(model.get_export_state(), checkpoint)
            output["export_verification"] = verify_export(checkpoint)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
