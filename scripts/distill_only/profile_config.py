#!/usr/bin/env python3
"""Profile an untrained resolved student config before launching an experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from fairseq import tasks
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "avhubert" / "conf" / "distill_only"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.manifest import (  # noqa: E402
    ManifestStore,
    atomic_write_json,
    utc_now,
)
from chapter3_distill_only.profiling import (  # noqa: E402
    profile_architecture_fields,
    profile_distillation_model,
)
from chapter3_distill_only.selection import (  # noqa: E402
    inherited_hydra_overrides,
    read_selection,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-selection", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="inherit a validation candidate before the C-to-D lock exists",
    )
    parser.add_argument("--destination-arch", choices=("transformer", "conformer"))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--data",
        default="/beegfs/data/shared/lrs3/433h_data_avhubert",
    )
    parser.add_argument(
        "--tokenizer",
        default="/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.from_selection and args.source_manifest:
        raise SystemExit("--from-selection and --source-manifest are mutually exclusive")
    selection_overrides = []
    source_manifest_path = None
    if args.from_selection:
        selection = read_selection(args.from_selection)
        source_manifest_path = Path(selection["selected_manifest"]).resolve()
        selection_overrides = inherited_hydra_overrides(
            selection,
            destination_arch=args.destination_arch,
            destination_experiment=args.config_name,
        )
    elif args.source_manifest:
        source_manifest_path = ManifestStore(args.source_manifest).path.resolve()
        selection_overrides = inherited_hydra_overrides(
            {
                "selected_manifest": str(args.source_manifest.resolve()),
                "materialized_overrides": [],
                "overrides_by_experiment": {},
            },
            destination_arch=args.destination_arch,
            destination_experiment=None,
        )
    overrides = [
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        f"task.data={args.data}",
        f"task.label_dir={args.data}",
        f"task.tokenizer_bpe_model={args.tokenizer}",
        f"hydra.run.dir={REPO_ROOT / 'exp' / 'chapter3_distill_only' / 'profile'}",
        *selection_overrides,
        *args.override,
    ]
    try:
        with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
            cfg = compose(config_name=args.config_name, overrides=overrides)
        # Importing chapter3_distill_only.profiling above has already loaded
        # the user package and registered all Fairseq components.
        task = tasks.setup_task(cfg.task)
        model = task.build_model(cfg.model)
        model.cpu()
        result = profile_distillation_model(model)
        source_identity = None
        if source_manifest_path is not None:
            source_manifest = ManifestStore(source_manifest_path).read()
            source_identity = {
                "path": str(source_manifest_path),
                "immutable_sha256": source_manifest["immutable_sha256"],
                "config_digest": source_manifest["immutable"].get(
                    "config_digest"
                ),
            }
        result.update(
            {
                "schema_version": "chapter3-config-profile/v1",
                "measured_at": utc_now(),
                "config_name": args.config_name,
                "resolved_config": OmegaConf.to_container(
                    cfg, resolve=True, enum_to_str=True
                ),
                "selection": (
                    str(args.from_selection.resolve())
                    if args.from_selection
                    else None
                ),
                "source_manifest": source_identity,
                "overrides": overrides,
                **profile_architecture_fields(cfg.model),
            }
        )
        atomic_write_json(args.output, result)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
