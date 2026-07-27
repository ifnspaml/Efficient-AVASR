#!/usr/bin/env python3
"""Guarded launcher for the isolated Chapter 3 distillation-only pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3_distill_only.evaluation import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    EvaluationCondition,
    EvaluationProtocol,
    EvaluationProtocolError,
    load_evaluation_protocol,
    measurement_metadata,
    validate_protocol_artifacts,
)
from chapter3_distill_only.manifest import (  # noqa: E402
    ManifestError,
    ManifestStore,
    build_provenance,
    sha256_file,
    sha256_json,
    utc_now,
)
from chapter3_distill_only.profiling import PeakMemoryMonitor  # noqa: E402
from chapter3_distill_only.selection import (  # noqa: E402
    inherited_hydra_overrides,
    read_selection,
    require_final_selection,
)


BASELINE_BRANCH = "dev_li_pro6000"
BASELINE_COMMIT = "4502130ce4470fe8b0456fd5b466bef043f383f9"
OLD_REPO = Path("/home/zhengyangli/work/distil-av-hubert")
OLD_REPO_COMMIT = "13e27bcb44710c244ef48899d4458107fc35b2cb"
DEFAULT_TEACHER = Path(
    "/home/zhengyangli/work/av_hubert_pre_trained_models/phd_thesis/base_vox_iter5.pt"
)
DEFAULT_DATA = Path("/beegfs/data/shared/lrs3/433h_data_avhubert")
DEFAULT_TOKENIZER = Path("/beegfs/data/shared/lrs3/spm1000/spm_unigram1000.model")
DEFAULT_NOISE = Path("/beegfs/data/shared/lrs3/noise/musan/tsv/all")
FIXED_FILE_SHA256 = {
    DEFAULT_TEACHER: "995e53b2e6ed143546df6228bc683ab3dcc9366fecd047dbe2488deabb8af3db",
    DEFAULT_DATA / "train.tsv": "bd3f0ac8f6de7ba9da69cb24e23c7b1c1e88ece903c416d29897389f22349996",
    DEFAULT_DATA / "valid.tsv": "1533c8aa7857bfe17227d159201cd407d673ac75935ef0ab02ec81206de78d27",
    DEFAULT_DATA / "test.tsv": "8a4d0967a9c5b788c0af0a16df57b360ed718d02a610341f998187ac4eadb311",
    DEFAULT_TOKENIZER: "e50bb215df1779bd2dbff133ba7c8d3211c4f0c9702de4c54a51dc48a894e620",
    DEFAULT_NOISE / "train.tsv": "f77fae3c54281fd06a6e7014664aec79c800b58a4eb0491defd5479c17f8cfe5",
}
CONFIG_ROOT = REPO_ROOT / "avhubert" / "conf" / "distill_only"
OUTPUT_ROOT = REPO_ROOT / "exp" / "chapter3_distill_only"
DEFAULT_EVALUATION_PROTOCOL = DEFAULT_PROTOCOL_PATH
NOISE_OVERRIDE_KEYS = (
    "override.noise_wav",
    "override.noise_prob",
    "override.noise_snr",
    "override.noise_method",
)
SELECTED_EXPERIMENTS = {
    "d1_selected_transformer",
    "d2_selected_conformer",
    "e1_selected_clean",
    "e2_selected_noisy",
    "s1_selected_main",
    "s2_optional_two_stage",
}
ALIAS_EXPERIMENTS = {
    "d1_selected_transformer",
    "e1_selected_clean",
    "s1_selected_main",
}
EXPECTED_SELECTION_KIND = {
    "d1_selected_transformer": "c_to_d",
    "d2_selected_conformer": "c_to_d",
    "e1_selected_clean": "d_to_e",
    "e2_selected_noisy": "d_to_e",
    "s1_selected_main": "final",
    "s2_optional_two_stage": "final",
}
PROTECTED_TRACKED_PATHS = (
    "avhubert/conf/distill",
    "avhubert/hubert_distill.py",
    "avhubert/hubert_distill_criterion.py",
    "avhubert/prune.py",
    "avhubert/merge.py",
    "avhubert/save_final_ckpt.py",
    "scripts/run_pruning.sh",
    "scripts/run_pruning_merging.sh",
    "scripts/run_merging.sh",
)
STAGES = (
    "prepare",
    "encoder",
    "stage1",
    "stage2",
    "export",
    "finetune",
    "validate",
    "test",
    "all",
)


class PreflightError(RuntimeError):
    pass


def _git(*arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and process.returncode:
        raise PreflightError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _deep_merge(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_composed_config(path: Path, stack: tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Resolve the simple same-directory Hydra defaults used by this suite."""

    path = path.resolve()
    if path in stack:
        raise PreflightError(f"recursive config defaults: {' -> '.join(map(str, stack + (path,)))}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PreflightError(f"config {path} must contain a mapping")
    defaults = loaded.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise PreflightError(f"config defaults must be a list in {path}")
    result: Dict[str, Any] = {}
    inserted_self = False
    for default in defaults:
        if default == "_self_":
            _deep_merge(result, loaded)
            inserted_self = True
            continue
        if isinstance(default, str):
            default_name = default
        elif isinstance(default, dict) and len(default) == 1:
            group, name = next(iter(default.items()))
            if name in (None, "null"):
                continue
            default_name = str(Path(str(group)) / str(name))
        else:
            raise PreflightError(f"unsupported default {default!r} in {path}")
        inherited_path = path.parent / (
            default_name if default_name.endswith(".yaml") else f"{default_name}.yaml"
        )
        _deep_merge(
            result,
            load_composed_config(inherited_path, stack=stack + (path,)),
        )
    if not inserted_self:
        _deep_merge(result, loaded)
    return result


def _get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return default
        value = value[component]
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _layer_ids(value: Any, field: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in str(value).split(",") if str(item).strip())
    except ValueError as exc:
        raise PreflightError(f"{field} must be a comma-separated integer list") from exc
    if not layers or len(set(layers)) != len(layers) or any(item < 0 for item in layers):
        raise PreflightError(f"{field} must contain unique, non-negative layer IDs")
    return layers


def validate_safe_config(config: Mapping[str, Any]) -> None:
    """Reject pruning, merging, composite optimization, and invalid schedules."""

    errors = []
    if _get(config, "optimizer._name") != "adam":
        errors.append("optimizer._name must be ordinary adam")
    if _get(config, "optimizer.groups") is not None:
        errors.append("composite optimizer groups are forbidden")
    if _as_bool(_get(config, "criterion.use_reg", False)):
        errors.append("criterion.use_reg must be false")
    if _as_bool(_get(config, "criterion.use_merge", False)):
        errors.append("criterion.use_merge must be false")
    pruning_units = _get(config, "model.pruning_units", "")
    if pruning_units not in (None, "", [], ()):
        errors.append("model.pruning_units must be empty")
    model_name = str(_get(config, "model._name", ""))
    criterion_name = str(_get(config, "criterion._name", ""))
    if model_name != "av_hubert_distill_only":
        errors.append("model must use the isolated av_hubert_distill_only registration")
    if criterion_name != "av_hubert_distill_only":
        errors.append("criterion must use the isolated av_hubert_distill_only registration")
    if int(_get(config, "optimization.clip_norm", -1)) != 10:
        errors.append("optimization.clip_norm must be 10")
    if _get(config, "lr_scheduler._name") != "polynomial_decay":
        errors.append("lr_scheduler must be polynomial_decay")
    if float(_get(config, "lr_scheduler.power", -1)) != 1.0:
        errors.append("polynomial-decay power must be 1")
    schedule_mode = str(_get(config, "model.schedule_mode", "continuous_75k"))
    if schedule_mode not in {"continuous_75k", "two_stage_50k_25k"}:
        errors.append(f"unsupported schedule mode: {schedule_mode}")
    if schedule_mode == "two_stage_50k_25k":
        expected_max_update = 50000
        expected_warmup = 15000
        expected_total = 50000
    else:
        expected_max_update = 75000
        expected_warmup = 15000
        expected_total = 75000
    if int(_get(config, "optimization.max_update", -1)) != expected_max_update:
        errors.append(f"optimization.max_update must be {expected_max_update}")
    lr = _get(config, "optimization.lr", [])
    if not isinstance(lr, list) or len(lr) != 1 or float(lr[0]) != 0.002:
        errors.append("stage-1/main optimization.lr must be [0.002]")
    if int(_get(config, "lr_scheduler.warmup_updates", -1)) != expected_warmup:
        errors.append(f"warm-up must be {expected_warmup} updates")
    if int(_get(config, "lr_scheduler.total_num_update", -1)) != expected_total:
        errors.append(f"scheduler total_num_update must be {expected_total}")
    betas = str(_get(config, "optimizer.adam_betas", "")).replace(" ", "")
    if betas not in ("(0.9,0.999)", "[0.9,0.999]"):
        errors.append("Adam betas must be (0.9,0.999)")
    if float(_get(config, "optimizer.adam_eps", -1)) != 1e-8:
        errors.append("Adam epsilon must be 1e-8")
    if float(_get(config, "optimizer.weight_decay", -1)) != 0.0:
        errors.append("weight decay must be 0")
    update_freq = _get(config, "optimization.update_freq", [])
    max_tokens = int(_get(config, "dataset.max_tokens", -1))
    if (
        not isinstance(update_freq, list)
        or len(update_freq) != 1
        or max_tokens <= 0
        or max_tokens * int(update_freq[0]) != 16000
    ):
        errors.append("effective batch must preserve max_tokens * update_freq = 16000")
    targets = str(_get(config, "model.teacher_target_layers", "")).replace(" ", "")
    if targets != "0,4,8,12":
        errors.append("main teacher targets must be exactly 0,4,8,12")
    try:
        target_ids = _layer_ids(targets, "model.teacher_target_layers")
    except PreflightError as exc:
        errors.append(str(exc))
        target_ids = ()
    head_mode = _get(config, "model.distill_head_mode")
    student_depth = int(_get(config, "model.student_depth", -1))
    if head_mode == "layer_to_layer":
        matches = str(_get(config, "model.student_match_layers", "")).replace(" ", "")
        if matches != "0,4,8,12":
            errors.append("main layer-to-layer student mapping must be 0,4,8,12")
        try:
            match_ids = _layer_ids(matches, "model.student_match_layers")
            if len(match_ids) != len(target_ids):
                errors.append("teacher targets and student mappings must have equal length")
            if match_ids and max(match_ids) > student_depth:
                errors.append("student mapping exceeds the configured student depth")
        except PreflightError as exc:
            errors.append(str(exc))
    elif head_mode != "historical_pred_heads":
        errors.append("distill_head_mode must be historical_pred_heads or layer_to_layer")
    embed_dim = int(_get(config, "model.student_embed_dim", -1))
    attention_heads = int(_get(config, "model.student_attention_heads", -1))
    if attention_heads <= 0 or embed_dim <= 0 or embed_dim % attention_heads:
        errors.append("student embed dimension must be divisible by a positive head count")
    if _get(config, "model.student_arch") == "conformer":
        kernel = int(_get(config, "model.student_conformer_kernel", 0))
        if kernel <= 0 or kernel % 2 == 0:
            errors.append("Conformer kernel must be positive and odd")
    for field in (
        "distill_loss_type",
        "l1_weight",
        "l2_weight",
        "cosine_weight",
        "cosine_type",
        "feature_penalty_weight",
    ):
        model_value = _get(config, f"model.{field}")
        criterion_value = _get(config, f"criterion.{field}")
        if model_value != criterion_value:
            errors.append(f"model.{field} must equal criterion.{field}")
    noise_probability = float(_get(config, "task.noise_prob", 0.0))
    if noise_probability != float(_get(config, "task.distillation_noise_prob", -1)):
        errors.append("task noise probability fields must agree")
    if noise_probability != float(_get(config, "model.distillation_noise_prob", -1)):
        errors.append("task and model noise probability fields must agree")
    noise_mirrors = (
        (
            "snr",
            str(_get(config, "task.noise_snr")),
            str(_get(config, "task.distillation_noise_snr")),
            str(_get(config, "model.distillation_noise_snr")),
        ),
        (
            "method",
            str(_get(config, "task.noise_method")),
            str(_get(config, "task.distillation_noise_method")),
            str(_get(config, "model.distillation_noise_method")),
        ),
        (
            "manifest root",
            str(_get(config, "task.noise_wav")),
            str(_get(config, "task.distillation_noise_manifest_root")),
            str(_get(config, "model.distillation_noise_manifest_root")),
        ),
        (
            "train-only flag",
            str(_get(config, "task.distillation_noise_train_only")),
            str(_get(config, "task.distillation_noise_train_only")),
            str(_get(config, "model.distillation_noise_train_only")),
        ),
    )
    for label, base_value, task_value, model_value in noise_mirrors:
        if base_value != task_value or task_value != model_value:
            errors.append(f"task/model distillation noise {label} fields must agree")
    if noise_probability:
        if not _as_bool(_get(config, "task.distillation_noise_train_only", False)):
            errors.append("distillation noise must be explicitly train-only")
        if noise_probability != 0.25:
            errors.append("the configured noisy control must use probability 0.25")
        if float(_get(config, "task.noise_snr", 999)) != 0:
            errors.append("the configured noisy control must use SNR 0 dB")
        if str(_get(config, "task.noise_method", "")).lower() != "rms":
            errors.append("active encoder-training noise must use task.noise_method=rms")
        if str(_get(config, "task.distillation_noise_method", "")).lower() != "rms":
            errors.append(
                "active encoder-training noise must use "
                "task.distillation_noise_method=rms"
            )
    for dotted in ("task.noise_method", "task.distillation_noise_method"):
        method = str(_get(config, dotted, "rms")).lower()
        if method == "itut":
            errors.append(f"{dotted}=itut is forbidden during encoder training")
    flattened = json.dumps(config, sort_keys=True).lower()
    for forbidden in ("log_alpha", "lagrange", "physical_prun"):
        if forbidden in flattened:
            errors.append(f"forbidden pruning token present: {forbidden}")
    if errors:
        raise PreflightError("unsafe distillation-only config:\n- " + "\n- ".join(errors))


def validate_fixed_resolved_inputs(
    config: Mapping[str, Any], args: argparse.Namespace
) -> None:
    expected = {
        "model.teacher_path": args.teacher,
        "task.data": args.data,
        "task.label_dir": args.data,
        "task.tokenizer_bpe_model": args.tokenizer,
    }
    errors = []
    for dotted, path in expected.items():
        value = _get(config, dotted)
        if value is None or Path(str(value)).resolve() != path.resolve():
            errors.append(f"{dotted} must resolve to {path}")
    if float(_get(config, "task.noise_prob", 0.0)):
        for dotted in (
            "task.noise_wav",
            "task.distillation_noise_manifest_root",
            "model.distillation_noise_manifest_root",
        ):
            value = _get(config, dotted)
            if value is None or Path(str(value)).resolve() != args.noise_root.resolve():
                errors.append(f"{dotted} must resolve to {args.noise_root}")
    if errors:
        raise PreflightError(
            "resolved config changed fixed protocol inputs:\n- "
            + "\n- ".join(errors)
        )


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path.resolve()), str(root.resolve()))) == str(
            root.resolve()
        )
    except ValueError:
        return False


def _require_path(path: Path, label: str, *, directory: Optional[bool] = None) -> None:
    if not path.exists():
        raise PreflightError(f"{label} does not exist: {path}")
    if directory is True and not path.is_dir():
        raise PreflightError(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise PreflightError(f"{label} must be a file: {path}")


def _require_digest(path: Path) -> None:
    expected = FIXED_FILE_SHA256[path]
    actual = sha256_file(path)
    if actual != expected:
        raise PreflightError(
            f"fixed input digest mismatch for {path}: {actual} != {expected}"
        )


def _archive_configs() -> list[Path]:
    result_root = OLD_REPO / "s3prl" / "result" / "pretrain"
    if not result_root.is_dir():
        return []
    return sorted(
        [
            *result_root.rglob("config_model.yaml"),
            *result_root.rglob("config_runner.yaml"),
        ]
    )


def preflight(
    *,
    experiment: str,
    config_path: Path,
    config: Mapping[str, Any],
    teacher: Path,
    data: Path,
    tokenizer: Path,
    noise_root: Path,
    output_root: Path,
    run_dir: Path,
    selection_path: Optional[Path],
) -> Dict[str, Any]:
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    if branch == "dev_li":
        raise PreflightError("branch dev_li is explicitly forbidden")
    if branch != BASELINE_BRANCH and not branch.startswith("feat/"):
        raise PreflightError(
            f"expected {BASELINE_BRANCH} or a feature branch, found {branch}"
        )
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        check=False,
    )
    if ancestor.returncode:
        raise PreflightError(f"HEAD does not descend from required baseline {BASELINE_COMMIT}")
    worktree_status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if worktree_status:
        raise PreflightError(
            "worktree must be fully clean before launch so the implementation "
            "commit identifies every executed file:\n" + worktree_status
        )
    protected_diff = _git(
        "diff",
        "--name-only",
        f"{BASELINE_COMMIT}..HEAD",
        "--",
        *PROTECTED_TRACKED_PATHS,
    )
    if protected_diff:
        raise PreflightError(
            "existing joint-DP files differ from the required baseline:\n"
            + protected_diff
        )
    _require_path(OLD_REPO, "historical distillation repository", directory=True)
    old_commit_process = subprocess.run(
        ["git", "-C", str(OLD_REPO), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if (
        old_commit_process.returncode
        or old_commit_process.stdout.strip() != OLD_REPO_COMMIT
    ):
        raise PreflightError(
            "historical reference must remain at "
            f"{OLD_REPO_COMMIT}; found {old_commit_process.stdout.strip() or 'unavailable'}"
        )
    _require_path(config_path, "experiment config", directory=False)
    if teacher.resolve() != DEFAULT_TEACHER.resolve():
        raise PreflightError(f"teacher must be the fixed checkpoint {DEFAULT_TEACHER}")
    if data.resolve() != DEFAULT_DATA.resolve():
        raise PreflightError(f"data must be the fixed LRS3 setup {DEFAULT_DATA}")
    if tokenizer.resolve() != DEFAULT_TOKENIZER.resolve():
        raise PreflightError(f"tokenizer must be fixed at {DEFAULT_TOKENIZER}")
    if noise_root.resolve() != DEFAULT_NOISE.resolve():
        raise PreflightError(f"noise root must be fixed at {DEFAULT_NOISE}")
    _require_path(teacher, "teacher checkpoint", directory=False)
    _require_path(data, "LRS3 data", directory=True)
    _require_path(tokenizer, "tokenizer", directory=False)
    for fixed_path in FIXED_FILE_SHA256:
        _require_path(fixed_path, "fixed protocol artifact", directory=False)
        _require_digest(fixed_path)
    if float(_get(config, "task.noise_prob", 0.0)):
        _require_path(noise_root, "MUSAN noise manifest root", directory=True)
        _require_path(noise_root / "train.tsv", "training noise manifest", directory=False)
    if not _within(output_root, OUTPUT_ROOT):
        raise PreflightError(
            f"output root must remain within canonical isolation root {OUTPUT_ROOT}"
        )
    if not _within(run_dir, output_root):
        raise PreflightError(f"run directory escapes isolated output root: {run_dir}")
    if experiment in SELECTED_EXPERIMENTS and selection_path is None:
        raise PreflightError(f"{experiment} requires --from-selection")
    if selection_path is not None and not _within(
        selection_path, output_root / "selection"
    ):
        raise PreflightError(
            f"selection lock must be below {output_root / 'selection'}"
        )
    validate_safe_config(config)
    return {
        "branch": branch,
        "commit": commit,
        "remote": _git("remote", "get-url", "origin", check=False) or None,
        "tracked_status": [],
    }


def _parse_override_value(value: str) -> Any:
    if value == "":
        return ""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_hydra_overrides(
    config: Mapping[str, Any], overrides: Iterable[str]
) -> Dict[str, Any]:
    """Materialize the subset of Hydra's dotted assignment syntax we emit."""

    output: Dict[str, Any] = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise PreflightError(f"unsupported Hydra override: {override!r}")
        dotted, raw_value = override.split("=", 1)
        dotted = dotted.lstrip("+")
        components = dotted.split(".")
        target: MutableMapping[str, Any] = output
        for component in components[:-1]:
            current = target.get(component)
            if not isinstance(current, MutableMapping):
                current = {}
                target[component] = current
            target = current
        target[components[-1]] = _parse_override_value(raw_value)
    return output


def _delete_dotted(config: MutableMapping[str, Any], dotted: str) -> None:
    components = dotted.split(".")
    current: Any = config
    for component in components[:-1]:
        if not isinstance(current, MutableMapping):
            return
        current = current.get(component)
    if isinstance(current, MutableMapping):
        current.pop(components[-1], None)


def _control_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        output = []
        for key in sorted(set(left).union(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                output.append(child)
            else:
                output.extend(_control_differences(left[key], right[key], child))
        return output
    return [] if left == right else [prefix]


def validate_selected_control(
    experiment: str,
    config: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]],
) -> None:
    """Ensure D2/E2 change only the protocol fields authorized by the design."""

    if experiment not in {"d2_selected_conformer", "e2_selected_noisy"}:
        return
    if selection is None:
        raise PreflightError(f"{experiment} requires a frozen selection")
    selected_manifest = ManifestStore(selection["selected_manifest"]).read()
    selected_config = copy.deepcopy(
        selected_manifest["immutable"]["resolved_config"]
    )
    destination_config = copy.deepcopy(dict(config))
    allowed = {"hydra"}
    if experiment == "d2_selected_conformer":
        allowed.update(
            {
                "model.student_arch",
                "model.student_ffn_dim",
            }
        )
        match = selection.get("conformer_match")
        if not isinstance(match, Mapping):
            raise PreflightError("D2 requires frozen Conformer matching evidence")
        if int(_get(config, "model.student_ffn_dim", -1)) != int(
            match["selected_student_ffn_dim"]
        ):
            raise PreflightError(
                "D2 FFN does not match the frozen Conformer matching artifact"
            )
    else:
        allowed.update(
            {
                "task.noise_prob",
                "task.noise_snr",
                "task.noise_num",
                "task.noise_method",
                "task.noise_wav",
                "task.distillation_noise_prob",
                "task.distillation_noise_snr",
                "task.distillation_noise_method",
                "task.distillation_noise_manifest_root",
                "task.distillation_noise_train_only",
                "model.distillation_noise_prob",
                "model.distillation_noise_snr",
                "model.distillation_noise_method",
                "model.distillation_noise_manifest_root",
                "model.distillation_noise_train_only",
            }
        )
    for dotted in allowed:
        if dotted == "hydra":
            selected_config.pop("hydra", None)
            destination_config.pop("hydra", None)
        else:
            _delete_dotted(selected_config, dotted)
            _delete_dotted(destination_config, dotted)
    differences = _control_differences(selected_config, destination_config)
    if differences:
        raise PreflightError(
            f"{experiment} changes fields outside its controlled ablation: "
            + ", ".join(differences[:20])
        )


def deduplicate_overrides(overrides: Iterable[str]) -> list[str]:
    """Keep only the final value for each Hydra key."""

    reversed_output = []
    seen = set()
    for override in reversed(list(overrides)):
        key = override.split("=", 1)[0].lstrip("+")
        if key not in seen:
            reversed_output.append(override)
            seen.add(key)
    return list(reversed(reversed_output))


def _base_overrides(args: argparse.Namespace, run_path: Path) -> list[str]:
    overrides = [
        f"task.data={args.data}",
        f"task.label_dir={args.data}",
        f"task.tokenizer_bpe_model={args.tokenizer}",
        f"model.teacher_path={args.teacher}",
        f"distributed_training.distributed_world_size={args.gpus}",
        f"distributed_training.nprocs_per_node={args.gpus}",
        f"dataset.max_tokens={args.max_tokens}",
        f"optimization.update_freq=[{args.update_freq}]",
        f"dataset.num_workers={args.workers}",
        f"common.seed={args.seed}",
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        f"hydra.run.dir={run_path}",
    ]
    if args.experiment == "e2_selected_noisy":
        overrides.extend(
            (
                "task.noise_prob=0.25",
                "task.noise_snr=0",
                "task.noise_num=1",
                "task.noise_method=rms",
                f"task.noise_wav={args.noise_root}",
                "task.distillation_noise_prob=0.25",
                "task.distillation_noise_snr=0",
                "task.distillation_noise_method=rms",
                f"task.distillation_noise_manifest_root={args.noise_root}",
                "task.distillation_noise_train_only=true",
                "model.distillation_noise_prob=0.25",
                "model.distillation_noise_snr=0",
                "model.distillation_noise_method=rms",
                f"model.distillation_noise_manifest_root={args.noise_root}",
                "model.distillation_noise_train_only=true",
            )
        )
    return overrides


def _encoder_command(
    args: argparse.Namespace,
    *,
    stage: str,
    selection_overrides: Sequence[str],
) -> tuple[list[str], Path]:
    if stage in ("encoder", "stage1"):
        encoder_dir = (
            args.run_dir / "encoder" / "stage1"
            if args.experiment == "s2_optional_two_stage"
            else args.run_dir / "encoder"
        )
    elif stage == "stage2":
        encoder_dir = args.run_dir / "encoder" / "stage2"
    else:
        raise ValueError(stage)
    overrides = [
        *_base_overrides(args, encoder_dir),
        *selection_overrides,
    ]
    if stage in ("stage1",) or (
        stage == "encoder" and args.experiment == "s2_optional_two_stage"
    ):
        overrides.extend(
            (
                "optimization.max_update=50000",
                "optimization.lr=[0.002]",
                "lr_scheduler.warmup_updates=15000",
                "lr_scheduler.total_num_update=50000",
                "model.schedule_mode=two_stage_50k_25k",
                "model.max_update=50000",
                "model.warmup_updates=15000",
            )
        )
    elif stage == "stage2":
        stage1_checkpoint = args.run_dir / "encoder" / "stage1" / "checkpoints" / "checkpoint_last.pt"
        if not args.dry_run:
            _require_path(stage1_checkpoint, "S2 stage-1 checkpoint", directory=False)
        overrides.extend(
            (
                "optimization.max_update=25000",
                "optimization.lr=[0.0001]",
                "lr_scheduler.warmup_updates=5000",
                "lr_scheduler.total_num_update=25000",
                "model.schedule_mode=two_stage_50k_25k",
                "model.max_update=25000",
                "model.warmup_updates=5000",
                "model.initialization_policy=warm_start_distilled",
                f"checkpoint.finetune_from_model={stage1_checkpoint}",
            )
        )
    overrides.extend(args.override)
    command = [
        args.fairseq_train,
        "--config-dir",
        str(CONFIG_ROOT),
        "--config-name",
        f"{args.experiment}.yaml",
        *deduplicate_overrides(overrides),
    ]
    return command, encoder_dir


def _encoder_checkpoint(args: argparse.Namespace) -> Path:
    if args.experiment == "s2_optional_two_stage":
        return args.run_dir / "encoder" / "stage2" / "checkpoints" / "checkpoint_last.pt"
    return args.run_dir / "encoder" / "checkpoints" / "checkpoint_last.pt"


def _export_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    checkpoint = _encoder_checkpoint(args)
    _require_path(checkpoint, "encoder checkpoint", directory=False)
    output = args.run_dir / "export" / "student.pt"
    return (
        [
            sys.executable,
            "-m",
            "chapter3_distill_only.exporter",
            "--distilled-checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--verify",
        ],
        output,
    )


def _profile_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    checkpoint = _encoder_checkpoint(args)
    _require_path(checkpoint, "encoder checkpoint", directory=False)
    output = args.run_dir / "measurements" / "profile.json"
    return (
        [
            sys.executable,
            str(Path(__file__).with_name("profile_checkpoint.py")),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--user-dir",
            str(REPO_ROOT / "chapter3_distill_only"),
        ],
        output,
    )


def _finetune_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    student = args.run_dir / "export" / "student.pt"
    _require_path(student, "exported student checkpoint", directory=False)
    output = args.run_dir / "finetune"
    command = [
        args.fairseq_train,
        "--config-dir",
        str(REPO_ROOT / "avhubert" / "conf" / "av-finetune"),
        "--config-name",
        "base_noise_pt_noise_ft_433h.yaml",
        f"task.data={args.data}",
        f"task.label_dir={args.data}",
        f"task.tokenizer_bpe_model={args.tokenizer}",
        f"task.noise_wav={args.noise_root}",
        "task.noise_prob=0.25",
        "task.noise_snr=0",
        f"model.w2v_path={student}",
        f"distributed_training.distributed_world_size={args.gpus}",
        f"distributed_training.nprocs_per_node={args.gpus}",
        f"optimization.update_freq=[{args.finetune_update_freq}]",
        f"dataset.num_workers={args.workers}",
        f"common.seed={args.seed}",
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        f"hydra.run.dir={output}",
    ]
    _assert_command_rejects_itut_training(command, label="fine-tuning")
    return command, output


def _assert_command_rejects_itut_training(
    command: Sequence[str], *, label: str
) -> None:
    for token in command:
        if str(token).startswith("task.noise_method=itut") or str(token).startswith(
            "task.distillation_noise_method=itut"
        ):
            raise PreflightError(f"{label} must not use ITU-T training noise: {token}")


def _subset_measurement_key(subset: str) -> str:
    return "validation" if subset == "valid" else subset


def _decode_command_for_condition(
    args: argparse.Namespace,
    protocol: EvaluationProtocol,
    condition: EvaluationCondition,
    *,
    checkpoint: Path,
) -> tuple[EvaluationCondition, list[str], Path]:
    result_dir = (
        args.run_dir
        / "evaluation"
        / condition.output_relative(protocol_noise_method=protocol.noise_method)
    )
    command = [
        sys.executable,
        "-B",
        str(REPO_ROOT / "avhubert" / "infer_s2s.py"),
        "--config-dir",
        str(REPO_ROOT / "avhubert" / "conf"),
        "--config-name",
        "s2s_decode.yaml",
        f"dataset.gen_subset={condition.subset}",
        f"common_eval.path={checkpoint}",
        f"common_eval.results_path={result_dir}",
        "override.modalities=['audio','video']",
        f"hydra.run.dir={result_dir}",
        f"common.user_dir={REPO_ROOT / 'chapter3_distill_only'}",
        f"common.seed={protocol.evaluation_seed}",
    ]
    if condition.noise_type is None:
        for key in NOISE_OVERRIDE_KEYS:
            if any(str(token).startswith(f"{key}=") for token in command):
                raise PreflightError(
                    f"clean decode command must omit {key}: {condition.name}"
                )
    else:
        if condition.noise_method != protocol.noise_method:
            raise PreflightError(
                "reported evaluation cannot replace "
                f"override.noise_method={protocol.noise_method}"
            )
        if condition.noise_root is None or condition.snr_db is None:
            raise PreflightError(f"incomplete noisy condition: {condition.name}")
        command.extend(
            (
                f"override.noise_wav={condition.noise_root}",
                "override.noise_prob=1",
                f"override.noise_snr={condition.snr_db}",
                f"override.noise_method={protocol.noise_method}",
            )
        )
        if not any(
            str(token) == f"override.noise_method={protocol.noise_method}"
            for token in command
        ):
            raise PreflightError("noisy evaluation lost the protocol noise method")
    return condition, command, result_dir


def _decode_commands(
    args: argparse.Namespace,
    *,
    phase: str,
    require_checkpoint: bool = True,
    validate_artifacts: bool = True,
) -> list[tuple[EvaluationCondition, list[str], Path]]:
    protocol: EvaluationProtocol = args.evaluation_protocol_obj
    checkpoint = args.run_dir / "finetune" / "checkpoints" / "checkpoint_best.pt"
    if require_checkpoint:
        _require_path(checkpoint, "fine-tuning checkpoint", directory=False)
    elif not checkpoint.exists():
        checkpoint = Path("/nonexistent/finetune/checkpoints/checkpoint_best.pt")
    if validate_artifacts:
        validate_protocol_artifacts(protocol, phases=(phase,))
    return [
        _decode_command_for_condition(
            args, protocol, condition, checkpoint=checkpoint
        )
        for condition in protocol.conditions(phase)
    ]


def _wer_measurement(
    result_dir: Path,
    *,
    protocol: EvaluationProtocol,
    condition: EvaluationCondition,
) -> Dict[str, Any]:
    candidates = sorted(result_dir.glob("wer.*"))
    if not candidates:
        raise PreflightError(f"decoder produced no WER artifact in {result_dir}")
    artifact = candidates[-1]
    first_line = artifact.read_text(encoding="utf-8").splitlines()[0]
    try:
        value = float(first_line.split(":", 1)[1].strip().rstrip("%"))
    except (IndexError, ValueError) as exc:
        raise PreflightError(f"cannot parse WER from {artifact}: {first_line!r}") from exc
    return measurement_metadata(
        protocol,
        condition,
        value=value,
        artifact=artifact,
        artifact_sha256=sha256_file(artifact),
    )


def _checkpoint_num_updates(checkpoint: Path) -> int:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    history = state.get("optimizer_history")
    if not isinstance(history, list) or not history:
        raise PreflightError(
            f"checkpoint has no optimizer update history: {checkpoint}"
        )
    try:
        return int(history[-1]["num_updates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError(
            f"checkpoint has no valid final update: {checkpoint}"
        ) from exc


def _expected_stage_updates(args: argparse.Namespace, stage: str) -> int:
    if args.experiment != "s2_optional_two_stage":
        return 75000
    return 50000 if stage in {"encoder", "stage1"} else 25000


def _device_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
    try:
        return int(visible)
    except ValueError:
        return 0


def wandb_env_for_stage(experiment: str, stage: str) -> Dict[str, str]:
    """Name Fairseq wandb runs after the experiment instead of 'checkpoints'."""

    return {
        "WANDB_RUN_GROUP": experiment,
        "WANDB_NAME": f"{experiment}-{stage}",
    }


def run_command(
    command: Sequence[str],
    *,
    stage: str,
    run_dir: Path,
    manifest: ManifestStore,
    experiment: str,
) -> Dict[str, Any]:
    log_path = run_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    monotonic_start = time.monotonic()
    manifest.update(
        {
            "runtime": {
                "active_stage": stage,
                "commands": {stage: list(map(str, command))},
                "stage_started_at": {stage: started},
            },
            "artifacts": {"logs": {stage: str(log_path)}},
            "failure": None,
        }
    )
    env = os.environ.copy()
    env.update(wandb_env_for_stage(experiment, stage))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started}] $ {shlex.join(map(str, command))}\n")
        log.flush()
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        monitor = PeakMemoryMonitor(process.pid, device_index=_device_index()).start()
        returncode = process.wait()
        memory = monitor.stop().as_dict()
    duration = time.monotonic() - monotonic_start
    completed = utc_now()
    values = {
        "runtime": {
            "active_stage": None,
            "stage_completed_at": {stage: completed},
            "duration_seconds": {stage: duration},
            "returncodes": {stage: returncode},
        },
        "measurements": {"peak_memory": {stage: memory}},
    }
    manifest.update(values)
    if returncode:
        manifest.record_failure(
            stage=stage,
            message=f"command failed; see {log_path}",
            returncode=returncode,
        )
        raise RuntimeError(f"{stage} failed with exit code {returncode}; see {log_path}")
    return values


def _immutable_manifest(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_digest: str,
    hydra_overrides: Sequence[str],
    selection_record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    provenance = build_provenance(
        REPO_ROOT,
        baseline_branch=BASELINE_BRANCH,
        baseline_commit=BASELINE_COMMIT,
        old_repo=OLD_REPO,
        archived_configs=_archive_configs(),
    )
    checkpoint_hashes = {
        "teacher": sha256_file(args.teacher),
        "tokenizer": sha256_file(args.tokenizer),
        "data_train_manifest": sha256_file(args.data / "train.tsv"),
        "data_valid_manifest": sha256_file(args.data / "valid.tsv"),
        "data_test_manifest": sha256_file(args.data / "test.tsv"),
        "noise_train_manifest": sha256_file(args.noise_root / "train.tsv"),
    }
    protocol: EvaluationProtocol = args.evaluation_protocol_obj
    noise_prob = float(_get(config, "task.noise_prob", 0.0))
    return {
        "provenance": provenance,
        "experiment": args.experiment,
        "seed": args.seed,
        "evaluation_seed": protocol.evaluation_seed,
        "config_path": str(args.config_path),
        "config_sha256": sha256_file(args.config_path),
        "config_digest": config_digest,
        "resolved_config": config,
        "launcher_arguments": vars_for_manifest(args),
        "hydra_overrides": list(hydra_overrides),
        "selection": copy.deepcopy(selection_record),
        "paths": {
            "teacher_checkpoint": str(args.teacher),
            "data": str(args.data),
            "tokenizer": str(args.tokenizer),
            "noise_manifest_root": str(args.noise_root),
            "output_directory": str(args.run_dir),
            "evaluation_protocol": str(protocol.path),
        },
        "checkpoint_sha256": checkpoint_hashes,
        "noise_protocol": {
            "encoder_training": {
                "mode": "rms" if noise_prob else "clean",
                "noise_prob": noise_prob,
                "noise_method": (
                    str(_get(config, "task.noise_method", "rms"))
                    if noise_prob
                    else None
                ),
            },
            "downstream_finetuning": {
                "noise_prob": 0.25,
                "noise_snr": 0,
                "noise_num": 1,
                "noise_method": "rms",
            },
            "reported_validation": {
                "noise_method": protocol.noise_method,
                "phase": "screening",
            },
            "reported_final_inference": {
                "noise_method": protocol.noise_method,
                "phase": "final",
            },
        },
        "evaluation_protocol": protocol.with_manifest_hashes(),
    }


def vars_for_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        if key in {"stage", "dry_run", "evaluation_protocol_obj"}:
            continue
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _ensure_reusable_run(run_dir: Path, config_digest: str, commit: str) -> None:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return
    store = ManifestStore(run_dir)
    if not store.exists():
        raise PreflightError(f"refusing nonempty unowned run directory: {run_dir}")
    current = store.read()
    immutable = current["immutable"]
    existing_commit = _get(immutable, "provenance.implementation.commit")
    if immutable.get("config_digest") != config_digest or existing_commit != commit:
        raise PreflightError(
            f"run directory digest/commit mismatch; choose a new output: {run_dir}"
        )


def _selection_overrides(
    experiment: str,
    selection_path: Optional[Path],
) -> tuple[list[str], Optional[Dict[str, Any]]]:
    if selection_path is None:
        return [], None
    record = read_selection(selection_path)
    expected_kind = EXPECTED_SELECTION_KIND.get(experiment)
    if expected_kind is not None and record["kind"] != expected_kind:
        raise PreflightError(
            f"{experiment} requires a {expected_kind} selection, got {record['kind']}"
        )
    destination_arch = None
    if experiment == "d1_selected_transformer":
        destination_arch = "transformer"
    elif experiment == "d2_selected_conformer":
        destination_arch = "conformer"
    return (
        inherited_hydra_overrides(
            record,
            destination_arch=destination_arch,
            destination_experiment=experiment,
        ),
        record,
    )


def _record_actual_hydra_config(
    manifest: ManifestStore,
    *,
    stage: str,
    run_directory: Path,
) -> None:
    path = run_directory / ".hydra" / "config.yaml"
    _require_path(path, f"{stage} resolved Hydra config", directory=False)
    try:
        resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"cannot read resolved Hydra config {path}: {exc}") from exc
    manifest.update(
        {
            "artifacts": {
                "resolved_hydra_configs": {
                    stage: {"path": str(path), "sha256": sha256_file(path)}
                }
            },
            "runtime": {"actual_resolved_configs": {stage: resolved}},
        }
    )


def _dry_run_plan(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    selection_overrides: Sequence[str],
    config_digest: str,
) -> None:
    commands: Dict[str, Any] = {}
    requested = _expanded_stages(args)
    for stage in requested:
        try:
            if stage in ("encoder", "stage1", "stage2"):
                command, _ = _encoder_command(
                    args, stage=stage, selection_overrides=selection_overrides
                )
                _assert_command_rejects_itut_training(command, label=stage)
                commands[stage] = command
            elif stage == "export":
                commands[stage] = _export_command(args)[0]
                commands["profile"] = _profile_command(args)[0]
            elif stage == "finetune":
                commands[stage] = _finetune_command(args)[0]
            elif stage == "validate":
                commands[stage] = [
                    {
                        "condition": condition.name,
                        "subset": condition.subset,
                        "phase": condition.evaluation_phase,
                        "command": command,
                        "output": str(output),
                    }
                    for condition, command, output in _decode_commands(
                        args,
                        phase="screening",
                        require_checkpoint=False,
                        validate_artifacts=False,
                    )
                ]
            elif stage == "test":
                commands[stage] = [
                    {
                        "condition": condition.name,
                        "subset": condition.subset,
                        "phase": condition.evaluation_phase,
                        "command": command,
                        "output": str(output),
                    }
                    for condition, command, output in _decode_commands(
                        args,
                        phase="final",
                        require_checkpoint=False,
                        validate_artifacts=False,
                    )
                ]
        except (PreflightError, EvaluationProtocolError, ImportError) as exc:
            commands[stage] = {"blocked_until_dependency_exists": str(exc)}
    protocol: EvaluationProtocol = args.evaluation_protocol_obj
    print(
        json.dumps(
            {
                "dry_run": True,
                "experiment": args.experiment,
                "run_dir": str(args.run_dir),
                "config_digest": config_digest,
                "evaluation_protocol": {
                    "path": str(protocol.path),
                    "sha256": protocol.sha256,
                    "evaluation_seed": protocol.evaluation_seed,
                    "expected_final_condition_count": (
                        protocol.expected_final_condition_count
                    ),
                },
                "resolved_config": config,
                "selection_overrides": list(selection_overrides),
                "commands": commands,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _expanded_stages(args: argparse.Namespace) -> list[str]:
    if args.stage != "all":
        return [] if args.stage == "prepare" else [args.stage]
    if args.experiment == "s2_optional_two_stage":
        return ["stage1", "stage2", "export", "finetune", "validate"]
    return ["encoder", "export", "finetune", "validate"]


def _reuse_selected_alias(
    args: argparse.Namespace,
    selection_record: Mapping[str, Any],
) -> int:
    """Reuse the frozen source run for D1/E1/S1 instead of retraining it."""

    selected = ManifestStore(selection_record["selected_manifest"])
    manifest = selected.read()
    if args.dry_run:
        planned = {
            "dry_run": True,
            "experiment": args.experiment,
            "action": "reuse_selected_run",
            "selected_manifest": str(selected.path.resolve()),
            "selected_state": manifest["state"],
            "requested_stage": args.stage,
        }
        if args.stage in {"validate", "test"}:
            phase = "screening" if args.stage == "validate" else "final"
            args.run_dir = selected.path.parent
            planned["commands"] = {
                args.stage: [
                    {
                        "condition": condition.name,
                        "subset": condition.subset,
                        "phase": condition.evaluation_phase,
                        "command": command,
                        "output": str(output),
                    }
                    for condition, command, output in _decode_commands(
                        args,
                        phase=phase,
                        require_checkpoint=False,
                        validate_artifacts=False,
                    )
                ]
            }
            protocol: EvaluationProtocol = args.evaluation_protocol_obj
            planned["evaluation_protocol"] = {
                "path": str(protocol.path),
                "sha256": protocol.sha256,
                "evaluation_seed": protocol.evaluation_seed,
                "expected_final_condition_count": (
                    protocol.expected_final_condition_count
                ),
            }
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 0
    if args.stage == "test":
        if manifest["state"] == "test_complete":
            print(selected.path)
            return 0
        if manifest["state"] != "selection_frozen":
            raise ManifestError(
                f"test requires selection_frozen, found {manifest['state']}"
            )
        require_final_selection(
            args.from_selection,
            current_manifest=selected.path,
        )
        args.run_dir = selected.path.parent
        protocol: EvaluationProtocol = args.evaluation_protocol_obj
        artifacts: Dict[str, Dict[str, str]] = {"validation": {}, "test": {}}
        for condition, command, output in _decode_commands(args, phase="final"):
            stage_name = f"final_{condition.subset}_{condition.name}"
            run_command(
                command,
                stage=stage_name,
                run_dir=args.run_dir,
                manifest=selected,
                experiment=args.experiment,
            )
            measurement = _wer_measurement(
                output, protocol=protocol, condition=condition
            )
            subset_key = _subset_measurement_key(condition.subset)
            artifacts[subset_key][condition.name] = measurement["artifact"]
            selected.update(
                {
                    "measurements": {
                        "wer": {"final": {subset_key: {condition.name: measurement}}}
                    }
                }
            )
        selected.transition(
            "test_complete",
            values={"artifacts": {"final": artifacts}, "failure": None},
        )
    else:
        selected.update(
            {
                "runtime": {
                    "reused_as": {
                        args.experiment: {
                            "selection": str(args.from_selection),
                            "recorded_at": utc_now(),
                        }
                    }
                }
            }
        )
    print(selected.path)
    return 0


def execute(args: argparse.Namespace) -> int:
    config = load_composed_config(args.config_path)
    selection_overrides, selection_record = _selection_overrides(
        args.experiment, args.from_selection
    )
    provenance = preflight(
        experiment=args.experiment,
        config_path=args.config_path,
        config=config,
        teacher=args.teacher,
        data=args.data,
        tokenizer=args.tokenizer,
        noise_root=args.noise_root,
        output_root=args.output_root,
        run_dir=args.run_dir,
        selection_path=args.from_selection,
    )
    if args.experiment in ALIAS_EXPERIMENTS:
        if selection_record is None:
            raise PreflightError(f"{args.experiment} requires a selection record")
        return _reuse_selected_alias(args, selection_record)
    effective_overrides = _base_overrides(args, args.run_dir / "encoder")
    effective_overrides.extend(selection_overrides)
    effective_overrides.extend(args.override)
    effective_overrides = deduplicate_overrides(effective_overrides)
    effective_config = apply_hydra_overrides(config, effective_overrides)
    validate_safe_config(effective_config)
    validate_fixed_resolved_inputs(effective_config, args)
    validate_selected_control(
        args.experiment, effective_config, selection_record
    )
    digest_value = {
        "resolved_config": effective_config,
        "seed": args.seed,
        "selection_sha256": (
            selection_record.get("selection_sha256") if selection_record else None
        ),
        "selection_overrides": selection_overrides,
        "explicit_overrides": args.override,
    }
    config_digest = sha256_json(digest_value)
    _ensure_reusable_run(args.run_dir, config_digest, provenance["commit"])
    if args.dry_run:
        _dry_run_plan(args, effective_config, selection_overrides, config_digest)
        return 0
    hydra_overrides = effective_overrides
    manifest = ManifestStore(args.run_dir)
    manifest.create(
        _immutable_manifest(
            args,
            effective_config,
            config_digest,
            hydra_overrides,
            selection_record,
        )
    )
    if args.stage == "prepare":
        print(manifest.path)
        return 0

    for stage in _expanded_stages(args):
        state = manifest.read()["state"]
        if stage in ("encoder", "stage1", "stage2"):
            if state == "prepared":
                manifest.transition("encoder_running")
            elif state != "encoder_running":
                if state in ("encoder_complete", "exported", "finetune_running", "finetune_complete", "validation_complete", "selection_frozen", "test_complete"):
                    continue
                raise ManifestError(f"cannot run encoder from state {state}")
            command, encoder_dir = _encoder_command(
                args, stage=stage, selection_overrides=selection_overrides
            )
            _assert_command_rejects_itut_training(command, label=stage)
            run_command(
                command,
                stage=stage,
                run_dir=args.run_dir,
                manifest=manifest,
                experiment=args.experiment,
            )
            _record_actual_hydra_config(
                manifest,
                stage=stage,
                run_directory=encoder_dir,
            )
            checkpoint = encoder_dir / "checkpoints" / "checkpoint_last.pt"
            _require_path(checkpoint, f"{stage} checkpoint", directory=False)
            final_update = _checkpoint_num_updates(checkpoint)
            expected_update = _expected_stage_updates(args, stage)
            if final_update != expected_update:
                raise PreflightError(
                    f"{stage} checkpoint stopped at update {final_update}; "
                    f"expected {expected_update}"
                )
            is_final_encoder_stage = (
                args.experiment != "s2_optional_two_stage" or stage == "stage2"
            )
            manifest.update(
                {
                    "artifacts": {
                        "encoder_checkpoints": {
                            stage: {
                                "path": str(checkpoint),
                                "sha256": sha256_file(checkpoint),
                                "final_update": final_update,
                            }
                        }
                    },
                    "runtime": {"final_update": {stage: final_update}},
                    "failure": None,
                }
            )
            if is_final_encoder_stage:
                manifest.transition("encoder_complete")
        elif stage == "export":
            if state in ("exported", "finetune_running", "finetune_complete", "validation_complete", "selection_frozen", "test_complete"):
                continue
            if state != "encoder_complete":
                raise ManifestError(f"cannot export from state {state}")
            command, output = _export_command(args)
            run_command(
                command,
                stage=stage,
                run_dir=args.run_dir,
                manifest=manifest,
                experiment=args.experiment,
            )
            _require_path(output, "exported checkpoint", directory=False)
            profile_command, profile_output = _profile_command(args)
            run_command(
                profile_command,
                stage="profile",
                run_dir=args.run_dir,
                manifest=manifest,
                experiment=args.experiment,
            )
            try:
                profile = json.loads(profile_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PreflightError(f"cannot read numeric profile {profile_output}: {exc}") from exc
            manifest.transition(
                "exported",
                values={
                    "artifacts": {
                        "exported_checkpoint": {
                            "path": str(output),
                            "sha256": sha256_file(output),
                        },
                        "profile": str(profile_output),
                    },
                    "measurements": {"profile": profile},
                    "failure": None,
                },
            )
        elif stage == "finetune":
            if state in ("finetune_complete", "validation_complete", "selection_frozen", "test_complete"):
                continue
            if state == "exported":
                manifest.transition("finetune_running")
            elif state != "finetune_running":
                raise ManifestError(f"cannot fine-tune from state {state}")
            command, output = _finetune_command(args)
            run_command(
                command,
                stage=stage,
                run_dir=args.run_dir,
                manifest=manifest,
                experiment=args.experiment,
            )
            checkpoint = output / "checkpoints" / "checkpoint_best.pt"
            _require_path(checkpoint, "fine-tuning checkpoint", directory=False)
            manifest.transition(
                "finetune_complete",
                values={
                    "artifacts": {
                        "finetune_checkpoint": {
                            "path": str(checkpoint),
                            "sha256": sha256_file(checkpoint),
                        }
                    },
                    "failure": None,
                },
            )
        elif stage == "validate":
            if state in ("validation_complete", "selection_frozen", "test_complete"):
                continue
            if state != "finetune_complete":
                raise ManifestError(f"cannot validate from state {state}")
            protocol = args.evaluation_protocol_obj
            artifacts = {}
            for condition, command, output in _decode_commands(
                args, phase="screening"
            ):
                run_command(
                    command,
                    stage=f"validate_{condition.name}",
                    run_dir=args.run_dir,
                    manifest=manifest,
                    experiment=args.experiment,
                )
                measurement = _wer_measurement(
                    output, protocol=protocol, condition=condition
                )
                artifacts[condition.name] = measurement["artifact"]
                manifest.update(
                    {
                        "measurements": {
                            "wer": {"validation": {condition.name: measurement}}
                        }
                    }
                )
            manifest.transition(
                "validation_complete",
                values={"artifacts": {"validation": artifacts}, "failure": None},
            )
        elif stage == "test":
            if state == "test_complete":
                continue
            if state != "selection_frozen":
                raise ManifestError(f"test requires selection_frozen, found {state}")
            require_final_selection(
                args.final_selection, current_manifest=manifest.path
            )
            protocol = args.evaluation_protocol_obj
            artifacts = {"validation": {}, "test": {}}
            for condition, command, output in _decode_commands(args, phase="final"):
                run_command(
                    command,
                    stage=f"final_{condition.subset}_{condition.name}",
                    run_dir=args.run_dir,
                    manifest=manifest,
                    experiment=args.experiment,
                )
                measurement = _wer_measurement(
                    output, protocol=protocol, condition=condition
                )
                subset_key = _subset_measurement_key(condition.subset)
                artifacts[subset_key][condition.name] = measurement["artifact"]
                manifest.update(
                    {
                        "measurements": {
                            "wer": {
                                "final": {subset_key: {condition.name: measurement}}
                            }
                        }
                    }
                )
            manifest.transition(
                "test_complete",
                values={"artifacts": {"final": artifacts}, "failure": None},
            )
    print(manifest.path)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    available = sorted(
        path.stem
        for path in CONFIG_ROOT.glob("*.yaml")
        if path.stem != "base_continuous_75k"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=available)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--stage", choices=STAGES, default="prepare")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-selection", type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--noise-root", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=DEFAULT_EVALUATION_PROTOCOL,
        help="Versioned ITU-T post-fine-tuning evaluation protocol YAML",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help=(
            "Fixed seed for reported validation/inference noise selection; "
            "defaults to the protocol evaluation_seed"
        ),
    )
    parser.add_argument(
        "--final-selection",
        type=Path,
        default=OUTPUT_ROOT / "selection" / "final.json",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--finetune-update-freq", type=int, default=8)
    parser.add_argument("--fairseq-train", default="fairseq-hydra-train")
    args = parser.parse_args(argv)
    args.config_path = CONFIG_ROOT / f"{args.experiment}.yaml"
    args.output_root = args.output_root.resolve()
    args.run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else (args.output_root / args.experiment / f"seed_{args.seed}").resolve()
    )
    for name in ("teacher", "data", "tokenizer", "noise_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.from_selection:
        args.from_selection = args.from_selection.resolve()
    args.final_selection = args.final_selection.resolve()
    args.evaluation_protocol = args.evaluation_protocol.resolve()
    try:
        protocol = load_evaluation_protocol(args.evaluation_protocol)
    except EvaluationProtocolError as exc:
        parser.error(str(exc))
    if args.evaluation_seed is not None:
        protocol = EvaluationProtocol(
            path=protocol.path,
            schema_version=protocol.schema_version,
            noise_method=protocol.noise_method,
            evaluation_seed=args.evaluation_seed,
            speech_level_dbov=protocol.speech_level_dbov,
            noise_roots=protocol.noise_roots,
            screening=protocol.screening,
            final=protocol.final,
            raw=protocol.raw,
        )
    else:
        args.evaluation_seed = protocol.evaluation_seed
    args.evaluation_protocol_obj = protocol
    if args.gpus != 1:
        parser.error("the controlled protocol requires exactly one GPU")
    if args.stage in ("stage1", "stage2") and args.experiment != "s2_optional_two_stage":
        parser.error("--stage stage1/stage2 is only valid for s2_optional_two_stage")
    if (
        args.max_tokens <= 0
        or args.update_freq <= 0
        or args.workers < 0
        or args.finetune_update_freq <= 0
    ):
        parser.error("batch and worker arguments must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return execute(parse_args(argv))
    except (
        ManifestError,
        PreflightError,
        EvaluationProtocolError,
        RuntimeError,
        ValueError,
        ImportError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
