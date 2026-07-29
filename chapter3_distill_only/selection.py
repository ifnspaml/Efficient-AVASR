"""Immutable scientific-selection locks for the Chapter 3 experiment sequence."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from .manifest import (
    MANIFEST_FILENAME,
    RUN_KIND_EVALUATION,
    ManifestError,
    ManifestStore,
    atomic_write_json,
    manifest_run_kind,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
)
from .profiling import validate_config_profile


SELECTION_SCHEMA = "chapter3-distill-selection/v1"
SELECTION_KINDS = ("c_to_d", "d_to_e", "final")

# Only scientific design fields are inherited. Paths, teacher state, schedule, and
# output settings stay owned by the destination experiment config.
MODEL_INHERITABLE_FIELDS = (
    "student_arch",
    "student_depth",
    "student_embed_dim",
    "student_ffn_dim",
    "student_attention_heads",
    "student_dropout",
    "student_attention_dropout",
    "student_activation_dropout",
    "student_layerdrop",
    "student_layer_norm_first",
    "student_dropout_input",
    "student_conv_pos",
    "student_conv_pos_groups",
    "student_conformer_kernel",
    "student_conformer_attention_type",
    "student_conformer_position_type",
    "distill_head_mode",
    "teacher_target_layers",
    "student_match_layers",
    "initialization_policy",
    "prediction_head_hidden_dim",
    "modality_dropout",
    "audio_dropout",
    "feature_grad_mult",
    "l1_weight",
    "l2_weight",
    "cosine_weight",
    "cosine_type",
    "feature_penalty_weight",
    "distill_loss_type",
)
CRITERION_INHERITABLE_FIELDS = (
    "distill_loss_type",
    "l1_weight",
    "l2_weight",
    "cosine_weight",
    "cos_weight",
    "cosine_type",
    "cos_type",
    "feature_penalty_weight",
)
SECTION_INHERITABLE_FIELDS = {
    "common": (
        "amp",
        "fp16",
        "log_format",
        "log_interval",
        "wandb_project",
    ),
    "task": (
        "data",
        "label_dir",
        "tokenizer_bpe_model",
        "normalize",
        "labels",
        "single_target",
        "fine_tuning",
        "stack_order_audio",
        "tokenizer_bpe_name",
        "max_sample_size",
        "modalities",
        "image_aug",
        "pad_audio",
        "random_crop",
    ),
    "dataset": (
        "num_workers",
        "max_tokens",
        "validate_after_updates",
        "validate_interval",
        "train_subset",
        "valid_subset",
    ),
    "optimization": (
        "max_update",
        "clip_norm",
        "update_freq",
        "lr",
    ),
    "optimizer": (
        "_name",
        "adam_betas",
        "adam_eps",
        "weight_decay",
    ),
    "lr_scheduler": (
        "_name",
        "power",
        "warmup_updates",
        "total_num_update",
    ),
}
TASK_NOISE_FIELDS = (
    "noise_prob",
    "noise_snr",
    "noise_num",
    "noise_method",
    "noise_wav",
    "distillation_noise_prob",
    "distillation_noise_snr",
    "distillation_noise_method",
    "distillation_noise_manifest_root",
    "distillation_noise_train_only",
)
MODEL_NOISE_FIELDS = (
    "distillation_noise_prob",
    "distillation_noise_snr",
    "distillation_noise_method",
    "distillation_noise_manifest_root",
    "distillation_noise_train_only",
)


class SelectionError(ManifestError):
    """Raised for invalid or conflicting scientific selection locks."""


@contextmanager
def _selection_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _manifest_path(path: os.PathLike[str] | str) -> Path:
    candidate = Path(path).resolve()
    if candidate.name != MANIFEST_FILENAME and candidate.suffix != ".json":
        candidate = candidate / MANIFEST_FILENAME
    return candidate


def _validation_snapshot(
    manifest: Mapping[str, Any], manifest_path: Path
) -> Dict[str, Dict[str, Any]]:
    try:
        validation = manifest["measurements"]["wer"]["validation"]
    except (KeyError, TypeError) as exc:
        try:
            validation = manifest["measurements"]["wer"]["valid"]
        except (KeyError, TypeError) as nested_exc:
            raise SelectionError(
                f"candidate {manifest_path} has no validation WER measurements"
            ) from nested_exc
    snapshot: Dict[str, Dict[str, Any]] = {}
    for condition in ("clean", "babble_0db"):
        measurement = validation.get(condition)
        if not isinstance(measurement, Mapping):
            raise SelectionError(
                f"candidate {manifest_path} lacks {condition} validation WER"
            )
        artifact = Path(str(measurement.get("artifact", ""))).resolve()
        recorded_hash = str(measurement.get("artifact_sha256", ""))
        try:
            value = float(measurement["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionError(
                f"candidate {manifest_path} has invalid {condition} WER"
            ) from exc
        if not artifact.is_file():
            raise SelectionError(
                f"candidate {manifest_path} WER artifact is missing: {artifact}"
            )
        actual_hash = sha256_file(artifact)
        if not recorded_hash or actual_hash != recorded_hash:
            raise SelectionError(
                f"candidate {manifest_path} WER artifact digest mismatch: {artifact}"
            )
        snapshot[condition] = {
            "value": value,
            "artifact": str(artifact),
            "artifact_sha256": actual_hash,
        }
    return snapshot


def _metric_condition(metric: str) -> str:
    lowered = metric.lower()
    if "test" in lowered or (
        "validation" not in lowered and "valid" not in lowered
    ):
        raise SelectionError(
            "scientific selections must name a validation-only metric"
        )
    if "babble" in lowered and ("0db" in lowered or "0_db" in lowered):
        return "babble_0db"
    if "clean" in lowered:
        return "clean"
    raise SelectionError(
        "selection metric must identify clean or babble_0db validation WER"
    )


def _load_conformer_match(path: os.PathLike[str] | str) -> Dict[str, Any]:
    match_path = Path(path).resolve()
    match = read_json(match_path)
    if match.get("schema_version") != "chapter3-conformer-match/v1":
        raise SelectionError(f"invalid Conformer match artifact: {match_path}")
    selected = match.get("selected")
    if not isinstance(selected, Mapping):
        raise SelectionError("Conformer match artifact has no selected candidate")
    try:
        ffn_dim = int(selected["student_ffn_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionError("Conformer match selected FFN is invalid") from exc
    if ffn_dim <= 0 or ffn_dim % 128:
        raise SelectionError(
            "selected Conformer FFN must be a positive multiple of 128"
        )
    if match.get("hydra_override") != f"model.student_ffn_dim={ffn_dim}":
        raise SelectionError("Conformer match Hydra override is inconsistent")

    profile_entries = [
        (
            match.get("target_profile_path"),
            match.get("target_profile_sha256"),
            "transformer",
            match.get("target"),
        )
    ]
    candidates = match.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SelectionError("Conformer match must retain every scored candidate")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise SelectionError("invalid Conformer match candidate")
        profile_entries.append(
            (
                candidate.get("profile_path"),
                candidate.get("profile_sha256"),
                "conformer",
                candidate,
            )
        )
    loaded_profiles = []
    for raw_path, recorded_hash, expected_arch, embedded in profile_entries:
        profile_path = Path(str(raw_path or "")).resolve()
        if not profile_path.is_file() or sha256_file(profile_path) != recorded_hash:
            raise SelectionError(
                f"Conformer input profile missing or changed: {profile_path}"
            )
        profile = read_json(profile_path)
        try:
            validated = validate_config_profile(
                profile, expected_arch=expected_arch
            )
        except ValueError as exc:
            raise SelectionError(
                f"invalid {expected_arch} profile {profile_path}: {exc}"
            ) from exc
        if not isinstance(embedded, Mapping):
            raise SelectionError(
                f"Conformer match does not embed {profile_path}"
            )
        for field in (
            "student_arch",
            "student_depth",
            "student_embed_dim",
            "student_ffn_dim",
            "student_attention_heads",
            "deployed_backbone_parameters",
            "flops",
            "source_manifest",
            "resolved_config",
        ):
            if embedded.get(field) != profile.get(field):
                raise SelectionError(
                    f"Conformer match embedded {field} differs from "
                    f"profile {profile_path}"
                )
        loaded_profiles.append(
            {
                "path": str(profile_path),
                "sha256": recorded_hash,
                "profile": profile,
                "validated": validated,
            }
        )
    if match.get("source_manifest") != loaded_profiles[0]["validated"][
        "source_manifest"
    ]:
        raise SelectionError(
            "Conformer match source identity differs from its target profile"
        )
    selected_profile_path = str(
        Path(str(selected.get("profile_path", ""))).resolve()
    )
    selected_profile = next(
        (
            entry
            for entry in loaded_profiles[1:]
            if entry["path"] == selected_profile_path
        ),
        None,
    )
    if selected_profile is None:
        raise SelectionError(
            "selected Conformer candidate is not one of the sealed profiles"
        )
    if selected_profile["validated"]["student_ffn_dim"] != ffn_dim:
        raise SelectionError(
            "selected Conformer FFN differs from its actual profile"
        )
    return {
        "path": str(match_path),
        "sha256": sha256_file(match_path),
        "selected_student_ffn_dim": ffn_dim,
        "hydra_override": match["hydra_override"],
        "record": match,
        "_profiles": loaded_profiles,
    }


def _assert_config_value(
    *,
    selected: Mapping[str, Any],
    profiled: Mapping[str, Any],
    section: str,
    field: str,
    profile_path: str,
) -> None:
    selected_section = selected.get(section)
    if not isinstance(selected_section, Mapping) or field not in selected_section:
        return
    profiled_section = profiled.get(section)
    if (
        not isinstance(profiled_section, Mapping)
        or profiled_section.get(field) != selected_section[field]
    ):
        raise SelectionError(
            f"profile {profile_path} does not preserve selected "
            f"{section}.{field}"
        )


def _verify_conformer_match_binding(
    match: Mapping[str, Any],
    *,
    selected_path: Path,
    selected_manifest: Mapping[str, Any],
) -> None:
    """Prove every matching profile was derived from the selected C run."""

    expected_source = {
        "path": str(selected_path.resolve()),
        "immutable_sha256": selected_manifest["immutable_sha256"],
        "config_digest": selected_manifest["immutable"].get("config_digest"),
    }
    if match.get("record", {}).get("source_manifest") != expected_source:
        raise SelectionError(
            "Conformer matching artifact is not bound to the selected C manifest"
        )
    selected_config = _resolved_config(selected_manifest)
    selected_model = selected_config.get("model")
    if not isinstance(selected_model, Mapping):
        raise SelectionError(
            "selected C manifest has no immutable resolved model config"
        )
    profiles = match.get("_profiles")
    if not isinstance(profiles, list) or len(profiles) < 2:
        raise SelectionError("Conformer matching evidence has no candidate profiles")
    for index, entry in enumerate(profiles):
        validated = entry["validated"]
        profile = entry["profile"]
        profile_path = entry["path"]
        if validated["source_manifest"] != expected_source:
            raise SelectionError(
                f"profile {profile_path} is not bound to the selected C manifest"
            )
        profiled_config = validated["resolved_config"]
        profiled_model = profiled_config["model"]
        if index == 0:
            required_model_fields = MODEL_INHERITABLE_FIELDS
            if profiled_model.get("student_arch") != "transformer":
                raise SelectionError("Conformer target profile must be Transformer")
        else:
            required_model_fields = tuple(
                field
                for field in MODEL_INHERITABLE_FIELDS
                if field not in {"student_arch", "student_ffn_dim"}
            )
            if profiled_model.get("student_arch") != "conformer":
                raise SelectionError(
                    f"candidate profile {profile_path} is not Conformer"
                )
        for field in required_model_fields:
            _assert_config_value(
                selected=selected_config,
                profiled=profiled_config,
                section="model",
                field=field,
                profile_path=profile_path,
            )
        for field in CRITERION_INHERITABLE_FIELDS:
            _assert_config_value(
                selected=selected_config,
                profiled=profiled_config,
                section="criterion",
                field=field,
                profile_path=profile_path,
            )
        for section, fields in SECTION_INHERITABLE_FIELDS.items():
            for field in fields:
                _assert_config_value(
                    selected=selected_config,
                    profiled=profiled_config,
                    section=section,
                    field=field,
                    profile_path=profile_path,
                )
        for field in TASK_NOISE_FIELDS:
            _assert_config_value(
                selected=selected_config,
                profiled=profiled_config,
                section="task",
                field=field,
                profile_path=profile_path,
            )
        for field in MODEL_NOISE_FIELDS:
            _assert_config_value(
                selected=selected_config,
                profiled=profiled_config,
                section="model",
                field=field,
                profile_path=profile_path,
            )


def create_selection(
    path: os.PathLike[str] | str,
    *,
    kind: str,
    candidates: Iterable[os.PathLike[str] | str],
    selected: os.PathLike[str] | str,
    metric: str,
    rationale: str,
    approver: str,
    materialized_overrides: Optional[Sequence[str]] = None,
    overrides_by_experiment: Optional[Mapping[str, Sequence[str]]] = None,
    conformer_match: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Create a write-once selection record.

    An identical retry is accepted. Any attempted mutation requires a new path,
    preserving the decision trail used to gate later experiments.
    """

    if kind not in SELECTION_KINDS:
        raise SelectionError(f"selection kind must be one of {SELECTION_KINDS}")
    metric_condition = _metric_condition(metric)
    if kind == "c_to_d" and conformer_match is None:
        raise SelectionError(
            "c_to_d selection requires a hashed Conformer matching artifact"
        )
    conformer_match_record = (
        _load_conformer_match(conformer_match)
        if conformer_match is not None
        else None
    )
    candidate_paths = tuple(_manifest_path(item) for item in candidates)
    selected_path = _manifest_path(selected)
    if selected_path not in candidate_paths:
        raise SelectionError("selected manifest must be present in candidates")
    destination = Path(path).resolve()
    static_request = {
        "kind": kind,
        "metric": metric,
        "rationale": rationale,
        "approver": approver,
        "candidate_paths": [str(item) for item in candidate_paths],
        "selected_manifest": str(selected_path),
        "materialized_overrides": list(materialized_overrides or ()),
        "overrides_by_experiment": {
            str(experiment): list(overrides)
            for experiment, overrides in (overrides_by_experiment or {}).items()
        },
        "conformer_match": (
            {
                "path": conformer_match_record["path"],
                "sha256": conformer_match_record["sha256"],
            }
            if conformer_match_record is not None
            else None
        ),
    }

    def existing_matches(existing: Mapping[str, Any]) -> bool:
        existing_request = {
            "kind": existing.get("kind"),
            "metric": existing.get("metric"),
            "rationale": existing.get("rationale"),
            "approver": existing.get("approver"),
            "candidate_paths": [
                item.get("manifest") for item in existing.get("candidates", ())
            ],
            "selected_manifest": existing.get("selected_manifest"),
            "materialized_overrides": existing.get("materialized_overrides", []),
            "overrides_by_experiment": existing.get("overrides_by_experiment", {}),
            "conformer_match": (
                {
                    "path": existing["conformer_match"].get("path"),
                    "sha256": existing["conformer_match"].get("sha256"),
                }
                if isinstance(existing.get("conformer_match"), Mapping)
                else None
            ),
        }
        return existing_request == static_request

    if destination.exists():
        existing = read_selection(destination)
        if not existing_matches(existing):
            raise SelectionError(f"selection lock is immutable: {destination}")
        return existing

    candidate_records = []
    for candidate_path in candidate_paths:
        manifest = ManifestStore(candidate_path).read()
        if manifest["state"] not in (
            "validation_complete",
            "selection_frozen",
            "evaluation_complete",
        ):
            raise SelectionError(
                f"candidate {candidate_path} has not completed validation"
            )
        candidate_records.append(
            {
                "manifest": str(candidate_path),
                "immutable_sha256": manifest["immutable_sha256"],
                "state_at_selection": manifest["state"],
                "validation_wer": _validation_snapshot(
                    manifest, candidate_path
                ),
            }
        )
    selected_manifest = ManifestStore(selected_path).read()
    selected_finetune = selected_manifest.get("artifacts", {}).get(
        "finetune_checkpoint"
    )
    selected_finetune_manifest = selected_path
    if manifest_run_kind(selected_manifest) == RUN_KIND_EVALUATION:
        parent = selected_manifest["immutable"].get("parent_artifact")
        if not isinstance(parent, Mapping):
            raise SelectionError(
                "selected evaluation has no fine-tuning parent artifact"
            )
        selected_finetune_manifest = _manifest_path(
            parent.get("parent_finetune_manifest", "")
        )
        selected_finetune = {
            "path": parent.get("checkpoint_path"),
            "sha256": parent.get("checkpoint_sha256"),
        }
    selected_finetune_identity = None
    if isinstance(selected_finetune, Mapping):
        checkpoint = Path(str(selected_finetune.get("path", ""))).resolve()
        if checkpoint.is_file():
            checkpoint_hash = sha256_file(checkpoint)
            recorded_hash = selected_finetune.get("sha256")
            if recorded_hash and recorded_hash != checkpoint_hash:
                raise SelectionError(
                    f"selected fine-tuning checkpoint changed: {checkpoint}"
                )
            selected_finetune_identity = {
                "manifest": str(selected_finetune_manifest),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
            }
    if conformer_match_record is not None:
        _verify_conformer_match_binding(
            conformer_match_record,
            selected_path=selected_path,
            selected_manifest=selected_manifest,
        )
    public_conformer_match = (
        {
            key: value
            for key, value in conformer_match_record.items()
            if not key.startswith("_")
        }
        if conformer_match_record is not None
        else None
    )
    record: Dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "kind": kind,
        "created_at": utc_now(),
        "metric": metric,
        "metric_condition": metric_condition,
        "rationale": rationale,
        "approver": approver,
        "candidates": candidate_records,
        "selected_manifest": str(selected_path),
        "selected_immutable_sha256": selected_manifest["immutable_sha256"],
        "selected_finetune_artifact": selected_finetune_identity,
        "selected_metric_value": next(
            candidate["validation_wer"][metric_condition]["value"]
            for candidate in candidate_records
            if candidate["manifest"] == str(selected_path)
        ),
        "conformer_match": public_conformer_match,
        "materialized_overrides": list(materialized_overrides or ()),
        "overrides_by_experiment": {
            str(experiment): list(overrides)
            for experiment, overrides in (overrides_by_experiment or {}).items()
        },
    }
    record["selection_sha256"] = sha256_json(
        {key: value for key, value in record.items() if key != "selection_sha256"}
    )
    with _selection_lock(destination):
        if destination.exists():
            existing = read_selection(destination)
            if not existing_matches(existing):
                raise SelectionError(f"selection lock is immutable: {destination}")
            return existing
        atomic_write_json(destination, record)
    return record


def read_selection(path: os.PathLike[str] | str) -> Dict[str, Any]:
    record = read_json(path)
    if record.get("schema_version") != SELECTION_SCHEMA:
        raise SelectionError(f"unsupported selection schema in {path}")
    if record.get("kind") not in SELECTION_KINDS:
        raise SelectionError(f"invalid selection kind in {path}")
    _metric_condition(str(record.get("metric", "")))
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SelectionError("selection record has no candidates")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise SelectionError("invalid candidate in selection record")
        candidate_path = _manifest_path(candidate.get("manifest", ""))
        candidate_manifest = ManifestStore(candidate_path).read()
        if candidate_manifest["immutable_sha256"] != candidate.get(
            "immutable_sha256"
        ):
            raise SelectionError(
                f"candidate manifest changed since selection: {candidate_path}"
            )
        current_snapshot = _validation_snapshot(
            candidate_manifest, candidate_path
        )
        if current_snapshot != candidate.get("validation_wer"):
            raise SelectionError(
                f"candidate validation evidence changed: {candidate_path}"
            )
    selected_manifest = _manifest_path(record.get("selected_manifest", ""))
    manifest = ManifestStore(selected_manifest).read()
    if manifest["immutable_sha256"] != record.get("selected_immutable_sha256"):
        raise SelectionError("selected manifest has changed since selection was frozen")
    identity = record.get("selected_finetune_artifact")
    if identity is not None:
        if not isinstance(identity, Mapping):
            raise SelectionError("selected fine-tuning artifact identity is invalid")
        identity_manifest = _manifest_path(identity.get("manifest", ""))
        if manifest_run_kind(manifest) == RUN_KIND_EVALUATION:
            parent = manifest["immutable"].get("parent_artifact", {})
            expected_identity_manifest = _manifest_path(
                parent.get("parent_finetune_manifest", "")
            )
        else:
            expected_identity_manifest = selected_manifest
        if identity_manifest != expected_identity_manifest:
            raise SelectionError(
                "selected fine-tuning artifact names another manifest"
            )
        checkpoint = Path(str(identity.get("checkpoint", ""))).resolve()
        expected_hash = str(identity.get("checkpoint_sha256", ""))
        if (
            not checkpoint.is_file()
            or not expected_hash
            or sha256_file(checkpoint) != expected_hash
        ):
            raise SelectionError(
                "selected fine-tuning checkpoint is missing or changed"
            )
    if record["kind"] == "c_to_d":
        conformer_match = record.get("conformer_match")
        if not isinstance(conformer_match, Mapping):
            raise SelectionError("c_to_d selection lacks Conformer matching evidence")
        current_match = _load_conformer_match(conformer_match.get("path", ""))
        _verify_conformer_match_binding(
            current_match,
            selected_path=selected_manifest,
            selected_manifest=manifest,
        )
        current_public_match = {
            key: value
            for key, value in current_match.items()
            if not key.startswith("_")
        }
        if current_public_match != conformer_match:
            raise SelectionError("Conformer matching evidence has changed")
    expected = sha256_json(
        {key: value for key, value in record.items() if key != "selection_sha256"}
    )
    if record.get("selection_sha256") != expected:
        raise SelectionError("selection record digest mismatch")
    return record


def _resolved_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    immutable = manifest.get("immutable", {})
    for key in ("resolved_config", "config"):
        value = immutable.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def inherited_hydra_overrides(
    selection: Mapping[str, Any],
    *,
    destination_arch: Optional[str] = None,
    destination_experiment: Optional[str] = None,
) -> list[str]:
    """Translate a frozen C/D choice into explicit, auditable Hydra overrides."""

    selected_manifest = ManifestStore(selection["selected_manifest"]).read()
    config = _resolved_config(selected_manifest)
    output = []
    model = config.get("model", {})
    if isinstance(model, Mapping):
        for field in MODEL_INHERITABLE_FIELDS:
            if field == "student_arch" and destination_arch is not None:
                continue
            if field in model and model[field] is not None:
                output.append(f"model.{field}={_hydra_value(model[field])}")
    criterion = config.get("criterion", {})
    if isinstance(criterion, Mapping):
        for field in CRITERION_INHERITABLE_FIELDS:
            if field in criterion and criterion[field] is not None:
                output.append(f"criterion.{field}={_hydra_value(criterion[field])}")
    for section_name, fields in SECTION_INHERITABLE_FIELDS.items():
        if (
            destination_experiment == "s2_optional_two_stage"
            and section_name in {"optimization", "lr_scheduler"}
        ):
            continue
        section = config.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for field in fields:
            if field in section and section[field] is not None:
                output.append(
                    f"{section_name}.{field}={_hydra_value(section[field])}"
                )

    # D2 must remain clean like C, while S2 preserves whichever noise setting
    # was frozen as the final model. E2 deliberately owns the only permitted
    # change to these fields.
    if destination_experiment != "e2_selected_noisy":
        task = config.get("task", {})
        if isinstance(task, Mapping):
            for field in TASK_NOISE_FIELDS:
                if field in task:
                    output.append(
                        f"task.{field}={_hydra_value(task[field])}"
                    )
        for field in MODEL_NOISE_FIELDS:
            if field in model:
                output.append(f"model.{field}={_hydra_value(model[field])}")
    if destination_arch is not None:
        output.append(f"model.student_arch={destination_arch}")
    conformer_match = selection.get("conformer_match")
    if destination_experiment == "d2_selected_conformer":
        if not isinstance(conformer_match, Mapping):
            raise SelectionError(
                "D2 materialization requires frozen Conformer matching evidence"
            )
    output.extend(str(item) for item in selection.get("materialized_overrides", ()))
    scoped = selection.get("overrides_by_experiment", {})
    if destination_experiment is not None and isinstance(scoped, Mapping):
        output.extend(str(item) for item in scoped.get(destination_experiment, ()))
    if destination_experiment == "d2_selected_conformer":
        # Matching evidence is authoritative and cannot be superseded by a
        # hand-written selection override.
        output.append(str(conformer_match["hydra_override"]))
    deduplicated_reversed = []
    seen = set()
    for item in reversed(output):
        key = item.split("=", 1)[0]
        if key not in seen:
            deduplicated_reversed.append(item)
            seen.add(key)
    return list(reversed(deduplicated_reversed))


def require_final_selection(
    path: os.PathLike[str] | str,
    *,
    current_manifest: os.PathLike[str] | str,
) -> Dict[str, Any]:
    """Enforce that test evaluation targets exactly the frozen final run."""

    record = read_selection(path)
    if record["kind"] != "final":
        raise SelectionError(f"test evaluation requires a final lock, got {record['kind']}")
    current_path = _manifest_path(current_manifest)
    selected_path = _manifest_path(record["selected_manifest"])
    if current_path != selected_path:
        raise SelectionError(
            f"test run {current_path} is not frozen final selection {selected_path}"
        )
    return record


def require_final_selection_artifact(
    path: os.PathLike[str] | str,
    *,
    finetune_manifest: os.PathLike[str] | str,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    """Authorize test evaluation for one exact fine-tuned artifact."""

    record = read_selection(path)
    if record["kind"] != "final":
        raise SelectionError(
            f"test evaluation requires a final lock, got {record['kind']}"
        )
    identity = record.get("selected_finetune_artifact")
    if not isinstance(identity, Mapping):
        raise SelectionError(
            "final selection does not identify a fine-tuning manifest and "
            "checkpoint hash"
        )
    requested_manifest = _manifest_path(finetune_manifest)
    selected_manifest = _manifest_path(identity.get("manifest", ""))
    if requested_manifest != selected_manifest:
        raise SelectionError(
            f"fine-tuning manifest {requested_manifest} is not selected "
            f"{selected_manifest}"
        )
    selected_hash = str(identity.get("checkpoint_sha256", ""))
    if not selected_hash or selected_hash != checkpoint_sha256:
        raise SelectionError(
            "fine-tuning checkpoint hash does not match the final selection"
        )
    return record


def freeze_selected_manifest(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Transition the selected manifest after a final decision is locked."""

    record = read_selection(path)
    if record["kind"] != "final":
        raise SelectionError("only final selection freezes the run lifecycle")
    store = ManifestStore(record["selected_manifest"])
    current = store.read()
    if manifest_run_kind(current) == RUN_KIND_EVALUATION:
        return current
    return store.transition(
        "selection_frozen",
        values={
            "artifacts": {"final_selection": str(Path(path).resolve())},
            "failure": None,
        },
    )


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)
