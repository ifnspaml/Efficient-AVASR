"""Independent evaluation-child manifests for Chapter 3 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from .evaluation import EvaluationCondition, EvaluationProtocol
from .lineage import (
    ArtifactRef,
    LineageError,
    evaluation_directory,
    sanitize_label,
)
from .manifest import (
    RUN_KIND_EVALUATION,
    ManifestError,
    ManifestStore,
    sha256_file,
    sha256_json,
    utc_now,
)


EVALUATION_PHASES = ("auto", "screening", "final")
EVALUATION_SUBSETS = ("valid", "test")


def parse_eval_subsets(value: str) -> tuple[str, ...]:
    components = tuple(item.strip() for item in value.split(","))
    if (
        not components
        or any(not item for item in components)
        or len(set(components)) != len(components)
        or any(item not in EVALUATION_SUBSETS for item in components)
    ):
        raise LineageError("--eval-subsets must be valid, test, or valid,test")
    return components


def resolve_evaluation_phase(
    requested: str, subsets: Sequence[str]
) -> str:
    if requested not in EVALUATION_PHASES:
        raise LineageError(
            f"--evaluation-phase must be one of {EVALUATION_PHASES}"
        )
    if requested == "auto":
        phase = "final" if "test" in subsets else "screening"
    else:
        phase = requested
    if phase == "screening" and "test" in subsets:
        raise LineageError("screening evaluation does not support the test subset")
    return phase


def selected_conditions(
    protocol: EvaluationProtocol,
    *,
    phase: str,
    subsets: Sequence[str],
) -> tuple[EvaluationCondition, ...]:
    requested = set(subsets)
    conditions = tuple(
        condition
        for condition in protocol.conditions(phase)
        if condition.subset in requested
    )
    missing = requested.difference(condition.subset for condition in conditions)
    if missing:
        raise LineageError(
            f"evaluation protocol has no {phase} conditions for: "
            + ", ".join(sorted(missing))
        )
    return conditions


def evaluation_digest_payload(
    *,
    parent: ArtifactRef,
    protocol: EvaluationProtocol,
    phase: str,
    subsets: Sequence[str],
    conditions: Sequence[EvaluationCondition],
    evaluation_seed: int,
    source_snapshot: Mapping[str, Any],
    command_overrides: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "protocol_sha256": protocol.sha256,
        "phase": phase,
        "subsets": list(subsets),
        "evaluation_seed": evaluation_seed,
        "conditions": [condition.as_manifest_dict() for condition in conditions],
        "command_overrides": list(command_overrides),
        "source": {
            "commit": source_snapshot["commit"],
            "tree": source_snapshot["tree"],
            "integrity_digest": source_snapshot["integrity_digest"],
        },
    }


def evaluation_immutable(
    *,
    experiment: str,
    seed: int,
    label: str,
    parent: ArtifactRef,
    protocol: EvaluationProtocol,
    phase: str,
    subsets: Sequence[str],
    conditions: Sequence[EvaluationCondition],
    evaluation_seed: int,
    source_snapshot: Mapping[str, Any],
    command_overrides: Sequence[str] = (),
) -> Dict[str, Any]:
    payload = evaluation_digest_payload(
        parent=parent,
        protocol=protocol,
        phase=phase,
        subsets=subsets,
        conditions=conditions,
        evaluation_seed=evaluation_seed,
        source_snapshot=source_snapshot,
        command_overrides=command_overrides,
    )
    return {
        "run_kind": RUN_KIND_EVALUATION,
        "experiment": experiment,
        "seed": seed,
        "parent_artifact": parent.parent_record(),
        "source_snapshot": dict(source_snapshot),
        "evaluation": {
            "label": label,
            "sanitized_label": sanitize_label(label),
            "phase": phase,
            "subsets": list(subsets),
            "protocol_path": str(protocol.path.resolve()),
            "protocol_sha256": protocol.sha256,
            "evaluation_seed": evaluation_seed,
            "conditions": [
                condition.as_manifest_dict() for condition in conditions
            ],
            "command_overrides": list(command_overrides),
            "configuration_digest": sha256_json(payload),
        },
    }


def prepare_evaluation_store(
    finetune_run: Path,
    *,
    label: str,
    immutable: Mapping[str, Any],
    allow_existing_source: bool,
) -> tuple[ManifestStore, bool]:
    path = evaluation_directory(finetune_run, label)
    store = ManifestStore(path)
    if not store.exists():
        store.create(immutable)
        return store, True
    current = store.read()
    existing = current["immutable"].get("evaluation")
    requested = immutable.get("evaluation")
    if not isinstance(existing, Mapping) or not isinstance(requested, Mapping):
        raise ManifestError(f"invalid evaluation manifest: {store.path}")
    existing_digest = existing.get("configuration_digest")
    requested_digest = requested.get("configuration_digest")
    if existing_digest != requested_digest and not (
        allow_existing_source
        and _logical_evaluation(existing) == _logical_evaluation(requested)
        and current["state"] != "evaluation_complete"
    ):
        raise ManifestError(
            f'Evaluation label "{label}" already exists with a different '
            "configuration digest.\n\n"
            f"Existing: {existing_digest}\nRequested: {requested_digest}\n\n"
            "Choose another --evaluation-label."
        )
    return store, False


def _logical_evaluation(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "label",
            "phase",
            "subsets",
            "protocol_sha256",
            "evaluation_seed",
            "conditions",
            "command_overrides",
        )
    }


def condition_key(condition: EvaluationCondition) -> str:
    return f"{condition.subset}:{condition.name}"


def execute_evaluation_conditions(
    *,
    store: ManifestStore,
    commands: Sequence[
        tuple[EvaluationCondition, Sequence[str], Path]
    ],
    run_condition: Callable[
        [EvaluationCondition, Sequence[str], Path], None
    ],
    measure_condition: Callable[
        [EvaluationCondition, Path], Mapping[str, Any]
    ],
) -> None:
    """Run/resume independent conditions owned by one evaluation manifest."""

    current = store.read()
    if current["state"] == "evaluation_complete":
        return
    if current["state"] == "prepared":
        store.transition("evaluation_running")
    for condition, command, output in commands:
        key = condition_key(condition)
        progress = (
            store.read().get("runtime", {}).get("conditions", {}).get(key)
        )
        if isinstance(progress, Mapping) and progress.get("status") == "complete":
            artifact = Path(str(progress.get("artifact", "")))
            if (
                artifact.is_file()
                and progress.get("artifact_sha256") == sha256_file(artifact)
            ):
                continue
            raise ManifestError(
                f"completed evaluation artifact changed for {key}: {artifact}"
            )
        store.update(
            {
                "runtime": {
                    "conditions": {
                        key: {
                            "status": "running",
                            "started_at": utc_now(),
                            "command": list(map(str, command)),
                            "output": str(output),
                        }
                    }
                }
            }
        )
        run_condition(condition, command, output)
        measurement = dict(measure_condition(condition, output))
        store.update(
            {
                "runtime": {
                    "conditions": {
                        key: {
                            "status": "complete",
                            "completed_at": utc_now(),
                            "artifact": measurement["artifact"],
                            "artifact_sha256": measurement["artifact_sha256"],
                        }
                    }
                },
                "measurements": {
                    "wer": {
                        condition.subset: {
                            condition.name: measurement
                        }
                    }
                },
                "failure": None,
            }
        )
    store.transition("evaluation_complete")
