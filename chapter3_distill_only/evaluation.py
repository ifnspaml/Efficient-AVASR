"""Versioned post-fine-tuning evaluation protocol for Chapter 3.

Encoder distillation and ASR fine-tuning keep RMS (or clean) training noise.
This module owns only the reported validation/inference matrix, which uses
ITU-T P.56 mixing through the existing ``avhubert.noise_utils`` path.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

from .manifest import sha256_file, sha256_json


SCHEMA_VERSION = "chapter3-evaluation/v1"
SUPPORTED_SUBSETS = frozenset({"valid", "test"})
SUPPORTED_NOISE_TYPES = frozenset({"babble", "music", "speech"})
FINAL_SNRS_DB = frozenset({-10, -5, 0, 5, 10})
SCREENING_KEYS = ("clean", "babble_0db", "speech_0db")
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "distill_only"
    / "evaluation_protocol_itut.yaml"
)


class EvaluationProtocolError(ValueError):
    """Raised when an evaluation protocol is missing or inconsistent."""


@dataclass(frozen=True)
class EvaluationCondition:
    name: str
    subset: str
    noise_type: Optional[str]
    noise_root: Optional[Path]
    snr_db: Optional[int]
    noise_method: Optional[str]
    evaluation_phase: str

    def output_relative(self, *, protocol_noise_method: str) -> Path:
        """Return the phase/method/type/SNR/subset output path fragment.

        Clean and noisy results share the protocol method directory so ITU-T
        and RMS evaluation trees never collide.
        """

        if self.noise_type is None:
            return (
                Path(self.evaluation_phase)
                / protocol_noise_method
                / "clean"
                / self.subset
            )
        if self.snr_db is None:
            raise EvaluationProtocolError(
                f"noisy condition {self.name!r} lacks snr_db for output layout"
            )
        return (
            Path(self.evaluation_phase)
            / protocol_noise_method
            / self.noise_type
            / str(self.snr_db)
            / self.subset
        )

    def as_manifest_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.noise_root is not None:
            payload["noise_root"] = str(self.noise_root)
        return payload


@dataclass(frozen=True)
class EvaluationProtocol:
    path: Path
    schema_version: str
    noise_method: str
    evaluation_seed: int
    speech_level_dbov: int
    noise_roots: Dict[str, Path]
    screening: tuple[EvaluationCondition, ...]
    final: tuple[EvaluationCondition, ...]
    raw: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def expected_final_condition_count(self) -> int:
        return len(self.final)

    def conditions(self, phase: str) -> tuple[EvaluationCondition, ...]:
        if phase == "screening":
            return self.screening
        if phase == "final":
            return self.final
        raise EvaluationProtocolError(f"unknown evaluation phase: {phase!r}")

    def as_manifest_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "noise_method": self.noise_method,
            "evaluation_seed": self.evaluation_seed,
            "speech_level_dbov": self.speech_level_dbov,
            "noise_roots": {
                key: str(path) for key, path in sorted(self.noise_roots.items())
            },
            "required_noise_manifests": sorted(
                str(path) for path in required_noise_manifests(self).values()
            ),
            "noise_manifest_sha256": {},
            "screening_condition_names": [item.name for item in self.screening],
            "final_condition_names": sorted(
                {item.name for item in self.final if item.subset == "valid"}
            ),
            "final_snrs_db": sorted(FINAL_SNRS_DB),
            "expected_final_condition_count": self.expected_final_condition_count,
            "screening": [item.as_manifest_dict() for item in self.screening],
            "final": [item.as_manifest_dict() for item in self.final],
            "parsed_protocol": dict(self.raw),
        }

    def with_manifest_hashes(self) -> Dict[str, Any]:
        """Serialize the protocol and attach hashes for reachable noise TSVs."""

        payload = self.as_manifest_dict()
        payload["noise_manifest_sha256"] = noise_manifest_hashes(self)
        return payload


def snr_condition_suffix(snr_db: int) -> str:
    """Map an integer SNR to a stable, unambiguous condition-name suffix."""

    if snr_db < 0:
        return f"m{abs(snr_db)}db"
    if snr_db > 0:
        return f"p{snr_db}db"
    return "0db"


def condition_name(noise_type: Optional[str], snr_db: Optional[int]) -> str:
    if noise_type is None:
        return "clean"
    if snr_db is None:
        raise EvaluationProtocolError("noisy conditions require an integer SNR")
    return f"{noise_type}_{snr_condition_suffix(snr_db)}"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationProtocolError(f"{label} must be a mapping")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationProtocolError(f"{label} must be an integer")
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationProtocolError(
            f"cannot read evaluation protocol {path}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise EvaluationProtocolError(
            f"evaluation protocol {path} must contain a mapping"
        )
    return loaded


def _noise_root(roots: Mapping[str, Path], noise_type: str) -> Path:
    try:
        return roots[noise_type]
    except KeyError as exc:
        raise EvaluationProtocolError(
            f"protocol lacks noise root for {noise_type!r}"
        ) from exc


def _screening_conditions(
    *,
    screening: Mapping[str, Any],
    noise_method: str,
    noise_roots: Mapping[str, Path],
) -> tuple[EvaluationCondition, ...]:
    subset = str(screening.get("subset", "valid"))
    if subset not in SUPPORTED_SUBSETS:
        raise EvaluationProtocolError(f"unsupported screening subset: {subset}")
    conditions: list[EvaluationCondition] = []
    if bool(screening.get("include_clean", True)):
        conditions.append(
            EvaluationCondition(
                name="clean",
                subset=subset,
                noise_type=None,
                noise_root=None,
                snr_db=None,
                noise_method=None,
                evaluation_phase="screening",
            )
        )
    raw_conditions = screening.get("conditions", [])
    if not isinstance(raw_conditions, list):
        raise EvaluationProtocolError(
            "screening_validation.conditions must be a list"
        )
    for item in raw_conditions:
        entry = _require_mapping(item, "screening condition")
        noise_type = str(entry.get("noise_type", ""))
        if noise_type not in SUPPORTED_NOISE_TYPES:
            raise EvaluationProtocolError(
                f"unsupported screening noise type: {noise_type!r}"
            )
        snr_db = _require_int(entry.get("snr_db"), "screening snr_db")
        name = str(entry.get("name") or condition_name(noise_type, snr_db))
        expected = condition_name(noise_type, snr_db)
        if name != expected:
            raise EvaluationProtocolError(
                f"screening condition name {name!r} must be {expected!r}"
            )
        conditions.append(
            EvaluationCondition(
                name=name,
                subset=subset,
                noise_type=noise_type,
                noise_root=_noise_root(noise_roots, noise_type),
                snr_db=snr_db,
                noise_method=noise_method,
                evaluation_phase="screening",
            )
        )
    names = [item.name for item in conditions]
    if names != list(SCREENING_KEYS):
        raise EvaluationProtocolError(
            "screening conditions must be exactly "
            f"{list(SCREENING_KEYS)}, found {names}"
        )
    return tuple(conditions)


def _final_conditions(
    *,
    final: Mapping[str, Any],
    noise_method: str,
    noise_roots: Mapping[str, Path],
) -> tuple[EvaluationCondition, ...]:
    subsets = final.get("subsets")
    if not isinstance(subsets, list) or not subsets:
        raise EvaluationProtocolError(
            "final_evaluation.subsets must be a non-empty list"
        )
    if set(subsets) != SUPPORTED_SUBSETS or len(subsets) != len(SUPPORTED_SUBSETS):
        raise EvaluationProtocolError(
            f"final_evaluation.subsets must be exactly {sorted(SUPPORTED_SUBSETS)}"
        )
    noise_types = final.get("noise_types")
    if (
        not isinstance(noise_types, list)
        or set(noise_types) != SUPPORTED_NOISE_TYPES
    ):
        raise EvaluationProtocolError(
            "final_evaluation.noise_types must be exactly "
            f"{sorted(SUPPORTED_NOISE_TYPES)}"
        )
    snrs = final.get("snrs_db")
    if not isinstance(snrs, list):
        raise EvaluationProtocolError("final_evaluation.snrs_db must be a list")
    snr_values = [_require_int(item, "final snr_db") for item in snrs]
    if set(snr_values) != FINAL_SNRS_DB or len(snr_values) != len(FINAL_SNRS_DB):
        raise EvaluationProtocolError(
            f"final_evaluation.snrs_db must be exactly {sorted(FINAL_SNRS_DB)}"
        )
    conditions: list[EvaluationCondition] = []
    for subset in subsets:
        subset_name = str(subset)
        if subset_name not in SUPPORTED_SUBSETS:
            raise EvaluationProtocolError(f"unsupported final subset: {subset_name}")
        if bool(final.get("include_clean", True)):
            conditions.append(
                EvaluationCondition(
                    name="clean",
                    subset=subset_name,
                    noise_type=None,
                    noise_root=None,
                    snr_db=None,
                    noise_method=None,
                    evaluation_phase="final",
                )
            )
        for noise_type in noise_types:
            noise_name = str(noise_type)
            for snr_db in snr_values:
                conditions.append(
                    EvaluationCondition(
                        name=condition_name(noise_name, snr_db),
                        subset=subset_name,
                        noise_type=noise_name,
                        noise_root=_noise_root(noise_roots, noise_name),
                        snr_db=snr_db,
                        noise_method=noise_method,
                        evaluation_phase="final",
                    )
                )
    if len(conditions) != 32:
        raise EvaluationProtocolError(
            f"final evaluation must expand to 32 conditions, found {len(conditions)}"
        )
    for subset in subsets:
        count = sum(1 for item in conditions if item.subset == subset)
        if count != 16:
            raise EvaluationProtocolError(
                f"final subset {subset} must have 16 conditions, found {count}"
            )
    return tuple(conditions)


def _assert_unique_phase_keys(
    conditions: Sequence[EvaluationCondition], phase: str
) -> None:
    seen = set()
    for condition in conditions:
        key = (condition.subset, condition.name)
        if key in seen:
            raise EvaluationProtocolError(
                f"duplicate {phase} condition {condition.name!r} on {condition.subset}"
            )
        seen.add(key)


def load_evaluation_protocol(
    path: os.PathLike[str] | str | Path,
) -> EvaluationProtocol:
    """Load, validate, and expand a versioned evaluation protocol."""

    protocol_path = Path(path).resolve()
    if not protocol_path.is_file():
        raise EvaluationProtocolError(
            f"evaluation protocol does not exist: {protocol_path}"
        )
    raw = _load_yaml(protocol_path)
    schema_version = str(raw.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise EvaluationProtocolError(
            f"unsupported evaluation schema {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    noise_method = str(raw.get("noise_method", "")).lower()
    if noise_method not in {"itut", "p56"}:
        raise EvaluationProtocolError(
            f"evaluation noise_method must be itut or p56, got {noise_method!r}"
        )
    evaluation_seed = _require_int(raw.get("evaluation_seed"), "evaluation_seed")
    speech_level_dbov = _require_int(
        raw.get("speech_level_dbov"), "speech_level_dbov"
    )
    roots_raw = _require_mapping(raw.get("noise_roots"), "noise_roots")
    noise_roots = {
        str(key): Path(str(value)) for key, value in roots_raw.items()
    }
    if set(noise_roots) != SUPPORTED_NOISE_TYPES:
        raise EvaluationProtocolError(
            f"noise_roots must define exactly {sorted(SUPPORTED_NOISE_TYPES)}"
        )
    screening = _screening_conditions(
        screening=_require_mapping(
            raw.get("screening_validation"), "screening_validation"
        ),
        noise_method=noise_method,
        noise_roots=noise_roots,
    )
    final = _final_conditions(
        final=_require_mapping(raw.get("final_evaluation"), "final_evaluation"),
        noise_method=noise_method,
        noise_roots=noise_roots,
    )
    _assert_unique_phase_keys(screening, "screening")
    _assert_unique_phase_keys(final, "final")
    for condition in (*screening, *final):
        if condition.noise_type is None:
            if any(
                value is not None
                for value in (
                    condition.noise_root,
                    condition.snr_db,
                    condition.noise_method,
                )
            ):
                raise EvaluationProtocolError(
                    f"clean condition {condition.name!r} must omit noise parameters"
                )
        else:
            if condition.noise_method != noise_method:
                raise EvaluationProtocolError(
                    f"noisy condition {condition.name!r} must use {noise_method}"
                )
            if condition.snr_db is None or condition.noise_root is None:
                raise EvaluationProtocolError(
                    f"noisy condition {condition.name!r} is incomplete"
                )
    return EvaluationProtocol(
        path=protocol_path,
        schema_version=schema_version,
        noise_method=noise_method,
        evaluation_seed=evaluation_seed,
        speech_level_dbov=speech_level_dbov,
        noise_roots=noise_roots,
        screening=screening,
        final=final,
        raw=raw,
    )


def required_noise_manifests(
    protocol: EvaluationProtocol, *, phases: Optional[Iterable[str]] = None
) -> Dict[str, Path]:
    """Return absolute paths of every noise TSV required by selected phases."""

    selected = set(phases or ("screening", "final"))
    manifests: Dict[str, Path] = {}
    for phase in selected:
        for condition in protocol.conditions(phase):
            if condition.noise_root is None:
                continue
            path = condition.noise_root / f"{condition.subset}.tsv"
            manifests[str(path)] = path
    return manifests


def noise_manifest_hashes(protocol: EvaluationProtocol) -> Dict[str, str]:
    """Hash every validation/test noise manifest referenced by the protocol."""

    output: Dict[str, str] = {}
    for path in required_noise_manifests(protocol).values():
        try:
            exists = path.is_file()
        except OSError:
            continue
        if exists:
            output[str(path)] = sha256_file(path)
    return output


def validate_protocol_artifacts(
    protocol: EvaluationProtocol, *, phases: Sequence[str]
) -> None:
    """Fail before decoding when noise manifests or ITU-T support are missing."""

    from avhubert.noise_utils import require_noise_method_available

    require_noise_method_available(protocol.noise_method)
    missing = []
    for path in required_noise_manifests(protocol, phases=phases).values():
        try:
            exists = path.is_file()
        except OSError as exc:
            missing.append(f"{path} ({exc})")
            continue
        if not exists:
            missing.append(str(path))
    if missing:
        raise EvaluationProtocolError(
            "missing required noise manifests:\n- " + "\n- ".join(missing)
        )


def measurement_metadata(
    protocol: EvaluationProtocol,
    condition: EvaluationCondition,
    *,
    value: float,
    artifact: Path,
    artifact_sha256: str,
) -> Dict[str, Any]:
    """Build the structured WER measurement recorded in the run manifest."""

    noise_manifest = None
    noise_manifest_sha256 = None
    if condition.noise_root is not None:
        noise_manifest_path = condition.noise_root / f"{condition.subset}.tsv"
        noise_manifest = str(noise_manifest_path)
        try:
            if noise_manifest_path.is_file():
                noise_manifest_sha256 = sha256_file(noise_manifest_path)
        except OSError:
            noise_manifest_sha256 = None
    return {
        "noise_method": condition.noise_method,
        "noise_type": condition.noise_type,
        "snr_db": condition.snr_db,
        "subset": condition.subset,
        "evaluation_phase": condition.evaluation_phase,
        "evaluation_seed": protocol.evaluation_seed,
        "speech_level_dbov": protocol.speech_level_dbov,
        "noise_manifest": noise_manifest,
        "noise_manifest_sha256": noise_manifest_sha256,
        "condition_name": condition.name,
        "value": value,
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
    }


def protocol_digest(protocol: EvaluationProtocol) -> str:
    return sha256_json(protocol.as_manifest_dict())
