"""Pure helpers for the opt-in Fairseq trainer checkpoint smoke.

The heavyweight smoke lives in ``smoke_trainer_checkpoint_resume.py`` and
loads the real AV-HuBERT checkpoint.  This module intentionally has no
repository-model imports so its state-comparison logic remains cheap to unit
test.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class ResumeStateError(AssertionError):
    """Raised when checkpoint state does not satisfy the resume contract."""


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    """Feed a deterministic, type-sensitive representation into ``digest``."""

    if torch.is_tensor(value):
        tensor = value.detach()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b":")
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(b":")
        # View as bytes before crossing devices. This also supports dtypes that
        # NumPy cannot represent directly (for example bfloat16).
        raw = (
            tensor.contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .cpu()
            .numpy()
        )
        digest.update(memoryview(raw))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, tuple):
        digest.update(b"tuple[")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        digest.update(b"sequence[")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
        return
    if isinstance(value, int):
        digest.update(f"int:{value}".encode("utf-8"))
        return
    if isinstance(value, float):
        # float.hex preserves signed zero, infinities, and all finite values
        # exactly. Canonicalize NaNs because their payload is not observable
        # through Python's float API.
        encoded = "nan" if math.isnan(value) else value.hex()
        digest.update(f"float:{encoded}".encode("utf-8"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("utf-8"))
        digest.update(encoded)
        return
    raise TypeError(
        "Unsupported checkpoint-state value for deterministic digest: "
        f"{type(value).__name__}"
    )


def state_digest(value: Any) -> str:
    """Return a deterministic SHA256 digest for nested Fairseq state."""

    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def iterator_position(state: Mapping[str, Any]) -> tuple[int, int]:
    """Return and validate Fairseq's one-based epoch and batch offset."""

    missing = {"epoch", "iterations_in_epoch"}.difference(state)
    if missing:
        raise ResumeStateError(
            f"iterator state is missing required fields: {sorted(missing)}"
        )
    epoch = int(state["epoch"])
    iterations = int(state["iterations_in_epoch"])
    if epoch < 1:
        raise ResumeStateError(f"iterator epoch must be >= 1, received {epoch}")
    if iterations < 0:
        raise ResumeStateError(
            f"iterator offset must be >= 0, received {iterations}"
        )
    return epoch, iterations


def require_exact(label: str, expected: Any, actual: Any) -> None:
    """Require bitwise-equivalent nested state and report useful digests."""

    expected_digest = state_digest(expected)
    actual_digest = state_digest(actual)
    if expected_digest != actual_digest:
        raise ResumeStateError(
            f"{label} was not restored exactly: "
            f"expected_sha256={expected_digest}, actual_sha256={actual_digest}"
        )


def require_within_stage_resume(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    """Validate the state components Fairseq promises to resume in-stage."""

    exact_fields = (
        "model_state",
        "optimizer_state",
        "lr_scheduler_state",
        "num_updates",
        "learning_rate",
        "iterator_state",
        "meter_state",
    )
    for field in exact_fields:
        if field not in expected or field not in actual:
            raise ResumeStateError(f"resume snapshot is missing {field!r}")
        require_exact(field, expected[field], actual[field])


def require_s2_warm_start(
    *,
    stage1_model_state: Any,
    stage1_iterator_state: Mapping[str, Any],
    fresh_stage2: Mapping[str, Any],
    loaded_stage2: Mapping[str, Any],
) -> None:
    """Validate S2: weights transfer, all training state starts fresh."""

    require_exact(
        "S2 model weights",
        stage1_model_state,
        loaded_stage2["model_state"],
    )
    for field in (
        "optimizer_state",
        "lr_scheduler_state",
        "learning_rate",
        "meter_state",
    ):
        require_exact(
            f"S2 fresh {field}",
            fresh_stage2[field],
            loaded_stage2[field],
        )
    if int(loaded_stage2["num_updates"]) != 0:
        raise ResumeStateError(
            "S2 stage 2 must start at update zero, received "
            f"{loaded_stage2['num_updates']}"
        )
    stage1_position = iterator_position(stage1_iterator_state)
    stage2_position = iterator_position(loaded_stage2["iterator_state"])
    if stage2_position != (1, 0):
        raise ResumeStateError(
            "S2 stage-2 iterator must restart at epoch 1, offset 0; "
            f"received {stage2_position}"
        )
    if stage1_position == stage2_position:
        raise ResumeStateError(
            "S2 smoke did not advance stage 1, so dataloader reset was not tested"
        )
