"""Distillation heads used by the Chapter 3 distillation-only models.

The prediction head intentionally follows the historical ``expand-last``
implementation in ``distil-av-hubert``: one expansion from the final student
representation, GELU, and independent split projections for every teacher
target.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


class SplitLinear(nn.Module):
    """Apply independent linear projections to contiguous input splits.

    Input has shape ``B x T x (N * input_dim)`` and output has shape
    ``B x T x (N * output_dim)``.  Keeping the split parameters in one tensor
    reproduces the layout of the historical distillation-only implementation
    while still ensuring that each target has independent weights and biases.
    """

    def __init__(self, input_dim: int, num_splits: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("SplitLinear dimensions must be positive")
        if num_splits <= 0:
            raise ValueError("SplitLinear num_splits must be positive")

        self.input_dim = int(input_dim)
        self.num_splits = int(num_splits)
        self.output_dim = int(output_dim)

        self.weight = nn.Parameter(
            torch.empty(self.num_splits, self.input_dim, self.output_dim)
        )
        self.bias = nn.Parameter(
            torch.empty(1, 1, self.num_splits, self.output_dim)
        )
        bound = self.input_dim**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 3:
            raise ValueError(
                "SplitLinear expects B x T x (N * D) input, "
                f"received {tuple(value.shape)}"
            )
        expected = self.num_splits * self.input_dim
        if value.size(-1) != expected:
            raise ValueError(
                f"SplitLinear expected final dimension {expected}, "
                f"received {value.size(-1)}"
            )

        batch, frames, _ = value.shape
        split = value.reshape(
            batch, frames, self.num_splits, self.input_dim
        )
        output = torch.einsum("btni,nio->btno", split, self.weight)
        output = output + self.bias
        return output.reshape(
            batch, frames, self.num_splits * self.output_dim
        )


class HistoricalPredictionHeads(nn.Module):
    """Predict multiple teacher targets from the final student output."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        target_layers: Sequence[int],
        hidden_dim: int = -1,
    ) -> None:
        super().__init__()
        targets = tuple(int(layer) for layer in target_layers)
        if not targets:
            raise ValueError("At least one teacher target is required")
        if len(set(targets)) != len(targets):
            raise ValueError(f"Teacher targets must be unique: {targets}")
        if student_dim <= 0 or teacher_dim <= 0:
            raise ValueError("Student and teacher dimensions must be positive")

        self.target_layers = targets
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.hidden_dim = (
            int(hidden_dim) if int(hidden_dim) > 0 else self.student_dim
        )
        self.expand = nn.Linear(
            self.student_dim, len(self.target_layers) * self.hidden_dim
        )
        self.activation = nn.GELU()
        self.projections = SplitLinear(
            self.hidden_dim, len(self.target_layers), self.teacher_dim
        )

    def forward(self, final_student: Tensor) -> Tensor:
        if final_student.ndim != 3:
            raise ValueError(
                "Historical prediction heads expect B x T x D input, "
                f"received {tuple(final_student.shape)}"
            )
        if final_student.size(-1) != self.student_dim:
            raise ValueError(
                f"Expected student dimension {self.student_dim}, "
                f"received {final_student.size(-1)}"
            )
        output = self.projections(
            self.activation(self.expand(final_student))
        )
        batch, frames, _ = output.shape
        return (
            output.reshape(
                batch,
                frames,
                len(self.target_layers),
                self.teacher_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )


class LayerToLayerAdapters(nn.Module):
    """Independent adapters for explicitly mapped student representations."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        teacher_target_layers: Sequence[int],
        student_match_layers: Sequence[int],
    ) -> None:
        super().__init__()
        teacher_layers = tuple(int(layer) for layer in teacher_target_layers)
        student_layers = tuple(int(layer) for layer in student_match_layers)
        if not teacher_layers:
            raise ValueError("At least one teacher target is required")
        if len(teacher_layers) != len(student_layers):
            raise ValueError(
                "teacher_target_layers and student_match_layers must have "
                "the same length"
            )
        if len(set(teacher_layers)) != len(teacher_layers):
            raise ValueError(
                f"Teacher target layers must be unique: {teacher_layers}"
            )
        if len(set(student_layers)) != len(student_layers):
            raise ValueError(
                f"Student match layers must be unique: {student_layers}"
            )

        self.teacher_target_layers = teacher_layers
        self.student_match_layers = student_layers
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.adapters = nn.ModuleList(
            nn.Linear(self.student_dim, self.teacher_dim)
            for _ in teacher_layers
        )
        if self.student_dim == self.teacher_dim:
            for adapter in self.adapters:
                nn.init.eye_(adapter.weight)
                nn.init.zeros_(adapter.bias)

    def forward(self, student_representations: Iterable[Tensor]) -> Tensor:
        representations = tuple(student_representations)
        if len(representations) != len(self.adapters):
            raise ValueError(
                f"Expected {len(self.adapters)} student representations, "
                f"received {len(representations)}"
            )
        projected = []
        reference_shape = None
        for index, (representation, adapter) in enumerate(
            zip(representations, self.adapters)
        ):
            if representation.ndim != 3:
                raise ValueError(
                    f"Student representation {index} must be B x T x D, "
                    f"received {tuple(representation.shape)}"
                )
            if representation.size(-1) != self.student_dim:
                raise ValueError(
                    f"Student representation {index} has dimension "
                    f"{representation.size(-1)}; expected {self.student_dim}"
                )
            current_shape = representation.shape[:2]
            if reference_shape is None:
                reference_shape = current_shape
            elif current_shape != reference_shape:
                raise ValueError(
                    "All mapped student representations must share batch "
                    f"and time dimensions; got {reference_shape} and "
                    f"{current_shape}"
                )
            projected.append(adapter(representation))
        return torch.stack(projected, dim=1)
