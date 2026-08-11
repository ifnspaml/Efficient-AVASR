"""Historical final-representation prediction heads for joint distillation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class SplitLinear(nn.Module):
    """Apply an independent linear projection to each contiguous input split."""

    def __init__(self, input_dim: int, num_splits: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or num_splits <= 0:
            raise ValueError("SplitLinear dimensions and num_splits must be positive")
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
        split = value.reshape(batch, frames, self.num_splits, self.input_dim)
        output = torch.einsum("btni,nio->btno", split, self.weight) + self.bias
        return output.reshape(batch, frames, self.num_splits * self.output_dim)


class HistoricalPredictionHeads(nn.Module):
    """Predict every selected teacher target from the final student output.

    This is intentionally equivalent to the completed Stage-L
    ``HistoricalPredictionHeads`` implementation.  It is duplicated in the
    legacy joint-DP user directory so Stage K can use it without importing or
    changing the completed distillation-only runtime.
    """

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
        output = self.projections(self.activation(self.expand(final_student)))
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
