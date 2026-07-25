"""Self-contained historical-style Conformer encoder.

The unified repository's vendored Fairseq predates its Conformer module.  This
port implements the architecture used by the old distillation-only repository:
Macaron feed-forward modules, standard self-attention, a biasless GLU
convolution module with BatchNorm and SiLU/Swish, and final layer normalization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from fairseq.modules import MultiheadAttention


class ConformerFeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = self.layer_norm(value)
        value = self.linear1(value)
        value = self.activation(value)
        value = self.dropout1(value)
        value = self.linear2(value)
        return self.dropout2(value)


class ConformerConvolutionModule(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "Conformer depthwise convolution kernel must be positive and odd"
            )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(
            embed_dim, 2 * embed_dim, kernel_size=1, bias=False
        )
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = self.layer_norm(value)
        value = value.transpose(1, 2)
        value = F.glu(self.pointwise_conv1(value), dim=1)
        value = self.depthwise_conv(value)
        value = self.batch_norm(value)
        value = self.activation(value)
        value = self.pointwise_conv2(value)
        value = self.dropout(value)
        return value.transpose(1, 2)


class HistoricalConformerLayer(nn.Module):
    """One pre-norm Macaron Conformer block operating on ``B x T x D``."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        attention_heads: int,
        kernel_size: int,
        dropout: float,
        attention_dropout: float,
        attention_type: str = "original",
    ) -> None:
        super().__init__()
        if embed_dim % attention_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by "
                f"attention_heads={attention_heads}"
            )
        if attention_type not in {"", "original", "standard"}:
            raise ValueError(
                "The historical Conformer port supports only standard "
                f"self-attention, not {attention_type!r}"
            )

        self.ffn1 = ConformerFeedForward(embed_dim, ffn_dim, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        # Use the same Fairseq MHA implementation as the archived
        # ConformerWav2Vec2EncoderLayer. It operates on T x B x D tensors.
        self.self_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            dropout=attention_dropout,
            bias=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvolutionModule(
            embed_dim, kernel_size, dropout
        )
        self.ffn2 = ConformerFeedForward(embed_dim, ffn_dim, dropout)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, value: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        value = value + 0.5 * self.ffn1(value)

        residual = value
        normalized = self.self_attn_layer_norm(value)
        normalized = normalized.transpose(0, 1)
        attention, weights = self.self_attn(
            query=normalized,
            key=normalized,
            value=normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        attention = attention.transpose(0, 1)
        value = residual + self.self_attn_dropout(attention)
        value = value + self.conv_module(value)
        value = value + 0.5 * self.ffn2(value)
        return self.final_layer_norm(value), weights


class HistoricalConformerEncoder(nn.Module):
    """AV-HuBERT-compatible stack of historical Conformer blocks."""

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        ffn_dim: int,
        attention_heads: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layerdrop: float = 0.0,
        attention_type: str = "original",
        position_type: str = "abs",
        max_positions: int = 100000,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("Conformer depth must be positive")
        if embed_dim <= 0 or ffn_dim <= 0 or attention_heads <= 0:
            raise ValueError("Conformer dimensions and head count must be positive")
        if embed_dim % attention_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by "
                f"attention_heads={attention_heads}"
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "Conformer depthwise convolution kernel must be positive and odd"
            )
        if position_type != "abs":
            raise ValueError(
                "The audited historical configuration uses absolute-position "
                f"mode; unsupported position type: {position_type!r}"
            )
        if not 0.0 <= layerdrop < 1.0:
            raise ValueError("layerdrop must be in [0, 1)")

        self.embedding_dim = int(embed_dim)
        self.depth = int(depth)
        self.layerdrop = float(layerdrop)
        self.position_type = position_type
        self._max_positions = int(max_positions)
        self.layers = nn.ModuleList(
            HistoricalConformerLayer(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                attention_heads=attention_heads,
                kernel_size=kernel_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                attention_type=attention_type,
            )
            for _ in range(depth)
        )
        self.input_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_bert_parameters)

    @staticmethod
    def _init_bert_parameters(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _prepare_input(
        self, value: Tensor, padding_mask: Optional[Tensor]
    ) -> Tensor:
        if value.ndim != 3:
            raise ValueError(
                f"Conformer input must be B x T x D, got {tuple(value.shape)}"
            )
        if value.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Conformer expected dimension {self.embedding_dim}, "
                f"got {value.size(-1)}"
            )
        if padding_mask is not None:
            if padding_mask.shape != value.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape B x T; received "
                    f"{tuple(padding_mask.shape)} for input {tuple(value.shape)}"
                )
            value = value.masked_fill(padding_mask.unsqueeze(-1), 0)
        return self.dropout(self.input_layer_norm(value))

    def forward(
        self,
        value: Tensor,
        padding_mask: Optional[Tensor] = None,
        layer: Optional[int] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Optional[Tensor]]]]:
        value = self._prepare_input(value, padding_mask)
        layer_results: List[Tuple[Tensor, Optional[Tensor]]] = []
        for index, conformer_layer in enumerate(self.layers):
            if not self.training or torch.rand(()) > self.layerdrop:
                value, attention = conformer_layer(value, padding_mask)
            else:
                attention = None
            if layer is not None:
                layer_results.append((value, attention))
            if layer is not None and index == layer:
                break
        return value, layer_results

    def get_intermediate_outputs(
        self, value: Tensor, padding_mask: Optional[Tensor] = None
    ) -> List[Tensor]:
        outputs = [value]
        value = self._prepare_input(value, padding_mask)
        for conformer_layer in self.layers:
            value, _ = conformer_layer(value, padding_mask)
            outputs.append(value)
        return outputs

    def max_positions(self) -> int:
        return self._max_positions

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
