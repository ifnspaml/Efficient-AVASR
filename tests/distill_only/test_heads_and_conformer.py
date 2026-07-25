"""Unit tests for architecture-independent Chapter 3 components."""

from __future__ import annotations

import unittest

import torch

from chapter3_distill_only.conformer import HistoricalConformerEncoder
from chapter3_distill_only.heads import (
    HistoricalPredictionHeads,
    LayerToLayerAdapters,
)


class HistoricalPredictionHeadsTest(unittest.TestCase):
    def test_four_target_shape_and_independent_gradients(self) -> None:
        targets = (0, 4, 8, 12)
        heads = HistoricalPredictionHeads(
            student_dim=16,
            teacher_dim=24,
            target_layers=targets,
            hidden_dim=12,
        )
        value = torch.randn(2, 7, 16, requires_grad=True)
        output = heads(value)
        self.assertEqual(output.shape, (2, 4, 7, 24))

        # A loss on one target must not update another target's split
        # projection. This is the key historical SplitLinear invariant.
        output[:, 0].sum().backward()
        gradient = heads.projections.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient[0].abs().sum()), 0.0)
        self.assertEqual(float(gradient[1:].abs().sum()), 0.0)

    def test_rejects_duplicate_targets_and_bad_input_shape(self) -> None:
        with self.assertRaises(ValueError):
            HistoricalPredictionHeads(8, 8, (0, 4, 4))
        heads = HistoricalPredictionHeads(8, 8, (0, 4))
        with self.assertRaises(ValueError):
            heads(torch.randn(2, 8))


class LayerToLayerAdaptersTest(unittest.TestCase):
    def test_mapping_shape(self) -> None:
        adapters = LayerToLayerAdapters(
            student_dim=8,
            teacher_dim=12,
            teacher_target_layers=(0, 4, 8, 12),
            student_match_layers=(0, 4, 8, 12),
        )
        values = [torch.randn(2, 5, 8) for _ in range(4)]
        self.assertEqual(adapters(values).shape, (2, 4, 5, 12))

    def test_invalid_mapping_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LayerToLayerAdapters(8, 8, (0, 4), (0,))
        with self.assertRaises(ValueError):
            LayerToLayerAdapters(8, 8, (0, 4), (0, 0))


class HistoricalConformerEncoderTest(unittest.TestCase):
    def test_intermediate_contract_and_backward(self) -> None:
        encoder = HistoricalConformerEncoder(
            depth=3,
            embed_dim=24,
            ffn_dim=40,
            attention_heads=4,
            kernel_size=7,
            dropout=0.0,
            attention_dropout=0.0,
        )
        value = torch.randn(2, 9, 24, requires_grad=True)
        padding_mask = torch.zeros(2, 9, dtype=torch.bool)
        padding_mask[1, -2:] = True
        outputs = encoder.get_intermediate_outputs(value, padding_mask)
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(item.shape == (2, 9, 24) for item in outputs))
        outputs[-1].square().mean().backward()
        self.assertIsNotNone(value.grad)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            HistoricalConformerEncoder(2, 25, 32, 4)
        with self.assertRaises(ValueError):
            HistoricalConformerEncoder(2, 24, 32, 4, kernel_size=8)
        with self.assertRaises(ValueError):
            HistoricalConformerEncoder(
                2, 24, 32, 4, position_type="rel_pos"
            )


if __name__ == "__main__":
    unittest.main()
