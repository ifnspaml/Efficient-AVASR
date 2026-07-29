"""CPU-only tests for the trainable seq2seq encoder interface."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "fairseq"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from fairseq.models import FairseqDecoder  # noqa: E402

from avhubert.hubert_asr import (  # noqa: E402
    AVHubertSeq2Seq,
    HubertEncoderWrapper,
)


@dataclass
class FakeConfig:
    decoder_embed_dim: int
    freeze_finetune_updates: int = 48000


class FakeEncoderCore(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.layerdrop = 0.1

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = self.linear(source)
        value = self.batch_norm(value.transpose(1, 2)).transpose(1, 2)
        return self.dropout(value)


class FakeW2VModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = FakeEncoderCore(input_dim, output_dim)

    def extract_finetune(self, source, padding_mask):
        return self.encoder(source), padding_mask


class FakeDecoder(FairseqDecoder):
    def __init__(self, input_dim: int) -> None:
        super().__init__(dictionary=None)
        self.projection = nn.Linear(input_dim, 3)

    def forward(self, prev_output_tokens, encoder_out, **kwargs):
        del prev_output_tokens, kwargs
        return self.projection(encoder_out["encoder_out"])


def build_model(
    encoder_dim: int,
    decoder_dim: int,
) -> tuple[AVHubertSeq2Seq, FakeW2VModel, FakeDecoder]:
    torch.manual_seed(4)
    backbone = FakeW2VModel(input_dim=7, output_dim=encoder_dim)
    cfg = FakeConfig(decoder_embed_dim=decoder_dim)
    encoder = HubertEncoderWrapper(backbone, cfg)
    decoder = FakeDecoder(decoder_dim)
    model = AVHubertSeq2Seq(encoder, decoder, None, cfg)
    model.train()
    return model, backbone, decoder


def sample() -> dict[str, torch.Tensor]:
    torch.manual_seed(9)
    return {
        "source": torch.randn(3, 5, 7) + 1.5,
        "padding_mask": torch.zeros(3, 5, dtype=torch.bool),
        "prev_output_tokens": torch.ones(3, 2, dtype=torch.long),
    }


class ProjectionDimensionTest(unittest.TestCase):
    def test_supported_dimension_pairs(self) -> None:
        for encoder_dim, decoder_dim in (
            (768, 768),
            (384, 768),
            (512, 768),
            (768, 512),
        ):
            with self.subTest(
                encoder_dim=encoder_dim, decoder_dim=decoder_dim
            ):
                model, _, _ = build_model(encoder_dim, decoder_dim)
                output = model.encoder(**sample())
                self.assertEqual(output["encoder_out"].shape[-1], decoder_dim)
                if encoder_dim == decoder_dim:
                    self.assertIsNone(model.encoder.proj)
                    self.assertFalse(
                        any(
                            key.startswith("encoder.proj.")
                            for key in model.state_dict()
                        )
                    )
                else:
                    projection = model.encoder.proj
                    assert projection is not None
                    self.assertEqual(
                        tuple(projection.weight.shape),
                        (decoder_dim, encoder_dim),
                    )
                    self.assertEqual(
                        tuple(projection.bias.shape), (decoder_dim,)
                    )
                    self.assertEqual(
                        sum(
                            parameter.numel()
                            for parameter in projection.parameters()
                        ),
                        encoder_dim * decoder_dim + decoder_dim,
                    )

    def test_forward_is_explicit_composition_and_projection_runs_once(self) -> None:
        for encoder_dim, decoder_dim in ((8, 8), (6, 8)):
            with self.subTest(encoder_dim=encoder_dim):
                model, _, _ = build_model(encoder_dim, decoder_dim)
                model.eval()
                inputs = sample()
                direct = model.encoder(**inputs)
                composed = model.encoder.apply_output_projection(
                    model.encoder.extract_backbone(**inputs)
                )
                self.assertTrue(
                    torch.equal(direct["encoder_out"], composed["encoder_out"])
                )
                if model.encoder.proj is not None:
                    calls = []
                    handle = model.encoder.proj.register_forward_hook(
                        lambda *unused: calls.append(1)
                    )
                    try:
                        model(**inputs)
                    finally:
                        handle.remove()
                    self.assertEqual(len(calls), 1)

    def test_projection_state_dict_compatibility(self) -> None:
        original, _, _ = build_model(6, 8)
        state = original.state_dict()
        self.assertIn("encoder.proj.weight", state)
        self.assertIn("encoder.proj.bias", state)

        compatible, _, _ = build_model(6, 8)
        compatible.load_state_dict(state, strict=True)
        self.assertTrue(
            torch.equal(
                original.encoder.proj.weight,
                compatible.encoder.proj.weight,
            )
        )

        incompatible, _, _ = build_model(5, 8)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            incompatible.load_state_dict(state, strict=False)

        equal, _, _ = build_model(8, 8)
        self.assertNotIn("encoder.proj.weight", equal.state_dict())
        self.assertNotIn("encoder.proj.bias", equal.state_dict())


class ProjectionGradientTest(unittest.TestCase):
    def _backward_at(self, update: int):
        model, backbone, decoder = build_model(6, 8)
        model.set_num_updates(update)
        output = model(**sample())
        output.sum().backward()
        return model, backbone, decoder

    def test_projection_trains_while_backbone_is_frozen(self) -> None:
        for update in (0, 47_999):
            with self.subTest(update=update):
                model, backbone, decoder = self._backward_at(update)
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in backbone.parameters()
                    )
                )
                assert model.encoder.proj is not None
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in model.encoder.proj.parameters()
                    )
                )
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in decoder.parameters()
                    )
                )
                visible = {id(parameter) for parameter in model.parameters()}
                self.assertTrue(
                    all(
                        id(parameter) in visible
                        for parameter in model.encoder.proj.parameters()
                    )
                )

    def test_boundary_update_unfreezes_backbone(self) -> None:
        model, backbone, decoder = self._backward_at(48_000)
        self.assertTrue(
            all(parameter.grad is not None for parameter in backbone.parameters())
        )
        assert model.encoder.proj is not None
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.encoder.proj.parameters())
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in decoder.parameters())
        )

    def test_batchnorm_dropout_and_layerdrop_behavior_is_preserved(self) -> None:
        model, backbone, _ = build_model(6, 8)
        model.set_num_updates(0)
        self.assertTrue(model.training)
        self.assertTrue(model.encoder.training)
        self.assertTrue(backbone.encoder.training)
        self.assertEqual(backbone.encoder.dropout.p, 0.25)
        self.assertEqual(backbone.encoder.layerdrop, 0.1)
        before = backbone.encoder.batch_norm.running_mean.detach().clone()
        model(**sample()).sum().backward()
        after = backbone.encoder.batch_norm.running_mean.detach().clone()
        self.assertFalse(torch.equal(before, after))
        self.assertIsNone(backbone.encoder.batch_norm.weight.grad)
        self.assertIsNone(backbone.encoder.batch_norm.bias.grad)


if __name__ == "__main__":
    unittest.main()
