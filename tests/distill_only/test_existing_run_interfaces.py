"""Non-destructive metadata checks for local C2/C3a exports."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from avhubert.hubert_asr import AVHubertSeq2SeqConfig


ROOT = Path(__file__).resolve().parents[2]


class ExistingRunInterfaceTest(unittest.TestCase):
    def test_local_exports_report_supported_dimensions_when_available(self) -> None:
        names = ("c2_t6_historical_heads", "c3a_t12_historical_heads")
        metadata = [
            ROOT
            / "exp"
            / "chapter3_distill_only"
            / name
            / "seed_1337"
            / "export"
            / "student.pt.metadata.json"
            for name in names
        ]
        if not all(path.is_file() for path in metadata):
            self.skipTest("local C2/C3a export metadata is unavailable")
        decoder_dim = AVHubertSeq2SeqConfig().decoder_embed_dim
        discovered = []
        for path in metadata:
            verification = json.loads(path.read_text(encoding="utf-8"))[
                "verification"
            ]
            encoder_dim = int(verification["encoder_embed_dim"])
            discovered.append(encoder_dim)
            self.assertGreater(encoder_dim, 0)
            self.assertEqual(decoder_dim, 768)
            projection_needed = encoder_dim != decoder_dim
            self.assertTrue(projection_needed)
            self.assertEqual(
                encoder_dim * decoder_dim + decoder_dim,
                295680,
            )
        self.assertEqual(discovered, [384, 384])


if __name__ == "__main__":
    unittest.main()
