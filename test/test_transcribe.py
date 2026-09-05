import unittest

from transcribe import (
    AudioChunk,
    TokenTimestamp,
    DEFAULT_ASR_CANDIDATES,
    TEA_ASR_MLX_MODEL,
    build_chunks,
    build_subtitle_cues,
    clean_asr_text,
    configure_tea_mlx_loader,
    format_srt_timestamp,
    normalize_alignment,
    _srt_text,
    strip_srt_punctuation,
    to_traditional,
    validate_cues,
    _asr_kwargs,
)


class CompatibilityTests(unittest.TestCase):
    def test_default_asr_candidate_is_tea(self):
        self.assertEqual(DEFAULT_ASR_CANDIDATES[0], TEA_ASR_MLX_MODEL)

    def test_clean_asr_text_removes_tokenizer_private_use_artifacts(self):
        self.assertEqual(
            clean_asr_text("甚至出現\ue3cd交易幾乎\ue1a8停滯的情況。"),
            "甚至出現交易幾乎停滯的情況。",
        )

    def test_tea_loader_enables_quantization_for_audio_tower(self):
        class StubModel:
            pass

        class StubQwen3Asr:
            Qwen3ASRModel = StubModel

        configure_tea_mlx_loader(StubQwen3Asr)
        self.assertTrue(StubModel.model_quant_predicate(None, None, None))

    def test_asr_kwargs_passes_context_and_hotwords_to_current_mlx_api(self):
        class StubModel:
            def generate(self, audio, *, system_prompt=None, hotwords=None):
                del audio, system_prompt, hotwords

        kwargs = _asr_kwargs(
            StubModel(),
            language="Chinese",
            context="Vocabulary: NVIDIA, Vera Rubin",
            hotwords=["NVIDIA", "Vera Rubin"],
            max_tokens=256,
            verbose=False,
        )
        self.assertEqual(kwargs["system_prompt"], "Vocabulary: NVIDIA, Vera Rubin")
        self.assertEqual(kwargs["hotwords"], ["NVIDIA", "Vera Rubin"])

    def test_asr_kwargs_keeps_context_for_older_api_with_hotwords(self):
        class StubModel:
            def generate(self, audio, *, context=None, hotwords=None):
                del audio, context, hotwords

        kwargs = _asr_kwargs(
            StubModel(),
            language="Chinese",
            context="Vocabulary: NVIDIA",
            hotwords=["NVIDIA"],
            max_tokens=256,
            verbose=False,
        )
        self.assertEqual(kwargs["context"], "Vocabulary: NVIDIA")
        self.assertEqual(kwargs["hotwords"], ["NVIDIA"])


class ChunkingTests(unittest.TestCase):
    def test_build_chunks_prefers_low_energy_boundary_and_covers_audio(self):
        energies = [1.0] * 2400
        energies[950] = 0.0
        chunks = build_chunks(
            duration=600.0,
            energies=energies,
            target_seconds=240.0,
            overlap_seconds=1.5,
            frame_seconds=0.25,
        )

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start, 0.0)
        self.assertEqual(chunks[-1].end, 600.0)
        self.assertTrue(230.0 <= chunks[0].end <= 250.0)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLessEqual(previous.start, current.start)
            self.assertLessEqual(current.start, previous.end)
            self.assertLessEqual(previous.end, current.end)


class FormattingTests(unittest.TestCase):
    def test_format_srt_timestamp_uses_comma_milliseconds(self):
        self.assertEqual(format_srt_timestamp(1.0), "00:00:01,000")
        self.assertEqual(format_srt_timestamp(61.234), "00:01:01,234")
        self.assertEqual(format_srt_timestamp(3661.9999), "01:01:01,999")

    def test_opencc_converts_only_text(self):
        self.assertEqual(
            to_traditional("甚至出现交易几乎停滞的情况。"),
            "甚至出現交易幾乎停滯的情況。",
        )

    def test_alignment_restores_punctuation_without_losing_timestamps(self):
        tokens = normalize_alignment(
            [
                {"text": "這", "start_time": 0.20, "end_time": 0.40},
                {"text": "是", "start_time": 0.40, "end_time": 0.60},
            ],
            raw_text="這是。",
            source_chunk=3,
        )

        self.assertEqual([token.text for token in tokens], ["這", "是", "。"])
        self.assertEqual((tokens[0].start, tokens[0].end), (0.20, 0.40))
        self.assertEqual((tokens[1].start, tokens[1].end), (0.40, 0.60))
        self.assertEqual(tokens[2].fallback, "punctuation_inferred")
        cues = build_subtitle_cues(tokens, duration=2.0)
        self.assertEqual(cues[0].text, "這是。")

    def test_alignment_keeps_zero_duration_character_items(self):
        tokens = normalize_alignment(
            [
                {"text": "交", "start_time": 1.50, "end_time": 1.76},
                {"text": "易", "start_time": 1.76, "end_time": 2.00},
                {"text": "幾", "start_time": 2.08, "end_time": 2.08},
                {"text": "乎", "start_time": 2.24, "end_time": 2.48},
            ],
            raw_text="交易幾乎",
        )

        self.assertEqual([token.text for token in tokens], ["交", "易", "幾", "乎"])
        self.assertGreater(tokens[2].end, tokens[2].start)

    def test_strip_srt_punctuation_keeps_text_and_product_name_intact(self):
        self.assertEqual(
            strip_srt_punctuation("甚至出現交易。Vera Rubin！Qwen3-ASR"),
            "甚至出現交易 Vera Rubin Qwen3-ASR",
        )

    def test_srt_serializer_can_remove_punctuation_without_changing_timing(self):
        text = _srt_text(
            [
                type(
                    "Cue",
                    (),
                    {"start": 1.0, "end": 2.0, "text": "這是 Vera Rubin。"},
                )()
            ],
            no_punctuation=True,
        )

        self.assertIn("00:00:01,000 --> 00:00:02,000", text)
        self.assertIn("這是 Vera Rubin", text)
        self.assertNotIn("。", text)


class SubtitleTests(unittest.TestCase):
    def test_cues_group_tokens_and_keep_product_phrase_together(self):
        tokens = [
            TokenTimestamp("這", 0.20, 0.38),
            TokenTimestamp("是", 0.38, 0.56),
            TokenTimestamp("一", 0.56, 0.74),
            TokenTimestamp("段", 0.74, 0.92),
            TokenTimestamp("適", 0.92, 1.10),
            TokenTimestamp("合", 1.10, 1.28),
            TokenTimestamp("YouTube", 1.35, 2.00),
            TokenTimestamp("的", 2.00, 2.18),
            TokenTimestamp("字", 2.18, 2.36),
            TokenTimestamp("幕", 2.36, 2.54),
            TokenTimestamp("，", 2.54, 2.70),
            TokenTimestamp("Vera Rubin", 2.85, 3.80),
            TokenTimestamp("不", 3.80, 4.00),
            TokenTimestamp("可", 4.00, 4.20),
            TokenTimestamp("拆", 4.20, 4.40),
            TokenTimestamp("開", 4.40, 4.60),
            TokenTimestamp("。", 4.60, 4.80),
        ]

        cues = build_subtitle_cues(tokens, duration=5.0)

        self.assertGreaterEqual(len(cues), 2)
        self.assertTrue(any("Vera Rubin" in cue.text for cue in cues))
        self.assertFalse(any(cue.text.strip() in {"Vera", "Rubin"} for cue in cues))
        self.assertTrue(all(cue.end - cue.start >= 0.5 for cue in cues))
        self.assertTrue(all(cue.end - cue.start <= 6.0 for cue in cues))
        validate_cues(cues, duration=5.0)

    def test_validate_cues_rejects_overlap(self):
        cues = [
            type("Cue", (), {"start": 0.0, "end": 1.0, "text": "a"})(),
            type("Cue", (), {"start": 0.9, "end": 1.5, "text": "b"})(),
        ]
        with self.assertRaises(ValueError):
            validate_cues(cues, duration=2.0)


if __name__ == "__main__":
    unittest.main()
