#!/usr/bin/env python3
"""Local Taiwan-Mandarin video transcription to Traditional Chinese SRT.

The default route probes the TEA-ASR MLX checkpoint first.  Its model-card
compatibility predicate is applied before loading, and the first real ASR call
must still succeed.  When it does not, the pipeline records the exception and
falls back to a maintained Qwen3 MLX checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_HF_HOME = PROJECT_DIR / "models" / "huggingface"
TEA_ASR_MLX_MODEL = "Alkd/TEA-ASR-1.1-MLX-4bit"
TEA_ASR_MLX_COMPATIBILITY_NOTE = (
    "套用 TEA-ASR-1.1 MLX 模型卡要求的 quantization predicate，讓 8-bit audio tower "
    "與 4-bit text decoder 正確載入。"
)
TEA_ASR_MLX_TEXT_CLEANUP_NOTE = (
    "清除 tokenizer 相容性造成的 Unicode Private Use Area 殘留字元，不改動時間戳。"
)
DEFAULT_ASR_CANDIDATES = (
    TEA_ASR_MLX_MODEL,
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
)
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
DEFAULT_CHUNK_SECONDS = 240.0
DEFAULT_OVERLAP_SECONDS = 1.5
MAX_ALIGNMENT_SECONDS = 270.0

# Add project-specific terms here.  The current MLX-Audio API accepts this
# list as hotwords; older compatible APIs can receive the generated context.
HOTWORDS = [
    "NVIDIA",
    "GeForce",
    "RTX",
    "Blackwell",
    "Vera Rubin",
    "CoWoS",
    "TSMC",
    "OpenAI",
    "Gemini",
    "Claude",
    "Qwen",
    "Kimi",
    "Cerebras",
]

try:
    import numpy as np
except ImportError:  # pragma: no cover - gives a clearer runtime error later
    np = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - gives a clearer runtime error later
    sf = None


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class PreparedAudioChunk:
    chunk: AudioChunk
    path: Path


@dataclass
class ASRChunk:
    index: int
    start: float
    end: float
    text: str
    language: str
    audio_path: Path
    segments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TokenTimestamp:
    text: str
    start: float
    end: float
    source_chunk: int | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass
class ModelInfo:
    name: str
    backend: str = "mlx"
    load_seconds: float = 0.0
    active_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    compatibility: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    tokens: list[TokenTimestamp]
    fallback: str | None = None
    error: str | None = None


@dataclass
class TranscriptBundle:
    full_transcript: str
    raw_transcript: str
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    characters: list[dict[str, Any]]
    model: dict[str, Any]
    aligner: dict[str, Any]
    processing_time: float
    audio_duration: float
    realtime_factor: float
    warnings: list[str]
    cues: list[SubtitleCue]


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("缺少 numpy，請在 taiwan-subtitle/.venv 重新安裝 requirements.txt")
    return np


def _require_soundfile() -> Any:
    if sf is None:
        raise RuntimeError("缺少 soundfile，請在 taiwan-subtitle/.venv 重新安裝 requirements.txt")
    return sf


def configure_huggingface_cache() -> Path:
    """Keep model downloads inside this project unless the caller overrides it."""

    cache_home = Path(os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))).expanduser()
    os.environ.setdefault("HF_HUB_CACHE", str(cache_home / "hub"))
    cache_home.mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    return cache_home


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(item) for item in command],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到系統工具：{command[0]}，請先用 Homebrew 安裝。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"命令失敗：{command[0]} {detail}") from exc


def probe_media(path: Path) -> MediaInfo:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{path}")
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe 無法讀取媒體長度：{path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"輸入媒體沒有有效音訊長度：{path}")
    return MediaInfo(path=path, duration=duration)


def extract_audio(path: Path, destination: Path) -> Path:
    """Decode the first audio stream without touching the source media."""

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(Path(path).resolve()),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg 沒有產生有效暫存 WAV：{destination}")
    return destination


def scan_energy(wav_path: Path, frame_seconds: float = 0.25) -> list[float]:
    """Scan RMS energy in bounded memory for low-energy chunk boundaries."""

    audio_file = _require_soundfile()
    arrays = _require_numpy()
    if frame_seconds <= 0:
        raise ValueError("frame_seconds 必須大於 0")
    energies: list[float] = []
    with audio_file.SoundFile(str(wav_path), "r") as source:
        sample_rate = int(source.samplerate)
        frame_size = max(1, round(sample_rate * frame_seconds))
        block_size = sample_rate * 30
        while True:
            block = source.read(block_size, dtype="float32", always_2d=False)
            if len(block) == 0:
                break
            block_array = arrays.asarray(block, dtype=arrays.float32)
            for start in range(0, len(block_array), frame_size):
                frame = block_array[start : start + frame_size]
                if len(frame) == 0:
                    continue
                rms = arrays.sqrt(arrays.mean(arrays.square(frame)))
                energies.append(float(rms))
            del block_array
    return energies


def build_chunks(
    duration: float,
    energies: Sequence[float],
    target_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    frame_seconds: float = 0.25,
) -> list[AudioChunk]:
    """Build overlapping windows and prefer the quietest nearby boundary."""

    if duration <= 0:
        return []
    if target_seconds <= 0:
        raise ValueError("target_seconds 必須大於 0")
    if overlap_seconds < 0 or overlap_seconds >= target_seconds:
        raise ValueError("overlap_seconds 必須介於 0 與 target_seconds 之間")
    if frame_seconds <= 0:
        raise ValueError("frame_seconds 必須大於 0")

    target_seconds = min(float(target_seconds), MAX_ALIGNMENT_SECONDS - 10.0)
    half_overlap = float(overlap_seconds) / 2.0
    chunks: list[AudioChunk] = []
    cursor = 0.0

    def find_boundary(desired: float) -> float:
        if not energies:
            return min(duration, desired)
        search_start = max(cursor + min(30.0, target_seconds * 0.25), desired - 20.0)
        search_end = min(duration, desired + 20.0)
        first = max(0, int(math.floor(search_start / frame_seconds)))
        last = min(len(energies) - 1, int(math.ceil(search_end / frame_seconds)))
        if first > last:
            return min(duration, desired)
        best_index = min(range(first, last + 1), key=lambda index: energies[index])
        candidate = best_index * frame_seconds
        if candidate <= cursor + 5.0:
            candidate = desired
        return max(cursor + 5.0, min(duration, candidate))

    while cursor < duration - 1e-6:
        remaining = duration - cursor
        if remaining <= target_seconds + half_overlap:
            chunks.append(AudioChunk(len(chunks), round(cursor, 3), round(duration, 3)))
            break

        desired = cursor + target_seconds
        boundary = find_boundary(desired)
        window_end = min(duration, boundary + half_overlap)
        next_cursor = max(cursor + 5.0, boundary - half_overlap)
        chunks.append(AudioChunk(len(chunks), round(cursor, 3), round(window_end, 3)))
        if next_cursor <= cursor + 1e-6:
            next_cursor = min(duration, cursor + target_seconds)
        cursor = next_cursor

    if chunks:
        chunks[-1] = AudioChunk(chunks[-1].index, chunks[-1].start, round(duration, 3))
    return chunks


def write_audio_chunk(wav_path: Path, chunk: AudioChunk, destination: Path) -> Path:
    """Read only one bounded chunk from the normalized WAV and write a temp WAV."""

    audio_file = _require_soundfile()
    with audio_file.SoundFile(str(wav_path), "r") as source:
        sample_rate = int(source.samplerate)
        source.seek(max(0, int(round(chunk.start * sample_rate))))
        frames = max(1, int(round(chunk.duration * sample_rate)))
        audio = source.read(frames, dtype="float32", always_2d=False)
    if len(audio) == 0:
        raise RuntimeError(f"音訊 chunk 為空：{chunk.index}")
    destination = Path(destination)
    audio_file.write(str(destination), audio, sample_rate, subtype="PCM_16")
    del audio
    return destination


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = int(max(0.0, float(seconds)) * 1000.0 + 1e-7)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@lru_cache(maxsize=1)
def _opencc_converter() -> Any:
    try:
        from opencc import OpenCC
    except ImportError as exc:  # pragma: no cover - dependency smoke covers this
        raise RuntimeError(
            "缺少 opencc-python-reimplemented，請在 taiwan-subtitle/.venv 重新安裝 requirements.txt"
        ) from exc
    return OpenCC("s2twp")


def to_traditional(text: str) -> str:
    return _opencc_converter().convert(text)


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
    )


_UNIT_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._'’+\-/][A-Za-z0-9]+)*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\w\s]",
    re.UNICODE,
)


def _text_units(text: str) -> list[str]:
    return _UNIT_PATTERN.findall(" ".join(str(text).split()))


def _join_units(units: Iterable[str]) -> str:
    output = ""
    opening = "([［「『【《〈（{"
    closing = "，。！？；：、」』】》〉）]}>,.!?;:"
    for raw_unit in units:
        unit = str(raw_unit)
        if not unit:
            continue
        if not output:
            output = unit
            continue
        previous = output[-1]
        if unit[0] in closing or previous in opening:
            output += unit
        elif is_cjk_char(previous) and is_cjk_char(unit[0]):
            output += unit
        elif previous.isascii() and unit[0].isascii() and (
            previous.isalnum() and unit[0].isalnum()
        ):
            output += " " + unit
        elif is_cjk_char(previous) != is_cjk_char(unit[0]):
            output += " " + unit
        else:
            output += unit
    return output.strip()


def _smart_join_strings(left: str, right: str) -> str:
    return _join_units(_text_units(left) + _text_units(right))


def build_context(hotwords: Sequence[str]) -> str:
    cleaned = [str(word).strip() for word in hotwords if str(word).strip()]
    return f"Vocabulary: {', '.join(cleaned)}" if cleaned else ""


def _is_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(
        unicodedata.category(char).startswith("P") for char in stripped
    )


def _infer_missing_punctuation(
    tokens: Sequence[TokenTimestamp], raw_text: str
) -> list[TokenTimestamp]:
    """Restore ASR punctuation that a character-level aligner may omit."""

    if not tokens or not raw_text.strip():
        return list(tokens)

    raw_units = _text_units(raw_text)
    raw_events: list[tuple[int, str]] = []
    raw_nonpunct_count = 0
    for unit in raw_units:
        if _is_punctuation(unit):
            raw_events.append((raw_nonpunct_count, unit))
        else:
            raw_nonpunct_count += 1
    if not raw_events:
        return list(tokens)

    existing_events: list[tuple[int, str]] = []
    existing_nonpunct_count = 0
    token_unit_counts: list[int] = []
    for token in tokens:
        count = 0
        for unit in _text_units(token.text):
            if _is_punctuation(unit):
                existing_events.append((existing_nonpunct_count, unit))
            else:
                existing_nonpunct_count += 1
                count += 1
        token_unit_counts.append(count)

    used_existing: set[int] = set()
    missing: list[tuple[int, str]] = []
    for before_count, punctuation in raw_events:
        match = None
        for index, (existing_before, existing_text) in enumerate(existing_events):
            if index in used_existing:
                continue
            if existing_text == punctuation and abs(existing_before - before_count) <= 1:
                match = index
                break
        if match is None:
            missing.append((before_count, punctuation))
        else:
            used_existing.add(match)
    if not missing:
        return list(tokens)

    inferred: list[TokenTimestamp] = []
    for before_count, punctuation in missing:
        if before_count <= 0:
            anchor = tokens[0]
            end = max(0.001, float(anchor.start))
            start = max(0.0, end - 0.01)
            if end <= start:
                end = start + 0.001
        else:
            cumulative = 0
            anchor_index = len(tokens) - 1
            for index, count in enumerate(token_unit_counts):
                if before_count <= cumulative + count:
                    anchor_index = index
                    break
                cumulative += count
            anchor = tokens[anchor_index]
            start = max(0.0, float(anchor.end))
            next_start = (
                float(tokens[anchor_index + 1].start)
                if anchor_index + 1 < len(tokens)
                else start + 0.001
            )
            end = min(start + 0.02, next_start) if next_start > start else start + 0.001
            if end <= start:
                end = start + 0.001
        inferred.append(
            TokenTimestamp(
                text=punctuation,
                start=round(start, 3),
                end=round(end, 3),
                source_chunk=anchor.source_chunk,
                fallback="punctuation_inferred",
            )
        )

    return sorted([*tokens, *inferred], key=lambda token: (token.start, token.end))


def _is_sentence_boundary(text: str) -> bool:
    return bool(text.strip()) and text.strip()[-1] in "。！？!?；;"


def _display_unit_count(text: str) -> int:
    units = _text_units(text)
    return sum(1 for unit in units if not unit.isspace())


def _render_token_texts(tokens: Sequence[TokenTimestamp]) -> str:
    result = ""
    opening = "([［「『【《〈（{"
    closing = "，。！？；：、」』】》〉）]}>,.!?;:"
    for token in tokens:
        text = " ".join(str(token.text).split())
        if not text:
            continue
        if not result:
            result = text
            continue
        previous = result[-1]
        if text[0] in closing or previous in opening:
            result += text
        elif previous.isascii() and text[0].isascii() and previous.isalnum() and text[0].isalnum():
            result += " " + text
        elif is_cjk_char(previous) and is_cjk_char(text[0]):
            result += text
        elif is_cjk_char(previous) != is_cjk_char(text[0]):
            result += " " + text
        else:
            result += text
    return result.strip()


def _make_cue(tokens: Sequence[TokenTimestamp]) -> SubtitleCue:
    text = _render_token_texts(tokens)
    start = min(token.start for token in tokens)
    end = max(token.end for token in tokens)
    return SubtitleCue(start=max(0.0, start), end=max(start, end), text=text)


def _merge_cues(left: SubtitleCue, right: SubtitleCue) -> SubtitleCue:
    return SubtitleCue(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        text=_smart_join_strings(left.text, right.text),
    )


def build_subtitle_cues(
    tokens: Sequence[TokenTimestamp], duration: float
) -> list[SubtitleCue]:
    """Group aligned tokens into readable, non-overlapping subtitle cues."""

    ordered = sorted(
        (token for token in tokens if str(token.text).strip()),
        key=lambda token: (token.start, token.end),
    )
    if not ordered:
        return []

    raw_cues: list[SubtitleCue] = []
    current: list[TokenTimestamp] = []
    current_count = 0
    for token in ordered:
        token_count = max(1, _display_unit_count(token.text))
        too_long = current and (
            current_count + token_count > 20
            or token.start - current[0].start >= 6.0
        )
        if too_long:
            raw_cues.append(_make_cue(current))
            current = []
            current_count = 0
        current.append(token)
        current_count += token_count
        if (
            (_is_sentence_boundary(token.text) or _is_punctuation(token.text))
            and current_count >= 3
        ):
            raw_cues.append(_make_cue(current))
            current = []
            current_count = 0
    if current:
        raw_cues.append(_make_cue(current))

    # Merge tiny cues before enforcing the minimum display duration.
    cues = list(raw_cues)
    index = 0
    while index < len(cues):
        if _display_unit_count(cues[index].text) <= 2 and len(cues) > 1:
            if index + 1 < len(cues):
                cues[index : index + 2] = [_merge_cues(cues[index], cues[index + 1])]
                continue
            cues[index - 1 : index + 1] = [_merge_cues(cues[index - 1], cues[index])]
            index = max(0, index - 1)
            continue
        index += 1

    normalized: list[SubtitleCue] = []
    for cue in cues:
        start = max(0.0, min(float(duration), cue.start))
        end = max(start, min(float(duration), cue.end))
        if end - start < 0.5 and normalized:
            normalized[-1] = _merge_cues(normalized[-1], SubtitleCue(start, end, cue.text))
            continue
        if end - start < 0.5:
            end = min(float(duration), start + 0.5)
        normalized.append(SubtitleCue(start, end, cue.text))

    # Clip a very small accidental overlap caused by rounded model timestamps.
    clipped: list[SubtitleCue] = []
    for cue in normalized:
        if clipped and cue.start < clipped[-1].end:
            adjusted_start = clipped[-1].end
            if cue.end <= adjusted_start:
                clipped[-1] = _merge_cues(clipped[-1], cue)
                continue
            cue = SubtitleCue(adjusted_start, cue.end, cue.text)
        clipped.append(cue)
    return [
        SubtitleCue(
            start=max(0.0, min(float(duration), cue.start)),
            end=max(0.0, min(float(duration), cue.end)),
            text=cue.text.strip(),
        )
        for cue in clipped
        if cue.text.strip() and cue.end > cue.start
    ]


def validate_cues(cues: Sequence[SubtitleCue], duration: float) -> None:
    previous_end = 0.0
    for index, cue in enumerate(cues, start=1):
        if cue.start < -1e-6 or cue.end > duration + 1e-3:
            raise ValueError(f"SRT cue {index} 超出音訊範圍")
        if cue.end <= cue.start:
            raise ValueError(f"SRT cue {index} 的 end 不大於 start")
        if index > 1 and cue.start < previous_end - 1e-3:
            raise ValueError(f"SRT cue {index} 與前一條字幕重疊")
        previous_end = cue.end


def _item_value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return default
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _split_aligned_text(text: str) -> list[str]:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    if not any(is_cjk_char(char) for char in cleaned):
        return [cleaned]

    parts: list[str] = []
    latin_buffer: list[str] = []

    def flush_latin() -> None:
        if latin_buffer:
            value = "".join(latin_buffer).strip()
            if value:
                parts.append(value)
            latin_buffer.clear()

    for char in cleaned:
        if is_cjk_char(char):
            flush_latin()
            parts.append(char)
        elif char.isspace():
            latin_buffer.append(char)
        elif unicodedata.category(char).startswith("P"):
            flush_latin()
            parts.append(char)
        else:
            latin_buffer.append(char)
    flush_latin()
    return parts


def normalize_alignment(
    items: Sequence[Any],
    offset: float = 0.0,
    raw_text: str = "",
    source_chunk: int | None = None,
) -> list[TokenTimestamp]:
    """Normalize MLX-Audio items and add the absolute chunk offset."""

    normalized: list[TokenTimestamp] = []
    previous_end = float(offset)
    for item in items:
        text = str(_item_value(item, "text", "word", default="")).strip()
        if not text:
            continue
        start_raw = _item_value(item, "start_time", "start", default=None)
        end_raw = _item_value(item, "end_time", "end", default=None)
        try:
            start = float(start_raw) + float(offset)
            end = float(end_raw) + float(offset)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        start = max(start, previous_end)
        if end <= start:
            # Qwen3 Forced Aligner can emit a valid character with identical
            # start/end values.  Keep it instead of silently dropping text.
            end = start + 0.001

        parts = _split_aligned_text(text)
        if not parts:
            continue
        span = end - start
        part_duration = span / len(parts)
        for part_index, part in enumerate(parts):
            part_start = start + part_index * part_duration
            part_end = end if part_index == len(parts) - 1 else start + (part_index + 1) * part_duration
            normalized.append(
                TokenTimestamp(
                    text=part,
                    start=round(part_start, 3),
                    end=round(part_end, 3),
                    source_chunk=source_chunk,
                    fallback="character_split_fallback" if len(parts) > 1 and not is_cjk_char(text[0]) else None,
                )
            )
        previous_end = end
    return _infer_missing_punctuation(normalized, raw_text)


def _mlx_memory() -> tuple[int | None, int | None]:
    try:
        import mlx.core as mx
    except ImportError:
        return None, None
    active = getattr(mx, "get_active_memory", None)
    peak = getattr(mx, "get_peak_memory", None)
    try:
        active_value = int(active()) if callable(active) else None
    except Exception:
        active_value = None
    try:
        peak_value = int(peak()) if callable(peak) else None
    except Exception:
        peak_value = None
    return active_value, peak_value


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None)
    if callable(clear):
        clear()
        return

    metal = getattr(mx, "metal", None)
    clear = getattr(metal, "clear_cache", None) if metal is not None else None
    if callable(clear):
        clear()


def _tea_model_quantization_predicate(_self: Any, _path: Any, _module: Any) -> bool:
    """Quantize every TEA module, including its already-quantized audio tower."""

    return True


def configure_tea_mlx_loader(qwen3_asr_module: Any | None = None) -> None:
    """Apply the compatibility hook documented by the TEA MLX model card."""

    if qwen3_asr_module is None:
        from mlx_audio.stt.models.qwen3_asr import qwen3_asr as qwen3_asr_module

    qwen3_asr_module.Qwen3ASRModel.model_quant_predicate = (
        _tea_model_quantization_predicate
    )


_PRIVATE_USE_AREA_RE = re.compile(r"[\uE000-\uF8FF]+")


def clean_asr_text(text: str) -> str:
    """Remove tokenizer artifacts while preserving normal transcript spacing."""

    cleaned = _PRIVATE_USE_AREA_RE.sub("", str(text or ""))
    cleaned = cleaned.replace("\x00", "")
    return " ".join(cleaned.replace("\n", " ").split())


def load_asr_model(model_id: str) -> tuple[Any, ModelInfo]:
    configure_huggingface_cache()
    from mlx_audio.stt import load

    started = time.perf_counter()
    compatibility: list[str] = []
    if model_id == TEA_ASR_MLX_MODEL:
        configure_tea_mlx_loader()
        compatibility.extend(
            [TEA_ASR_MLX_COMPATIBILITY_NOTE, TEA_ASR_MLX_TEXT_CLEANUP_NOTE]
        )
    model = load(model_id)
    active, peak = _mlx_memory()
    return model, ModelInfo(
        name=model_id,
        load_seconds=time.perf_counter() - started,
        active_memory_bytes=active,
        peak_memory_bytes=peak,
        compatibility=compatibility,
    )


def release_model(model: Any) -> None:
    if model is None:
        return
    close = getattr(model, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    del model
    gc.collect()
    clear_mlx_cache()


def _supported_parameters(callable_object: Any) -> tuple[set[str], bool]:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return set(), True
    return set(parameters), any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _asr_kwargs(
    model: Any,
    language: str,
    context: str,
    hotwords: Sequence[str],
    max_tokens: int,
    verbose: bool,
) -> dict[str, Any]:
    supported, accepts_kwargs = _supported_parameters(model.generate)
    kwargs: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if name in supported or accepts_kwargs:
            kwargs[name] = value

    add("language", language)
    add("max_tokens", max_tokens)
    add("temperature", 0.0)
    add("verbose", verbose)
    if "hotwords" in supported or accepts_kwargs:
        kwargs["hotwords"] = list(hotwords)
    if context:
        if "context" in supported:
            kwargs["context"] = context
        elif "system_prompt" in supported or accepts_kwargs:
            kwargs["system_prompt"] = context
    return kwargs


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        value = result.get("text", result.get("transcription", ""))
    else:
        value = getattr(result, "text", getattr(result, "transcription", ""))
    return clean_asr_text(str(value or ""))


def _result_language(result: Any, default: str = "Chinese") -> str:
    value = result.get("language") if isinstance(result, dict) else getattr(result, "language", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return str(value or default)


def _result_segments(result: Any, chunk: AudioChunk, language: str) -> list[dict[str, Any]]:
    raw_segments = result.get("segments") if isinstance(result, dict) else getattr(result, "segments", None)
    if not isinstance(raw_segments, list) or not raw_segments:
        return [
            {
                "text": _result_text(result),
                "start": chunk.start,
                "end": chunk.end,
                "language": language,
            }
        ]
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        text = clean_asr_text(str(_item_value(raw, "text", default="")))
        if not text:
            continue
        try:
            start = float(_item_value(raw, "start", "start_time", default=0.0)) + chunk.start
            end = float(_item_value(raw, "end", "end_time", default=chunk.duration)) + chunk.start
        except (TypeError, ValueError):
            start, end = chunk.start, chunk.end
        segments.append({"text": text, "start": start, "end": end, "language": language})
    return segments or [
        {"text": _result_text(result), "start": chunk.start, "end": chunk.end, "language": language}
    ]


def transcribe_audio_chunks(
    model: Any,
    chunks: Sequence[PreparedAudioChunk],
    context: str,
    hotwords: Sequence[str] = HOTWORDS,
    language: str = "Chinese",
    verbose: bool = False,
) -> list[ASRChunk]:
    results: list[ASRChunk] = []
    for prepared in chunks:
        chunk = prepared.chunk
        max_tokens = max(256, min(4096, int(math.ceil(chunk.duration * 10.0)) + 128))
        kwargs = _asr_kwargs(model, language, context, hotwords, max_tokens, verbose=False)
        if verbose:
            print(
                f"[ASR] chunk {chunk.index + 1}/{len(chunks)} "
                f"{format_srt_timestamp(chunk.start)}",
                flush=True,
            )
        result = model.generate(str(prepared.path), **kwargs)
        text = _result_text(result)
        detected_language = _result_language(result, language)
        if text:
            results.append(
                ASRChunk(
                    index=chunk.index,
                    start=chunk.start,
                    end=chunk.end,
                    text=text,
                    language=detected_language,
                    audio_path=prepared.path,
                    segments=_result_segments(result, chunk, detected_language),
                )
            )
        clear_mlx_cache()
    if not results:
        raise RuntimeError("ASR 對所有音訊 chunk 都沒有回傳文字")
    return results


def align_chunk(aligner: Any, chunk: ASRChunk, audio_path: Path) -> AlignmentResult:
    if not chunk.text.strip():
        return AlignmentResult(tokens=[], fallback="segment")
    result = aligner.generate(
        str(audio_path),
        text=chunk.text,
        language=chunk.language or "Chinese",
    )
    items = getattr(result, "items", result)
    if isinstance(items, dict):
        items = items.get("items", [])
    if not isinstance(items, (list, tuple)):
        items = list(items) if items is not None else []
    tokens = normalize_alignment(
        items,
        offset=chunk.start,
        raw_text=chunk.text,
        source_chunk=chunk.index,
    )
    if not tokens:
        return AlignmentResult(tokens=[], fallback="segment")
    return AlignmentResult(tokens=tokens)


def _segment_fallback_token(chunk: ASRChunk) -> TokenTimestamp:
    return TokenTimestamp(
        text=chunk.text,
        start=chunk.start,
        end=chunk.end,
        source_chunk=chunk.index,
        fallback="segment",
    )


def align_all_chunks(
    aligner_model_id: str,
    asr_chunks: Sequence[ASRChunk],
    wav_path: Path,
    warnings: list[str] | None = None,
    prepared_chunks: Sequence[PreparedAudioChunk] | None = None,
) -> tuple[list[TokenTimestamp], list[dict[str, Any]], list[str]]:
    del wav_path
    local_warnings = warnings if warnings is not None else []
    prepared_by_index = {
        prepared.chunk.index: prepared.path for prepared in (prepared_chunks or [])
    }
    tokens: list[TokenTimestamp] = []
    segment_records: list[dict[str, Any]] = []
    aligner = None
    try:
        aligner, _ = load_asr_model(aligner_model_id)
    except Exception as exc:
        message = f"ForcedAligner 載入失敗，全部改用 segment timestamp：{type(exc).__name__}: {exc}"
        local_warnings.append(message)
        for chunk in asr_chunks:
            tokens.append(_segment_fallback_token(chunk))
            segment_records.append(
                {
                    "text": chunk.text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "source_chunk": chunk.index,
                    "fallback": "segment",
                }
            )
        return tokens, segment_records, local_warnings

    try:
        for chunk in asr_chunks:
            try:
                path = prepared_by_index.get(chunk.index, chunk.audio_path)
                result = align_chunk(aligner, chunk, path)
                if result.tokens:
                    tokens.extend(result.tokens)
                    segment_records.append(
                        {
                            "text": chunk.text,
                            "start": chunk.start,
                            "end": chunk.end,
                            "source_chunk": chunk.index,
                            "fallback": result.fallback,
                        }
                    )
                else:
                    raise RuntimeError("aligner 回傳空 token")
            except Exception as exc:
                message = (
                    f"chunk {chunk.index} 對齊失敗，改用 segment timestamp："
                    f"{type(exc).__name__}: {exc}"
                )
                local_warnings.append(message)
                tokens.append(_segment_fallback_token(chunk))
                segment_records.append(
                    {
                        "text": chunk.text,
                        "start": chunk.start,
                        "end": chunk.end,
                        "source_chunk": chunk.index,
                        "fallback": "segment",
                    }
                )
            clear_mlx_cache()
    finally:
        release_model(aligner)
    return _deduplicate_tokens(tokens), segment_records, local_warnings


def _deduplicate_tokens(tokens: Sequence[TokenTimestamp]) -> list[TokenTimestamp]:
    by_chunk: dict[int, list[TokenTimestamp]] = {}
    for token in tokens:
        by_chunk.setdefault(token.source_chunk if token.source_chunk is not None else -1, []).append(token)
    deduplicated: list[TokenTimestamp] = []
    previous: list[TokenTimestamp] = []
    for chunk_index in sorted(by_chunk):
        current = sorted(by_chunk[chunk_index], key=lambda token: (token.start, token.end))
        previous_texts = [token.text.strip() for token in previous[-32:]]
        current_texts = [token.text.strip() for token in current[:32]]
        overlap = 0
        for size in range(min(len(previous_texts), len(current_texts), 32), 1, -1):
            if previous_texts[-size:] == current_texts[:size]:
                overlap = size
                break
        current = current[overlap:]
        deduplicated.extend(current)
        previous.extend(current)
    ordered = sorted(deduplicated, key=lambda token: (token.start, token.end))
    result: list[TokenTimestamp] = []
    for token in ordered:
        if result and token.start < result[-1].end - 0.03 and token.text.strip() == result[-1].text.strip():
            continue
        result.append(token)
    return result


def _deduplicate_transcript(chunks: Sequence[ASRChunk]) -> str:
    units: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.index):
        current = _text_units(chunk.text)
        overlap = 0
        for size in range(min(len(units), len(current), 32), 1, -1):
            if units[-size:] == current[:size]:
                overlap = size
                break
        units.extend(current[overlap:])
    return _join_units(units)


def _expand_fallback_tokens(tokens: Sequence[TokenTimestamp]) -> list[TokenTimestamp]:
    expanded: list[TokenTimestamp] = []
    for token in tokens:
        if token.fallback != "segment" or _display_unit_count(token.text) <= 20:
            expanded.append(token)
            continue
        units = _text_units(token.text)
        groups: list[list[str]] = []
        group: list[str] = []
        for unit in units:
            group.append(unit)
            if len(group) >= 18 and (unit in "。！？!?；;，,、" or len(group) >= 20):
                groups.append(group)
                group = []
        if group:
            groups.append(group)
        span = max(0.5, token.end - token.start)
        step = span / max(1, len(groups))
        for index, group_units in enumerate(groups):
            expanded.append(
                TokenTimestamp(
                    text=_join_units(group_units),
                    start=token.start + step * index,
                    end=token.end if index == len(groups) - 1 else token.start + step * (index + 1),
                    source_chunk=token.source_chunk,
                    fallback="segment",
                )
            )
    return expanded


def _token_record(token: TokenTimestamp, converted_text: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "text": converted_text if converted_text is not None else token.text,
        "start": round(float(token.start), 3),
        "end": round(float(token.end), 3),
    }
    if token.source_chunk is not None:
        record["source_chunk"] = token.source_chunk
    if token.fallback:
        record["fallback"] = token.fallback
    return record


def _character_records(tokens: Sequence[TokenTimestamp]) -> list[dict[str, Any]]:
    characters: list[dict[str, Any]] = []
    for token in tokens:
        parts = _split_aligned_text(token.text)
        if not parts:
            continue
        span = max(0.0, token.end - token.start)
        step = span / max(1, len(parts))
        for index, part in enumerate(parts):
            part_start = token.start + step * index
            part_end = token.end if index == len(parts) - 1 else token.start + step * (index + 1)
            part_token = TokenTimestamp(
                part,
                part_start,
                part_end,
                token.source_chunk,
                token.fallback,
            )
            characters.append(_token_record(part_token, to_traditional(part)))
    return characters


def strip_srt_punctuation(text: str) -> str:
    """Remove standalone punctuation while preserving alphanumeric product names."""

    return _join_units(
        unit for unit in _text_units(text) if not _is_punctuation(unit)
    )


def _srt_text(cues: Sequence[SubtitleCue], no_punctuation: bool = False) -> str:
    blocks: list[str] = []
    for cue in cues:
        display_text = (
            strip_srt_punctuation(cue.text) if no_punctuation else cue.text.strip()
        )
        if not display_text:
            continue
        index = len(blocks) + 1
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}",
                    display_text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(
    input_path: Path,
    output_dir: Path,
    transcript: TranscriptBundle,
    no_punctuation: bool = False,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    srt_path = output_dir / f"{stem}.srt"
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"
    _atomic_write(srt_path, _srt_text(transcript.cues, no_punctuation=no_punctuation))
    _atomic_write(txt_path, transcript.full_transcript + ("\n" if transcript.full_transcript else ""))
    json_payload = {
        "full_transcript": transcript.full_transcript,
        "raw_transcript": transcript.raw_transcript,
        "segments": transcript.segments,
        "words": transcript.words,
        "characters": transcript.characters,
        "model": transcript.model,
        "aligner": transcript.aligner,
        "processing_time": round(transcript.processing_time, 3),
        "audio_duration": round(transcript.audio_duration, 3),
        "realtime_factor": round(transcript.realtime_factor, 5),
        "warnings": transcript.warnings,
    }
    _atomic_write(json_path, json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n")
    return srt_path, txt_path, json_path


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def run_transcription(
    input_path: Path,
    output_dir: Path | None = None,
    asr_model: str | None = None,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    hotwords: Sequence[str] = HOTWORDS,
    language: str = "Chinese",
    keep_temp: bool = False,
    no_punctuation: bool = False,
    verbose: bool = False,
) -> tuple[tuple[Path, Path, Path], TranscriptBundle]:
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("此工具只允許在 Apple Silicon arm64 上執行。")
    if chunk_seconds > MAX_ALIGNMENT_SECONDS - 10.0:
        raise ValueError("chunk_seconds 不可超過 260 秒，需保留 ForcedAligner 的長度安全邊界")

    started = time.perf_counter()
    media = probe_media(input_path)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else media.path.parent
    warnings: list[str] = []
    context = build_context(hotwords)
    temp_root = PROJECT_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(temp_root)))
    model = None
    selected_info: ModelInfo | None = None
    asr_chunks: list[ASRChunk] | None = None

    try:
        normalized_wav = extract_audio(media.path, temp_dir / "audio-16k-mono.wav")
        energies = scan_energy(normalized_wav)
        chunks = build_chunks(
            media.duration,
            energies,
            target_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        if not chunks:
            raise RuntimeError("沒有可處理的音訊 chunk")
        prepared_chunks = [
            PreparedAudioChunk(
                chunk=chunk,
                path=write_audio_chunk(
                    normalized_wav,
                    chunk,
                    temp_dir / f"chunk-{chunk.index:05d}.wav",
                ),
            )
            for chunk in chunks
        ]

        candidates = (asr_model,) if asr_model else DEFAULT_ASR_CANDIDATES
        candidate_errors: list[str] = []
        for candidate in candidates:
            candidate_model = None
            try:
                if verbose:
                    print(f"[ASR] 載入 {candidate}", flush=True)
                candidate_model, info = load_asr_model(candidate)
                candidate_results = transcribe_audio_chunks(
                    candidate_model,
                    prepared_chunks,
                    context=context,
                    hotwords=hotwords,
                    language=language,
                    verbose=verbose,
                )
                model = candidate_model
                candidate_model = None
                asr_chunks = candidate_results
                selected_info = info
                break
            except Exception as exc:
                message = f"ASR {candidate} 失敗：{type(exc).__name__}: {exc}"
                candidate_errors.append(message)
                warnings.append(message)
                if verbose:
                    print(f"[ASR] {message}", file=sys.stderr, flush=True)
            finally:
                release_model(candidate_model)

        if asr_chunks is None or selected_info is None:
            raise RuntimeError("所有 ASR candidate 都失敗：\n" + "\n".join(candidate_errors))
        release_model(model)
        model = None

        aligned_tokens, segment_records, alignment_warnings = align_all_chunks(
            aligner_model,
            asr_chunks,
            normalized_wav,
            warnings=[],
            prepared_chunks=prepared_chunks,
        )
        warnings.extend(alignment_warnings)

        raw_transcript = _deduplicate_transcript(asr_chunks)
        full_transcript = to_traditional(raw_transcript)
        converted_tokens = [
            TokenTimestamp(
                text=to_traditional(token.text),
                start=token.start,
                end=token.end,
                source_chunk=token.source_chunk,
                fallback=token.fallback,
            )
            for token in aligned_tokens
        ]
        cue_tokens = _expand_fallback_tokens(converted_tokens)
        cues = build_subtitle_cues(cue_tokens, media.duration)
        if not cues and full_transcript:
            cues = [SubtitleCue(0.0, min(media.duration, max(0.5, media.duration)), full_transcript)]
            warnings.append("沒有可用 token timestamp，SRT 使用整段 fallback cue")
        validate_cues(cues, media.duration)

        for record in segment_records:
            record["text"] = to_traditional(str(record.get("text", "")))
        words = [_token_record(token, to_traditional(token.text)) for token in aligned_tokens]
        characters = _character_records(aligned_tokens)
        processing_time = time.perf_counter() - started
        bundle = TranscriptBundle(
            full_transcript=full_transcript,
            raw_transcript=raw_transcript,
            segments=segment_records,
            words=words,
            characters=characters,
            model={
                "name": selected_info.name,
                "backend": selected_info.backend,
                "load_seconds": round(selected_info.load_seconds, 3),
                "active_memory": _format_bytes(selected_info.active_memory_bytes),
                "active_memory_bytes": selected_info.active_memory_bytes,
                "peak_memory": _format_bytes(selected_info.peak_memory_bytes),
                "peak_memory_bytes": selected_info.peak_memory_bytes,
                "compatibility": selected_info.compatibility,
            },
            aligner={"name": aligner_model, "backend": "mlx"},
            processing_time=processing_time,
            audio_duration=media.duration,
            realtime_factor=processing_time / media.duration,
            warnings=warnings,
            cues=cues,
        )
        paths = write_outputs(
            media.path,
            output_dir,
            bundle,
            no_punctuation=no_punctuation,
        )
        if verbose:
            print(
                f"[done] ASR={selected_info.name} active={_format_bytes(selected_info.active_memory_bytes)} "
                f"chunks={len(chunks)} RTF={bundle.realtime_factor:.3f}",
                flush=True,
            )
        return paths, bundle
    finally:
        release_model(model)
        if keep_temp:
            print(f"[temp] 保留暫存資料：{temp_dir}", flush=True)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 Apple Silicon Mac 上將影片/音訊轉成繁體中文字幕 SRT、TXT、JSON。"
    )
    parser.add_argument("input", type=Path, help="mp4/mov/mkv/mp3/wav/m4a 等本機檔案")
    parser.add_argument("--output-dir", type=Path, default=None, help="輸出資料夾；預設為輸入檔所在目錄")
    parser.add_argument("--asr-model", default=None, help="指定單一 ASR model；預設依序探測 TEA/Qwen")
    parser.add_argument("--aligner-model", default=DEFAULT_ALIGNER_MODEL)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--hotword", action="append", default=[], help="額外 hotword，可重複指定")
    parser.add_argument("--no-default-hotwords", action="store_true")
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="只從 SRT 顯示文字移除獨立標點；TXT、JSON 與時間戳不變",
    )
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configured_hotwords = [] if args.no_default_hotwords else list(HOTWORDS)
    configured_hotwords.extend(args.hotword)
    try:
        paths, bundle = run_transcription(
            input_path=args.input,
            output_dir=args.output_dir,
            asr_model=args.asr_model,
            aligner_model=args.aligner_model,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            hotwords=configured_hotwords,
            language=args.language,
            keep_temp=args.keep_temp,
            no_punctuation=args.no_punctuation,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"SRT: {paths[0]}")
    print(f"TXT: {paths[1]}")
    print(f"JSON: {paths[2]}")
    print(f"ASR: {bundle.model['name']}")
    print(f"Aligner: {bundle.aligner['name']}")
    print(f"RTF: {bundle.realtime_factor:.3f}")
    if bundle.warnings:
        print(f"Warnings: {len(bundle.warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
