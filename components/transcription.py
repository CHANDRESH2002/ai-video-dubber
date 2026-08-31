"""
Transcription — audio to text, translated directly into English in one pass.

This ports the approach validated in pipeline.py's transcribe(): using
Whisper's built-in task="translate" (audio -> English text directly) instead
of task="transcribe" + a separate translation step. That matters for
code-switched speech (a speaker dropping in words from another language) —
transcribing forces every word into one language's script, mangling embedded
foreign words, before translation ever sees them. Decoding straight to
English sidesteps that entirely. Trade-off: only ever outputs English.

Two backends, picked by platform: mlx_whisper on macOS (Apple's MLX
framework needs Apple Silicon — this is what's been validated all along on
the Mac) and faster-whisper everywhere else (a CTranslate2-based Whisper
implementation with real CUDA acceleration, for Linux/rented-GPU boxes,
where mlx_whisper can't even install). Same large-v3 model either way, same
task="translate" behavior, same Segment output contract — only the engine
underneath differs.
"""

import sys
from pathlib import Path

from data_types import Segment


def _transcribe_mlx(audio_path: Path, language: str | None) -> list[dict]:
    import mlx_whisper

    kwargs = {}
    if language is not None:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        task="translate",
        word_timestamps=False,
        verbose=False,
        **kwargs,
    )
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]


def _transcribe_faster_whisper(audio_path: Path, language: str | None) -> list[dict]:
    from faster_whisper import WhisperModel

    from config import best_device

    device = best_device()
    ct2_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if ct2_device == "cuda" else "int8"
    model = WhisperModel("large-v3", device=ct2_device, compute_type=compute_type)

    segments, _info = model.transcribe(
        str(audio_path), task="translate", language=language, word_timestamps=False,
    )
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


def transcribe(audio_path: Path, duration: float | None = None, language: str | None = None) -> list[Segment]:
    """
    audio_path: the isolated vocals track (e.g. Demucs output).
    duration: the source video/audio's real duration, in seconds. Used to
        filter out Whisper hallucinations that land past the true end of
        content (e.g. "Closed Captioning provided by...") — pass this in
        rather than trusting audio_path's own duration, which separation/
        extraction can pad by a few ms, letting hallucinations slip past a
        same-file check. Pass None to skip this filtering.
    language: optional hint for the source language (e.g. "hi"). Leave None
        to let Whisper auto-detect — task="translate" outputs English either
        way, so this only affects detection accuracy, not output language.
    """
    if sys.platform == "darwin":
        raw_segments = _transcribe_mlx(audio_path, language)
    else:
        raw_segments = _transcribe_faster_whisper(audio_path, language)

    if duration is not None:
        raw_segments = [s for s in raw_segments if s["start"] < duration - 0.5]

    return [
        Segment(start=s["start"], end=s["end"], text=s["text"].strip())
        for s in raw_segments
    ]
