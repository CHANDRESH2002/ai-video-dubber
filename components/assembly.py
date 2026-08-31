"""
Assembly — mixes synthesized per-segment audio into one final dubbed video.

Ports pipeline.py's assemble(), extended for multiple speakers: instead of
one canvas + one cursor (pipeline.py assumed a single voice), each speaker
gets their OWN canvas and cursor. A cursor still stops a single speaker's
segments from overlapping themselves (summing overlapping speech from the
same person would produce unintelligible double-talk, same reasoning as
pipeline.py). But the per-speaker canvases are then summed together, which
DOES let different speakers overlap — real cross-talk is exactly what we
want preserved, and it's the whole reason diarization tracked overlapping
spans in the first place.
"""

import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from data_types import SynthesizedSegment

SR = 22050  # target sample rate for the assembled track (XTTS itself outputs 24kHz; resampled below)


def compute_actual_placements(segments: list[SynthesizedSegment]) -> dict[int, tuple[float, float]]:
    """
    Mirrors the per-speaker placement cursor used when building the final
    mix (a segment can't start before the previous one from the SAME
    speaker finished) WITHOUT actually building any audio. Returns
    {id(seg): (actual_start_seconds, actual_end_seconds)} for every segment.

    This is the single source of truth for "where did this line actually
    end up" — both _build_speaker_canvas below and anything that times
    subtitle tracks to the real audio (see components/multilang_package.py)
    call this, so the two can never silently disagree with each other by
    each re-deriving the cursor logic their own way.

    Keyed by id(seg) (Python object identity), not an index — callers must
    pass the same SynthesizedSegment objects to both this function and
    whatever else reads its result, not copies.
    """
    by_speaker = defaultdict(list)
    for seg in segments:
        by_speaker[seg.speaker].append(seg)

    placements: dict[int, tuple[float, float]] = {}
    for speaker, segs in by_speaker.items():
        cursor = 0.0
        for seg in sorted(segs, key=lambda s: s.start):
            actual_start = max(seg.start, cursor)
            actual_end = actual_start + seg.duration
            placements[id(seg)] = (actual_start, actual_end)
            cursor = actual_end
    return placements


def _build_speaker_canvas(
    segments: list[SynthesizedSegment],
    placements: dict[int, tuple[float, float]],
    num_samples: int,
) -> np.ndarray:
    """One speaker's segments placed on their own timeline, at the actual
    positions already decided by compute_actual_placements()."""
    canvas = np.zeros(num_samples)
    for seg in segments:
        data, seg_sr = sf.read(seg.path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        if seg_sr != SR:
            import torch
            import torchaudio
            t = torch.from_numpy(data).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(seg_sr, SR)
            data = resampler(t).squeeze(0).numpy()

        actual_start, _ = placements[id(seg)]
        start = int(actual_start * SR)
        end = start + len(data)

        if end > len(canvas):
            canvas = np.pad(canvas, (0, end - len(canvas)))

        canvas[start:end] += data

    return canvas


def build_mixed_audio(
    synthesized_segments: list[SynthesizedSegment],
    background_path: Path,
    video_duration: float,
    temp_dir: Path,
    out_name: str = "final_audio.wav",
) -> Path:
    """
    Places every segment at its actual (post-cursor) position, mixes with
    background, and returns the path to ONE finished audio track for this
    language. Split out from assemble() so a multi-language run (see
    components/multilang_package.py) can build several of these -- one per
    target language, sharing the same background track -- without muxing
    each into its own throwaway video first.
    """
    num_samples = int(video_duration * SR) + SR  # +1s padding
    placements = compute_actual_placements(synthesized_segments)

    by_speaker = defaultdict(list)
    for seg in synthesized_segments:
        by_speaker[seg.speaker].append(seg)

    print(f"Building per-speaker tracks for: {sorted(by_speaker.keys(), key=str)}")
    combined = np.zeros(num_samples)
    for speaker, segs in by_speaker.items():
        speaker_canvas = _build_speaker_canvas(segs, placements, num_samples)
        if len(speaker_canvas) > len(combined):
            combined = np.pad(combined, (0, len(speaker_canvas) - len(combined)))
        combined[:len(speaker_canvas)] += speaker_canvas

    peak = np.max(np.abs(combined))
    if peak > 0:
        combined = combined / peak * 0.85

    temp_dir.mkdir(parents=True, exist_ok=True)
    dubbed_speech = temp_dir / "dubbed_speech.wav"
    sf.write(str(dubbed_speech), combined, SR)

    final_audio = temp_dir / out_name
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(dubbed_speech),
        "-i", str(background_path),
        "-filter_complex",
        # normalize=0 -- without it, ffmpeg's amix silently auto-scales the
        # WHOLE mix down to avoid clipping, which (measured on a real run)
        # made the final track quieter than either input alone even though
        # neither was near clipping. The volume= weights below are already
        # the deliberate speech-forward balance; normalize=0 stops amix
        # from overriding them with its own guess.
        "[0:a]volume=1.0[speech]; [1:a]volume=0.8[bg]; [speech][bg]amix=inputs=2:duration=longest:normalize=0",
        str(final_audio)
    ], check=True, capture_output=True)

    return final_audio


def assemble(
    synthesized_segments: list[SynthesizedSegment],
    background_path: Path,
    input_video_path: Path,
    output_video_path: Path,
    temp_dir: Path,
) -> None:
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video_path)
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())

    final_audio = build_mixed_audio(synthesized_segments, background_path, duration, temp_dir)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_video_path),
        "-i", str(final_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_video_path)
    ], check=True, capture_output=True)

    print(f"Final video → {output_video_path}")
