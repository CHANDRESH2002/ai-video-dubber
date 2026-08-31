"""
Subtitles — burns the ORIGINAL-language text onto a dubbed video, so a
viewer can read what was actually said while hearing the translated/
synthesized voice. Independent of which TTS engine produced the dub.
"""

import subprocess
from pathlib import Path


def _srt_timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(entries: list[tuple[float, float, str]], srt_path: Path) -> None:
    """entries: [(start_seconds, end_seconds, text), ...], in chronological order."""
    lines = []
    for i, (start, end, text) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    """
    Hardcodes the subtitles into the video frames (not a toggleable soft
    track) so they're guaranteed visible in any player without the viewer
    needing to enable anything. Requires re-encoding the video stream
    (unlike the rest of the pipeline's muxing steps, which use -c:v copy)
    since burning text into frames isn't a stream-copy operation.
    """
    # ffmpeg's subtitles filter takes the srt path as part of a filter
    # string, where ':' and other special characters need escaping.
    escaped_srt = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles={escaped_srt}:force_style='FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3'",
        "-c:a", "copy",
        str(output_path),
    ], check=True, capture_output=True)
