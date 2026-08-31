"""
Speaker alignment — joins transcription output against diarization output
by timestamp overlap. Pure interval-matching logic, no ML model involved:
transcription knows *what* was said, diarization knows *who* was talking
*when*, and neither knows the other's half.
"""

from collections import defaultdict
from dataclasses import replace

from data_types import Segment, SpeakerSpan


def assign_speakers(segments: list[Segment], speaker_spans: list[SpeakerSpan]) -> list[Segment]:
    """
    For each transcribed segment, find whichever speaker accounts for the
    most overlapping time within that segment's [start, end) window, and
    tag the segment with that speaker. Use diarization's overlap-aware
    spans here (not the "exclusive" ones from extract_speaker_audio) — a
    transcript segment spanning a turn change or brief cross-talk still
    needs one best-guess speaker attributed to it, and only the full
    overlap-aware span list has enough information to pick the dominant one.

    Segments with no overlapping speaker span at all (e.g. Whisper found
    speech that diarization missed) are left with speaker=None.
    """
    tagged = []
    for seg in segments:
        overlap_by_speaker = defaultdict(float)
        for span in speaker_spans:
            overlap = min(seg.end, span.end) - max(seg.start, span.start)
            if overlap > 0:
                overlap_by_speaker[span.speaker] += overlap

        speaker = max(overlap_by_speaker, key=overlap_by_speaker.get) if overlap_by_speaker else None
        tagged.append(replace(seg, speaker=speaker))

    return tagged
