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


def build_speaker_blocks(
    spans: list[SpeakerSpan], gap_threshold: float = 1.5, max_duration: float | None = None,
) -> list[SpeakerSpan]:
    """
    Merges consecutive same-speaker EXCLUSIVE diarization spans (see
    components/diarization.py's exclusive=True -- non-overlapping by
    construction, safe to slice audio on) into continuous blocks, so
    per-speaker transcription (transcribe_per_speaker()) gets natural
    sentence-length audio instead of many tiny, fragmented, context-free
    ASR calls on pyannote's often-choppy raw turns (a single real speaker
    can easily produce 5-10+ short turns in under a minute, split by
    micro-pauses that aren't real speaker changes).

    Bridges gaps up to gap_threshold seconds between spans from the SAME
    speaker (a natural pause within continuous speech), but always cuts
    immediately at any genuine speaker change regardless of gap size --
    this is what guarantees a merged block can never contain more than one
    speaker. gap_threshold=1.5s is a first-pass heuristic (based on the
    gaps observed in real diarization output on this project's test
    clips), not empirically tuned across a broad sample -- revisit if
    blocks come out wrongly split or wrongly merged on a new clip.

    max_duration (optional, default None = unbounded): also cuts to a new
    block once the running block would exceed this many seconds, even for
    the same speaker with a short gap. Added for components_indic/'s
    Hindi-source pipeline -- IndicConformer (unlike SenseVoice) emits no
    punctuation at all, so there's no sentence-level split happening
    downstream inside a block; without a cap, one merged block could become
    a multi-sentence paragraph handed whole to translation and to IndicF5's
    synthesis, which (like most F5-style TTS) is built around utterance-
    length inputs. Left as None (no behavior change) for the existing
    English->Hindi pipeline, which doesn't need this -- SenseVoice's own
    punctuation-based reconstruct_segments() already keeps final segments
    short regardless of block length.
    """
    spans = sorted(spans, key=lambda s: s.start)
    blocks: list[SpeakerSpan] = []
    for span in spans:
        fits_gap = blocks and blocks[-1].speaker == span.speaker and span.start - blocks[-1].end <= gap_threshold
        fits_duration = max_duration is None or not blocks or (span.end - blocks[-1].start) <= max_duration
        if fits_gap and fits_duration:
            blocks[-1] = replace(blocks[-1], end=span.end)
        else:
            blocks.append(span)
    return blocks
