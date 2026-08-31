"""
Shared data contracts passed between pipeline components.

Named data_types.py (not types.py) deliberately — types.py would shadow
Python's own stdlib `types` module for any code run from this project
directory, which libraries like torch/pyannote rely on internally.
"""

from dataclasses import dataclass


@dataclass
class SpeakerSpan:
    """
    One span of time attributed to a single speaker by diarization.
    Purely acoustic — no idea what was said. Spans from different speakers
    may legitimately overlap (real cross-talk); spans from the SAME speaker
    should not.
    """
    start: float
    end: float
    speaker: str   # anonymous cluster id from pyannote, e.g. "SPEAKER_00"


@dataclass
class Segment:
    """
    One transcribed/translated line of speech. `speaker` is None until the
    alignment component tags it against diarization output — transcription
    on its own has no idea who said anything, same way diarization on its
    own has no idea what was said.
    """
    start: float
    end: float
    text: str                  # already-translated text (see components/transcription.py)
    speaker: str | None = None


@dataclass
class SynthesizedSegment:
    """One segment of synthesized dubbed audio, ready for assembly."""
    start: float
    end: float
    duration: float            # actual synthesized audio duration (post duration-matching)
    path: str
    speaker: str | None
    text: str
