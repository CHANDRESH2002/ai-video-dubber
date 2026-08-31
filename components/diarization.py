"""
Speaker diarization: "who spoke when," purely acoustic — no idea what was
said. That gets matched against transcription separately, in a later
component (timestamp-overlap join), not here.
"""

from pathlib import Path

from config import DIARIZATION_MODEL
from data_types import SpeakerSpan


def diarize(
    audio_path: Path,
    hf_token: str,
    num_speakers: int | None = None,
    exclusive: bool = False,
) -> list[SpeakerSpan]:
    """
    Run pyannote's speaker-diarization pipeline on a mono audio file.

    audio_path: path to the audio to diarize. Should be the isolated vocals
        track (e.g. Demucs output), not the raw mixed audio — a clean
        voice-only signal gives the speaker-embedding clustering step less
        noise to fight, consistent with how transcription already uses the
        vocals track in pipeline.py.
    num_speakers: pass if you already know the exact count (improves
        accuracy). Leave None to let pyannote estimate it via clustering.
    exclusive: pyannote produces two versions of its output from the same
        run — the default (exclusive=False) keeps overlapping spans from
        different speakers when they genuinely talk over each other, which
        is what a later multi-track assembly component will need to know
        about. exclusive=True instead returns pyannote's "exclusive"
        annotation, which drops overlapping speech entirely so each moment
        is attributed to at most one speaker. Use exclusive=True whenever
        you're about to pull raw audio out of the SAME single-track file
        (e.g. extract_speaker_audio() below) — audio.wav/vocals.wav is one
        mixed-down track, so grabbing a time range where two people are
        talking at once gets you both voices smashed together, not one
        clean voice. Diarization only labels time ranges; it can't un-mix
        audio that was recorded as a single track — that's a different,
        much harder problem (source separation).

    Returns a chronological list of SpeakerSpans.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    output = pipeline(str(audio_path), **kwargs)
    # pyannote.audio 4.x returns a DiarizeOutput wrapper; the actual
    # pyannote.core.Annotation (with .itertracks()) is one of these two:
    diarization = output.exclusive_speaker_diarization if exclusive else output.speaker_diarization

    spans = [
        SpeakerSpan(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    spans.sort(key=lambda s: s.start)
    return spans


def extract_speaker_audio(audio_path: Path, spans: list[SpeakerSpan], speaker: str, output_path: Path) -> Path:
    """
    Concatenate every span attributed to one speaker into a single audio
    file — "everything this speaker said, back to back." Two uses:
      1. Verification: if diarization got confused, this is where it'd show
         up audibly (a different voice mixed in, or missing lines).
      2. Later: this becomes the per-speaker reference clip for XTTS voice
         cloning, replacing the single global reference pipeline.py uses now.
    """
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(str(audio_path))
    chunks = [
        data[int(span.start * sr):int(span.end * sr)]
        for span in spans
        if span.speaker == speaker
    ]
    combined = np.concatenate(chunks) if chunks else np.zeros(0)
    sf.write(str(output_path), combined, sr)
    return output_path
