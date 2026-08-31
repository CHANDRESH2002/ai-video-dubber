"""
Evaluation — objective metrics instead of "sounds right to a human."

Metrics so far:
  1. Duration-match quality: formalizes what synthesize()'s per-segment
     prints already showed informally.

     First version measured deviation as a PERCENTAGE of each segment's
     target duration, which blows up for very short targets — "Yeah."
     (target 0.1s, actual 0.6s) is a 0.5s miss in absolute terms, but a
     nonsensical 500% in relative terms. Real evaluation runs saw this
     single-metric swing between 54% and 155% run to run purely because of
     which short segments happened to land where — not because duration-
     matching quality actually changed that much. Fixed by measuring
     deviation in seconds (doesn't blow up regardless of segment length)
     and by giving short segments a fixed minimum tolerance floor instead
     of a percentage-of-target one (nothing can be timed tighter than
     synthesis granularity allows, no matter how short the target is).
     `total_overflow_pct` didn't have this problem — it's a ratio of SUMS
     (total actual / total target), not an average of per-segment ratios,
     so one tiny segment can't dominate it — and is unchanged.
  2. Speaker-similarity: cosine similarity between a speaker's real
     reference clip and a synthesized clip claiming to clone that voice,
     using the same embedding model pyannote's diarization pipeline already
     uses internally (pyannote/wespeaker-voxceleb-resnet34-LM) — no new
     gated model, no new HuggingFace license to accept.

     Sanity range from manual validation on our test clip: same-speaker
     similarity ~0.69, cross-speaker (two different real people) ~0.19 —
     so this metric clearly separates "same voice" from "different voice,"
     it isn't just noise.
  3. Translation quality: CometKiwi (Unbabel/wmt22-cometkiwi-da), a
     reference-free Quality Estimation model — the standard approach (best
     correlation with human judgment in the WMT23 Metrics Shared Task) for
     scoring MT quality when there's no gold-standard reference translation
     to compare against, which is our situation for any of our own clips.

     Sanity check on our own data: a known-accurate translation scored 0.88;
     a known-hallucinated one (Whisper inventing "Rs. 16,000" with no
     matching number anywhere in the source) scored 0.52 — so it's actually
     catching the exact problem we found by hand earlier, not just noise.

Not yet covered (deferred — need ground truth data and/or new models):
DER (diarization accuracy), WER (transcription accuracy — needs a human-
verified reference transcript), and — the metric that actually measures
this project's original goal — emotion preservation, which needs a
speech-emotion-recognition model we don't have yet.
"""

from pathlib import Path

import numpy as np

from data_types import Segment, SynthesizedSegment


def duration_match_metrics(
    segments: list[Segment],
    synthesized: list[SynthesizedSegment],
    min_tolerance_s: float = 0.3,
    relative_tolerance: float = 0.10,
) -> dict:
    """
    min_tolerance_s: a segment counts as "on time" if it's within this many
        seconds of its target, OR within relative_tolerance of it —
        whichever is MORE forgiving. This is what actually fixes the short-
        segment problem: a 0.1s target segment only has to land within
        0.3s (the floor) to count as fine, instead of an unachievable
        0.01s (10% of 0.1s). A 6s target segment still has to land within
        0.6s (10% wins there, since it's bigger than the 0.3s floor).
    """
    targets = [s.end - s.start for s in segments if s.text.strip()]
    total_target = sum(targets)
    total_actual = sum(s.duration for s in synthesized)

    abs_deviations = []
    within_tolerance = 0
    for s in synthesized:
        target = s.end - s.start
        if target <= 0:
            continue
        deviation = abs(s.duration - target)
        abs_deviations.append(deviation)
        tolerance = max(min_tolerance_s, relative_tolerance * target)
        if deviation <= tolerance:
            within_tolerance += 1

    n = len(abs_deviations)
    return {
        "num_segments": len(synthesized),
        "total_target_s": round(total_target, 1),
        "total_actual_s": round(total_actual, 1),
        "total_overflow_pct": round((total_actual / total_target - 1) * 100, 1) if total_target else 0.0,
        "avg_absolute_deviation_s": round(sum(abs_deviations) / n, 2) if n else 0.0,
        "pct_within_tolerance": round(within_tolerance / n * 100, 1) if n else 0.0,
    }


def load_speaker_embedder(hf_token: str):
    """Load once, reuse across many speaker_similarity() calls — loading
    the model per-call would be wasteful when scoring many segments."""
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=hf_token)
    return Inference(model, window="whole")


def speaker_similarity(reference_path: Path, synthesized_path: Path, embedder) -> float:
    """Cosine similarity in [-1, 1] between two voices. Higher = closer match."""
    emb_ref = embedder(str(reference_path))
    emb_synth = embedder(str(synthesized_path))
    return float(np.dot(emb_ref, emb_synth) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_synth)))


def load_translation_quality_model(saving_directory: str = "temp/comet_models"):
    """Load once, reuse across many translation_quality() calls. Downloads
    the checkpoint on first use (a few GB) — requires having accepted the
    license at https://huggingface.co/Unbabel/wmt22-cometkiwi-da and a
    valid HF_TOKEN (same one used for pyannote)."""
    from comet import download_model, load_from_checkpoint
    path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory=saving_directory)
    return load_from_checkpoint(path)


def translation_quality(model, pairs: list[dict]) -> list[float]:
    """
    pairs: [{"src": original-language text, "mt": translated text}, ...]
    Returns one quality score per pair, roughly in [0, 1], higher = better.
    No reference translation needed — that's the whole point of using a
    reference-free QE model here.

    num_workers=1 (not comet's own default of 0 when gpus=0) works around a
    bug in comet 2.2.7: it unconditionally sets a DataLoader
    multiprocessing_context whenever MPS is available (true on Apple
    Silicon), which errors when combined with num_workers=0.
    """
    output = model.predict(pairs, batch_size=8, gpus=0, num_workers=1, progress_bar=False)
    return output.scores
