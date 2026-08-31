"""
Duration-fit retry loop.

Text length (word count, character count, ...) is not a reliable proxy for
how long Chatterbox's output will actually be -- the only way to know is to
really synthesize it and measure the resulting audio. So for each segment:
translate the English text, actually synthesize it, measure it. If it
overflows the original time slot, ask a local LLM (Ollama) to rewrite the
ENGLISH text more concisely (same meaning), retranslate, and measure again.
Repeat up to MAX_ATTEMPTS times, then keep whichever attempt came closest to
the target -- never condensing further once an attempt already fits.

Deliberately does NOT retry underrun (audio shorter than target): a short
dub just leaves silence, which doesn't cause the cascading-drift problem
overflow does (see components/assembly.py's per-speaker placement cursor --
only an overflowing segment pushes every later segment from that speaker
later too). There's nothing to "fix" about a segment finishing early.
"""

from dataclasses import dataclass
from typing import Callable

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
CONDENSE_MODEL = "qwen2.5:14b"

TOLERANCE_S = 0.05   # "exact match" -- effectively zero, bounded only by
                     # audio-sample/frame granularity, not a real slack window
MAX_ATTEMPTS = 4     # 1 original translation + up to 3 condensed retries
SHRINK_STEP = 0.20   # ask for ~20% shorter English each retry, compounding


@dataclass
class FitAttempt:
    english_text: str
    hindi_text: str
    audio_path: str
    duration: float
    overflow: float   # duration - target_duration ( >0 too long, <0 too short )


@dataclass
class FitResult:
    best: FitAttempt
    attempts: list[FitAttempt]
    fit_on_first_try: bool


def condense_english(text: str, shrink_fraction: float) -> str:
    """
    Ask a local LLM to rewrite `text` to roughly (1 - shrink_fraction) of its
    original length, preserving meaning. Runs entirely locally (Ollama on
    localhost) -- no cross-venv call needed, this is why condensing can live
    in the same process as translation (see CLAUDE.md's Environments section
    for what genuinely can't share a process: chatterbox-tts's pinned deps).
    """
    target_pct = round((1 - shrink_fraction) * 100)
    prompt = (
        f'Rewrite this English sentence to be approximately {target_pct}% of '
        f'its original length (by word count), while preserving its meaning '
        f'as closely as possible. Keep it natural, spoken-style English -- '
        f'this is a transcript of someone talking.\n\n'
        f'Original: "{text}"\n\n'
        f'Respond with ONLY the rewritten sentence -- no quotes, no '
        f'explanation, nothing else.'
    )

    response = requests.post(OLLAMA_URL, json={
        "model": CONDENSE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=60)
    response.raise_for_status()

    rewritten = response.json()["message"]["content"].strip()
    rewritten = rewritten.strip('"').strip("'").strip()
    return rewritten or text


def fit_segment(
    english_text: str,
    target_duration: float,
    translate_fn: Callable[[str], str],
    synthesize_fn: Callable[[str], tuple[str, float]],
    condense_fn: Callable[[str, float], str] = condense_english,
) -> FitResult:
    """
    translate_fn(english_text) -> hindi_text
    synthesize_fn(hindi_text) -> (audio_path, duration_seconds) -- a REAL
        synthesis call (Chatterbox), not an estimate. Caller decides how:
        directly in-process (fine on this Mac, where chatterbox-tts and
        pyannote coexist) or via a cross-venv worker (needed on the Linux
        GPU box, where they can't).
    condense_fn(english_text, shrink_fraction) -> shorter english_text,
        same meaning. Swappable for tests; defaults to the real Ollama call.
    """
    attempts: list[FitAttempt] = []
    current_text = english_text

    for attempt_num in range(MAX_ATTEMPTS):
        hindi = translate_fn(current_text)
        path, duration = synthesize_fn(hindi)
        overflow = duration - target_duration
        attempts.append(FitAttempt(current_text, hindi, path, duration, overflow))

        if overflow <= TOLERANCE_S:
            # Fits, or already under -- stop. Nothing to condense for.
            return FitResult(best=attempts[-1], attempts=attempts, fit_on_first_try=attempt_num == 0)

        if attempt_num == MAX_ATTEMPTS - 1:
            break

        current_text = condense_fn(current_text, SHRINK_STEP * (attempt_num + 1))

    # Closest to target wins, regardless of direction -- NOT max(0, overflow),
    # which would treat any undershoot as "perfect" no matter how severe
    # (e.g. Ollama over-condensing a line into something 3x too short) and
    # let it beat a mildly-over candidate that speed_adjust could have
    # trivially fixed within its safety bound.
    best = min(attempts, key=lambda a: abs(a.overflow))
    return FitResult(best=best, attempts=attempts, fit_on_first_try=False)
