"""
Hindi first-person gender-agreement corrector -- LLM-only (local Ollama,
qwen2.5:14b), no regex rule tier. Built from the ACTUAL failure patterns
observed in a real-speech baseline test (two full TED talks, one female
speaker one male): before correction, the female speaker's first-person
lines came out 1/9 correct (IndicTrans2 defaults to masculine regardless
of actual speaker gender); after, 9/9, with the already-correct male
speaker's lines (9/10) completely unchanged.

An earlier version of this file had a fast regex-rule tier that ran first
and only fell back to the LLM when no rule matched at all. That tier was
removed deliberately: because it returned immediately on ANY match, a
sentence like "मैं अकेला पड़ जाता हूँ" (adjective अकेला + verb जाता, both
needing to agree) only got its VERB fixed -- the rule matched "ता हूँ" and
returned before the LLM (whose prompt explicitly checks "any verb or
adjective") ever got a chance to see the sentence at all, leaving a
grammatically-inconsistent mixed result. Sending every sentence to the LLM
directly avoids that class of bug, at the cost of a slower, per-sentence
model call instead of a free string substitution for the common cases.

Known LLM failure mode (caught during a real-data audit, not
hypothetical): it once deleted an entire clause instead of just fixing
agreement. _looks_safe() below rejects any response that's suspiciously
shorter than the input and falls back to the ORIGINAL (uncorrected but at
least complete) text rather than risk silently losing content in
production.
"""
import requests

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL = "qwen2.5:14b"

# If the LLM's "corrected" sentence is shorter than this fraction of the
# original, treat it as a failed correction (content likely dropped) and
# keep the original text instead.
_MIN_LENGTH_RATIO = 0.7


def _looks_safe(original: str, corrected: str) -> bool:
    """Rejects suspiciously short LLM output -- guards against the
    observed sentence-deletion failure mode."""
    if not corrected.strip():
        return False
    return len(corrected) >= _MIN_LENGTH_RATIO * len(original)


def _llm_correct(text: str, target_gender: str) -> str:
    prompt = (
        f"This Hindi sentence is spoken by a {target_gender} speaker, referring "
        f"to themselves in first person (मैं). Check if any verb or adjective "
        f"needs to grammatically agree with a {target_gender} speaker's gender, "
        f"and fix it if wrong. If it's already correct or has no gender-marked "
        f"words, return it unchanged.\n\n"
        f'Sentence: "{text}"\n\n'
        f"Respond with ONLY the corrected (or unchanged) Hindi sentence -- "
        f"no explanation, no quotes."
    )
    response = requests.post(_OLLAMA_URL, json={
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].strip().strip('"').strip("'")


def correct_gender_agreement(text: str, target_gender: str) -> dict:
    """
    target_gender: "male" or "female" -- the KNOWN correct gender for this
    text's speaker (see components/speaker_gender.py). Callers should skip
    calling this entirely for "unknown" gender rather than guess.

    Returns {"text": corrected_text, "method": "llm"|"llm_rejected"|"none", "changed": bool}
    """
    if not text.strip():
        return {"text": text, "method": "none", "changed": False}

    try:
        llm_result = _llm_correct(text, target_gender)
    except requests.exceptions.RequestException as e:
        # Ollama not reachable (e.g. not installed on this machine) --
        # same fallback pattern as components/style_prompt.py: keep the
        # original, uncorrected text rather than crash the whole pipeline
        # over an optional refinement step.
        print(f"  [gender_agreement] Ollama unavailable, skipping correction: {e}")
        return {"text": text, "method": "llm_unavailable", "changed": False}

    if not _looks_safe(text, llm_result):
        return {"text": text, "method": "llm_rejected", "changed": False}
    return {"text": llm_result, "method": "llm", "changed": llm_result != text}
