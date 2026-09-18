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
