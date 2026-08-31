"""
Fallback translation path for target languages IndicTrans2 doesn't cover
(anything non-Indic — es, fr, de, ...). See components/translation.py for
why IndicTrans2 is the primary path for Indic languages: measured
head-to-head on the same clip, this Ollama/Qwen2.5:7b approach scored much
worse (avg CometKiwi 0.603 vs IndicTrans2's 0.803, 60% of lines flagged
low-quality vs 0%) and occasionally produced outright garbled non-target-
language output. Kept only because IndicTrans2 has no equivalent for
non-Indic languages yet — if a stronger free/local option for those
languages replaces this, this file should be replaced the same way the old
version of translation.py just was.

Requires Ollama running (`brew services start ollama`) with MODEL pulled
(`ollama pull qwen2.5:7b`) — not installed by default; this path is
untested since the model was removed from this machine to save memory.
"""

import re

import requests

from data_types import Segment

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"   # what's actually pulled locally; the 7b this originally
                        # targeted was removed from this machine to save memory
MAX_ATTEMPTS = 3

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
    "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "hu": "Hungarian", "ko": "Korean",
    "hi": "Hindi",
}

_SCRIPT_PATTERNS = {
    "cjk": re.compile(r"[一-鿿]"),
    "kana": re.compile(r"[぀-ヿ]"),
    "hangul": re.compile(r"[가-힣]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
    "arabic": re.compile(r"[؀-ۿ]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
}

_EXPECTED_SCRIPTS = {
    "zh": {"cjk"}, "ja": {"cjk", "kana"}, "ko": {"hangul"},
    "ru": {"cyrillic"}, "ar": {"arabic"}, "hi": {"devanagari"},
}


def _has_unexpected_script(text: str, target_lang: str) -> bool:
    expected = _EXPECTED_SCRIPTS.get(target_lang, set())
    return any(
        script not in expected and pattern.search(text)
        for script, pattern in _SCRIPT_PATTERNS.items()
    )


def _translate_one(text: str, context: str, target_lang: str) -> str | None:
    language_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    prompt = f"""Here is a conversation transcript, for context only:

{context}

Using that conversation to resolve ambiguous words, idioms, or domain-
specific meaning (e.g. financial or technical jargon), translate ONLY this
one line into {language_name} ({target_lang}):

"{text}"

Respond with ONLY the translated sentence, written in {language_name} — no
quotes, no explanation, no line numbers, no speaker tags, no other
language, nothing else."""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }, timeout=120)
        response.raise_for_status()

        translated = response.json()["message"]["content"].strip()
        translated = translated.strip('"').strip("'").strip()

        if translated and not _has_unexpected_script(translated, target_lang):
            return translated

        print(f"    ⚠ attempt {attempt}/{MAX_ATTEMPTS} looked wrong for "
              f"\"{text[:40]}\" -> \"{translated[:40]}\", retrying...")

    return None


def translate_lines(
    all_segments: list[Segment],
    non_empty: list[tuple[int, Segment]],
    target_lang: str,
) -> list[str]:
    """Returns translations in the same order as non_empty."""
    context = "\n".join(
        f"[{seg.speaker or 'UNKNOWN'}] {seg.text}"
        for seg in all_segments
    )

    results = []
    for _, seg in non_empty:
        text = seg.text.strip()
        translated = _translate_one(text, context, target_lang)
        results.append(translated or text)
    return results
