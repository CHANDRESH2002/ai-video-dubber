"""
Translation.

For Hindi (and any other Indic target we add XTTS support for later), uses
IndicTrans2 (ai4bharat/indictrans2-en-indic-dist-200M) — a translation-
specialized seq2seq model, not a chat LLM. Measured head-to-head against
the earlier Ollama/Qwen2.5:7b approach on the same 30-line real clip using
CometKiwi (reference-free MT quality estimation, see components/evaluation.py):

    Qwen2.5:7b (Ollama, per-line, chat-prompted): avg 0.603, 18/30 lines
        flagged low-quality (<0.6), worst line 0.308 — one line came back
        as outright garbage ("A positive..." -> "प Osborne"), not Hindi at
        all. A 7B general chat model asked to translate one line at a time
        apparently isn't reliable enough at this size.
    IndicTrans2-dist-200M (this file): avg 0.803, 0/30 lines low-quality,
        worst line 0.613. Also 6x smaller (750MB vs 4.7GB) and doesn't need
        Ollama running or a per-line network round trip.

Being a dedicated MT model (not a chat model) also removes the failure mode
that motivated the old script-anomaly retry logic entirely: there's no
"conversational" output for it to go off-script into, so it can't produce
the kind of non-Hindi garbage Qwen did.

Trade-off: IndicTrans2's en-indic model only covers Indic languages (hi,
bn, ta, te, ...). For everything else it doesn't cover (es, fr, de, ...),
this falls back to the earlier Ollama/local-LLM approach, which requires
`ollama pull <model>` for whatever OLLAMA_FALLBACK_MODEL names below.

No conversation-level context is passed to IndicTrans2 (each line is
translated independently) — unlike the old per-line-with-full-context Ollama
prompt. Measured quality was still far better despite this, so context
apparently wasn't the bottleneck; model reliability was.
"""

from dataclasses import replace

import re

from data_types import Segment

# ISO 639-1 (our config.py's XTTS_LANG_MAP) -> FLORES-200 code IndicTrans2
# expects. Only languages XTTS can actually speak are worth listing here;
# add more as XTTS/config.py gains Indic language support.
_INDICTRANS2_LANG_CODES = {
    "hi": "hin_Deva",
}
_SRC_LANG_CODE = "eng_Latn"
_INDICTRANS2_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"

_indictrans2_model = None
_indictrans2_tokenizer = None
_indictrans2_processor = None


def _load_indictrans2():
    global _indictrans2_model, _indictrans2_tokenizer, _indictrans2_processor
    if _indictrans2_model is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    from config import HF_TOKEN

    _indictrans2_tokenizer = AutoTokenizer.from_pretrained(
        _INDICTRANS2_MODEL, trust_remote_code=True, token=HF_TOKEN
    )
    _indictrans2_model = AutoModelForSeq2SeqLM.from_pretrained(
        _INDICTRANS2_MODEL, trust_remote_code=True, token=HF_TOKEN
    )
    _indictrans2_model.eval()
    _indictrans2_processor = IndicProcessor(inference=True)


def _translate_indictrans2(texts: list[str], target_lang: str) -> list[str]:
    import torch

    _load_indictrans2()
    tgt_code = _INDICTRANS2_LANG_CODES[target_lang]

    batch = _indictrans2_processor.preprocess_batch(
        texts, src_lang=_SRC_LANG_CODE, tgt_lang=tgt_code
    )
    inputs = _indictrans2_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")

    with torch.no_grad():
        # use_cache=False works around a bug in IndicTrans2's own
        # modeling_indictrans.py: its custom forward() assumes
        # past_key_values is either None or a legacy tuple, but current
        # transformers passes an empty Cache object by default, which
        # crashes with `AttributeError: 'NoneType' object has no attribute
        # 'shape'` deep in its cross-attention layer. Disabling the KV
        # cache sidesteps the broken code path entirely (slower, but this
        # is a 200M model on short lines — the cost is negligible).
        generated = _indictrans2_model.generate(
            **inputs, use_cache=False, min_length=0, max_length=256,
            num_beams=5, num_return_sequences=1,
        )

    decoded = _indictrans2_tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return _indictrans2_processor.postprocess_batch(decoded, lang=tgt_code)


def _simple_postprocess(text: str) -> str:
    """
    Stand-in for IndicProcessor.postprocess_batch() when postprocessing
    several beam candidates for the SAME sentence in one process — that
    triggers a real hang somewhere inside IndicTransToolkit/indic_nlp_library
    (confirmed via isolated testing: reproducible regardless of which
    candidate text goes first, and unrelated to any specific character
    content). Comparing raw decoder output against the one candidate that
    *did* postprocess successfully showed the only actual difference was a
    stray space before sentence-final punctuation — so that's the one thing
    replicated here, in place of calling the buggy function at all.
    """
    text = re.sub(r"\s+([।,.!?])", r"\1", text)
    return text.strip()


def translate_indictrans2_candidates(text: str, target_lang: str, num_candidates: int = 6) -> list[str]:
    """
    Returns several distinct valid translations of ONE line (via beam search
    diversity), instead of just the single best one. Used for duration-aware
    selection: different candidates are natural Hindi phrasings of the same
    English line but can differ meaningfully in length, so a later step can
    pick whichever comes closest to a target speaking duration.

    IMPORTANT: candidates are NOT already quality-ranked by fitness for our
    purposes — beam search still returns them in the model's own confidence
    order, and lower-confidence candidates can contain real translation
    errors (verified empirically: one candidate mistranslated "$3 million"
    as "30 million" -- a real 10x numeric error, not a stylistic variant).
    Callers MUST quality-filter (e.g. with CometKiwi) before picking by
    length -- never select on length alone.
    """
    import torch

    _load_indictrans2()
    tgt_code = _INDICTRANS2_LANG_CODES[target_lang]

    batch = _indictrans2_processor.preprocess_batch([text], src_lang=_SRC_LANG_CODE, tgt_lang=tgt_code)
    inputs = _indictrans2_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")

    with torch.no_grad():
        generated = _indictrans2_model.generate(
            **inputs, use_cache=False, min_length=0, max_length=256,
            num_beams=max(num_candidates, 5), num_return_sequences=num_candidates,
        )

    decoded = _indictrans2_tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Dedupe while preserving beam-confidence order -- beams frequently
    # collapse to near-identical or fully identical strings.
    seen = set()
    candidates = []
    for d in decoded:
        c = _simple_postprocess(d)
        if c not in seen:
            seen.add(c)
            candidates.append(c)
    return candidates


def translate_one(text: str, target_lang: str) -> str:
    """
    Single-string translation, for callers that need to translate one
    candidate at a time -- e.g. components/duration_fit.py's retry loop,
    which retranslates a freshly condensed English variant on each attempt
    and can't batch those calls (each depends on the previous attempt's
    measured duration).
    """
    text = text.strip()
    if not text:
        return text
    if target_lang in _INDICTRANS2_LANG_CODES:
        return _translate_indictrans2([text], target_lang)[0]
    from components.translation_ollama_fallback import translate_lines
    from data_types import Segment
    seg = Segment(start=0.0, end=0.0, text=text)
    return translate_lines([seg], [(0, seg)], target_lang)[0] or text


def translate_segments(segments: list[Segment], target_lang: str) -> list[Segment]:
    """
    Returns a NEW list of segments with .text replaced by the translation —
    does not mutate the input, so callers can keep the original English text
    around for evaluation (CometKiwi needs the (source, translation) pair).
    """
    non_empty = [(i, seg) for i, seg in enumerate(segments) if seg.text.strip()]
    if not non_empty:
        return list(segments)

    if target_lang in _INDICTRANS2_LANG_CODES:
        texts = [seg.text.strip() for _, seg in non_empty]
        translations = _translate_indictrans2(texts, target_lang)
    else:
        from components.translation_ollama_fallback import translate_lines
        translations = translate_lines(segments, non_empty, target_lang)

    results = list(segments)
    for (i, seg), translated in zip(non_empty, translations):
        results[i] = replace(seg, text=translated or seg.text)
    return results
