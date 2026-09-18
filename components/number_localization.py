"""
Converts digit sequences in English source text into spoken English number
words (e.g. "178,000" -> "one hundred and seventy-eight thousand"), kept in
English rather than translated -- validated by direct listening test to
sound natural in otherwise-Hindi speech (real Indian speech commonly says
numbers in English mid-sentence, e.g. "one seventy eight thousand", rather
than formal Hindi lakh/crore numbering).

Two real, separate problems this fixes, both found on a real podcast clip
about token counts/model sizes/percentages (eval-dataset content SenseVoice
had never been stress-tested on before):

1. IndicTrans2 just copies digit sequences straight through untranslated
   ("178,000" stays "178,000") -- it has no notion of localizing a numeral
   into spoken form, unlike ordinary vocabulary.
2. A naive fix (convert digits to English words, then translate the whole
   sentence) fails: IndicTrans2 translates the spelled-out number into
   Hindi words too, which is a valid but different choice (see
   scratchpad_reports/number_words_proof/ for that comparison) -- the goal
   here is specifically to KEEP the number in English.

The straightforward "protect this span from translation" placeholder trick
has a real gotcha, confirmed by direct testing: IndicTrans2 does not
reliably leave arbitrary letter-based placeholders alone -- it sometimes
phonetically transliterates them letter-by-letter into Devanagari (e.g.
"ZZNUM0ZZ" became "जेड. जेड. एन. यू. एम. 0. जेड. जेड.") or silently
changes their case, breaking substitution. Purely NUMERIC placeholders
don't have this problem, because raw digit runs are exactly what
IndicTrans2 already reliably copies through untouched (that's problem #1
above) -- so this uses that same behavior deliberately, instead of fighting
it.
"""
import re

from num2words import num2words

# The optional trailing bit (percent sign, or a bare "x" multiplier like
# "5x") is grouped WITH its leading whitespace so that whitespace is only
# ever consumed when that trailing bit is actually present -- an earlier
# version used separate \s* and %? quantifiers, which greedily ate the
# space after every plain number even with nothing following it (e.g.
# "178,000 GitHub" lost its space and became "...thousandGitHub").
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\s*%|\s*[xX]\b)?")


def protect_numbers(text: str) -> tuple[str, dict[str, str]]:
    """
    Replaces each digit sequence in `text` with a purely-numeric placeholder
    IndicTrans2 will pass through untouched, and returns the spoken English
    words it should be restored to after translation. Call restore_numbers()
    on the translated output with the returned map.
    """
    placeholders: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        raw = match.group(0)
        is_percent = raw.rstrip().endswith("%")
        is_multiplier = raw.rstrip()[-1:] in ("x", "X")
        num_str = re.sub(r"[,%xX\s]", "", raw)
        try:
            n = int(num_str)
        except ValueError:
            return raw
        words = num2words(n)
        if is_percent:
            words += " percent"
        elif is_multiplier:
            words += " x"
        key = f"90{len(placeholders)}09"
        placeholders[key] = words
        return key

    protected = _NUMBER_RE.sub(repl, text)
    return protected, placeholders


def restore_numbers(text: str, placeholders: dict[str, str]) -> str:
    """Substitutes each placeholder back with its English number words."""
    for key, words in placeholders.items():
        text = text.replace(key, words)
    return text
