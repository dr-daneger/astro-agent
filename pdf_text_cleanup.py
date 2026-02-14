"""
PDF Text Extraction & Broken-Word Cleanup

Extracts text from PDFs and repairs words that were split by erroneous spaces
during extraction — a common artifact caused by ligature characters (fi, fl, ff,
ffi, ffl) and font-level character spacing in PDF rendering.

Example: "fi rst" → "first", "di ffi cult" → "difficult", "e ff ect" → "effect"

Usage:
    from pdf_text_cleanup import extract_and_clean_pdf

    cleaned_text = extract_and_clean_pdf("my_book.pdf")
    # Or step by step:
    raw_text = extract_text_from_pdf("my_book.pdf")
    cleaned_text = clean_broken_words(raw_text)
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------

_dictionary: set[str] | None = None

# Common English inflectional suffixes used to expand the base dictionary.
_INFLECTION_SUFFIXES = ["s", "es", "ed", "ing", "er", "est", "ly", "ness", "ment", "tion", "ation"]


def _expand_with_inflections(base_words: set[str]) -> set[str]:
    """
    Given a set of base-form words, generate common inflected forms.

    Many dictionary sources (like gcide) only include base forms, so words
    like "words", "files", "affects" are missing.  This adds them.
    """
    expanded = set(base_words)
    for word in base_words:
        if not word.isalpha() or len(word) < 3:
            continue
        for suffix in _INFLECTION_SUFFIXES:
            expanded.add(word + suffix)
        # Handle consonant doubling: run → running, fit → fitted
        if len(word) >= 3 and word[-1] not in "aeiouy" and word[-2] in "aeiou" and word[-3] not in "aeiou":
            doubled = word + word[-1]
            expanded.add(doubled + "ing")
            expanded.add(doubled + "ed")
            expanded.add(doubled + "er")
            expanded.add(doubled + "est")
        # Handle silent-e dropping: file → filing, make → making
        if word.endswith("e"):
            stem = word[:-1]
            expanded.add(stem + "ing")
            expanded.add(stem + "ed")
            expanded.add(stem + "er")
            expanded.add(stem + "est")
            expanded.add(stem + "ation")
    return expanded


def _load_dictionary() -> set[str]:
    """Load an English word set from the best available source."""
    words: set[str] = set()

    # 1. english-words package (installed as a dependency)
    try:
        from english_words import get_english_words_set
        words = get_english_words_set(["gcide", "web2"], lower=True)
        if words:
            log.info(f"Loaded base dictionary with {len(words)} words (english-words)")
    except Exception:
        pass

    # 2. NLTK corpus
    if not words:
        try:
            from nltk.corpus import words as nltk_words
            words = {w.lower() for w in nltk_words.words()}
            if words:
                log.info(f"Loaded base dictionary with {len(words)} words (nltk)")
        except Exception:
            pass

    # 3. System dictionary
    if not words:
        for dict_path in ["/usr/share/dict/words", "/usr/share/dict/american-english"]:
            try:
                with open(dict_path) as f:
                    words = {line.strip().lower() for line in f if line.strip()}
                if words:
                    log.info(f"Loaded base dictionary with {len(words)} words ({dict_path})")
                    break
            except FileNotFoundError:
                continue

    if not words:
        log.warning("No dictionary source found — broken-word repair will rely on heuristics only")
        return set()

    # Expand with inflected forms to cover plurals, verb forms, etc.
    expanded = _expand_with_inflections(words)
    log.info(f"Dictionary expanded from {len(words)} to {len(expanded)} words (with inflections)")
    return expanded


def get_dictionary() -> set[str]:
    """Return the cached dictionary, loading it on first call."""
    global _dictionary
    if _dictionary is None:
        _dictionary = _load_dictionary()
    return _dictionary


def is_word(token: str) -> bool:
    """Check whether *token* is a recognized English word."""
    return token.lower() in get_dictionary()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path, page_range: tuple[int, int] | None = None) -> str:
    """
    Extract raw text from a PDF file.

    Tries pdfplumber first (better layout handling), falls back to pypdf.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.
    page_range : tuple(start, end), optional
        0-indexed inclusive page range.  ``None`` means all pages.

    Returns
    -------
    str
        The concatenated raw text of the selected pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try pdfplumber first, fall back to pypdf
    try:
        import pdfplumber
        return _extract_with_pdfplumber(pdf_path, page_range)
    except ImportError:
        log.info("pdfplumber not available, using pypdf")
    except Exception as e:
        log.warning(f"pdfplumber failed ({e}), falling back to pypdf")

    return _extract_with_pypdf(pdf_path, page_range)


def _extract_with_pdfplumber(pdf_path: Path, page_range: tuple[int, int] | None) -> str:
    import pdfplumber
    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        start, end = 0, total - 1
        if page_range is not None:
            start, end = max(0, page_range[0]), min(total - 1, page_range[1])
        for idx in range(start, end + 1):
            text = pdf.pages[idx].extract_text()
            if text:
                pages_text.append(text)
    raw = "\n\n".join(pages_text)
    log.info(f"Extracted {len(raw)} chars from {pdf_path.name} pages {start}-{end} (pdfplumber)")
    return raw


def _extract_with_pypdf(pdf_path: Path, page_range: tuple[int, int] | None) -> str:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    start, end = 0, total - 1
    if page_range is not None:
        start, end = max(0, page_range[0]), min(total - 1, page_range[1])
    pages_text: list[str] = []
    for idx in range(start, end + 1):
        text = reader.pages[idx].extract_text()
        if text:
            pages_text.append(text)
    raw = "\n\n".join(pages_text)
    log.info(f"Extracted {len(raw)} chars from {pdf_path.name} pages {start}-{end} (pypdf)")
    return raw


# ---------------------------------------------------------------------------
# Unicode ligature normalization
# ---------------------------------------------------------------------------

# Mapping of Unicode ligature characters to their ASCII equivalents.
# PDF fonts frequently use these, and text extractors preserve them as
# single Unicode code points, often with a trailing space that breaks words.
_LIGATURE_MAP = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ  (long s + t)
    "\ufb06": "st",   # ﬆ
}

# Pre-compiled regex for all ligature characters.
_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))


def normalize_ligatures(text: str) -> str:
    """
    Replace Unicode ligature characters (U+FB00–FB06) with their ASCII
    equivalents and remove the erroneous space that PDF extractors often
    insert after them.

    Example: ``"proﬁ table"``  →  ``"profitable"``
    """
    def _replace(m: re.Match) -> str:
        return _LIGATURE_MAP[m.group(0)]

    # Step 1: replace the ligature char with ASCII letters
    text = _LIGATURE_RE.sub(_replace, text)
    # Step 2: after replacement the space that was "inside" the ligature break
    # may now sit between two letter runs.  We collapse it only when the space
    # is flanked by word characters on both sides (i.e. it was an intra-word
    # break, not a real word boundary).  We do this conservatively by checking
    # for known ligature-ending patterns: the replacement leaves us with e.g.
    # "profi table" — we rejoin via the broken-word scanner later, so we just
    # need the ligature chars normalized here.
    return text


# ---------------------------------------------------------------------------
# Broken-word repair  (token-based scanner)
# ---------------------------------------------------------------------------

# Maximum number of consecutive fragments to try merging.
_MAX_MERGE_WINDOW = 3


def _strip_punctuation(token: str) -> tuple[str, str, str]:
    """
    Split a token into (leading_punct, alpha_core, trailing_punct).

    Examples:
        "hello"  → ("", "hello", "")
        "le."    → ("", "le", ".")
        "(word)" → ("(", "word", ")")
        '"Hi!'   → ('"', "Hi", "!")
    """
    start = 0
    end = len(token)
    while start < end and not token[start].isalpha():
        start += 1
    while end > start and not token[end - 1].isalpha():
        end -= 1
    return token[:start], token[start:end], token[end:]


def _is_fragment(core: str) -> bool:
    """Return True if *core* is a non-empty alphabetic string (a word fragment candidate)."""
    return bool(core) and core.isalpha()


def _should_merge(fragments: list[str], max_short: int) -> bool:
    """
    Decide whether merging *fragments* is appropriate.

    Returns True when:
    - The concatenation is a recognized English word, AND
    - At least one fragment is short (≤ *max_short* chars), AND
    - NOT all fragments are independently valid words.
    """
    candidate = "".join(fragments)
    if not is_word(candidate):
        return False
    has_short = any(len(f) <= max_short for f in fragments)
    all_real = all(is_word(f) for f in fragments)
    return has_short and not all_real


# Ligature fragment endings — when a token ends with one of these and the
# concatenation with the next token is a real word, it's almost certainly a
# ligature break regardless of fragment length.
_LIGATURE_ENDINGS = ("fi", "fl", "ff", "ffi", "ffl", "ft", "ct", "st")

# Standalone ligature fragments that appear as the entire left token.
# These are technically dictionary words ("fi" = music note, "fl" = abbreviation)
# but when they appear before another fragment in PDF text, they're almost
# always a ligature break.  We merge more aggressively for these.
_LIGATURE_STANDALONE = {"fi", "fl", "ff", "ffi", "ffl", "st"}


def _has_ligature_boundary(core_a: str, core_b: str) -> bool:
    """
    Return True if the boundary between *core_a* and *core_b* looks like a
    ligature break.  This catches cases like "profi" + "table" where neither
    fragment is short enough for the general heuristic.
    """
    lower_a = core_a.lower()
    return any(lower_a.endswith(lig) for lig in _LIGATURE_ENDINGS)


def _is_ligature_fragment(core: str) -> bool:
    """
    Return True if *core* is exactly a known standalone ligature fragment.
    E.g. "fi", "fl", "ff".  These override the "both are real words" safety
    check because they almost never appear as actual words in PDF text.
    """
    return core.lower() in _LIGATURE_STANDALONE


def _clean_line(line: str) -> str:
    """
    Process a single line of text, merging broken-word fragments.

    Scans tokens left-to-right with a sliding window of up to 3 tokens.
    At each position it tries, in order:

    1. **3-token merge** — for triple ligature breaks ("di ffi cult").
    2. **2-token general merge** — when at least one fragment is short (≤3).
    3. **2-token ligature merge** — when token A ends with a ligature fragment
       (fi, fl, ff, …) regardless of length, and A+B is a dictionary word.

    Handles trailing punctuation (e.g. "fi le." → "file.").
    """
    tokens = line.split(" ")
    result: list[str] = []
    i = 0

    while i < len(tokens):
        merged = False

        # --- Try 3-token merge first (e.g. "di ffi cult" → "difficult") ---
        if i + 2 < len(tokens):
            pre_a, core_a, suf_a = _strip_punctuation(tokens[i])
            pre_b, core_b, suf_b = _strip_punctuation(tokens[i + 1])
            pre_c, core_c, suf_c = _strip_punctuation(tokens[i + 2])
            # Only the first token may have leading punct, middle must be clean,
            # last may have trailing punct.
            if (_is_fragment(core_a) and not suf_a
                    and _is_fragment(core_b) and not pre_b and not suf_b
                    and _is_fragment(core_c) and not pre_c):
                if _should_merge([core_a, core_b, core_c], max_short=4):
                    merged_word = core_a + core_b + core_c
                    log.debug(f"Triple merge: '{core_a} {core_b} {core_c}' → '{merged_word}'")
                    result.append(pre_a + merged_word + suf_c)
                    i += 3
                    merged = True

        # --- Try 2-token merge (e.g. "fi rst" → "first", "fi le." → "file.") ---
        if not merged and i + 1 < len(tokens):
            pre_a, core_a, suf_a = _strip_punctuation(tokens[i])
            pre_b, core_b, suf_b = _strip_punctuation(tokens[i + 1])
            # First token may have leading punct, second may have trailing punct.
            # No punct should appear between the two cores.
            if (_is_fragment(core_a) and not suf_a
                    and _is_fragment(core_b) and not pre_b):
                # General short-fragment merge
                if _should_merge([core_a, core_b], max_short=3):
                    merged_word = core_a + core_b
                    log.debug(f"Merge: '{core_a} {core_b}' → '{merged_word}'")
                    result.append(pre_a + merged_word + suf_b)
                    i += 2
                    merged = True
                # Ligature fragment override — "fi", "fl", "ff" etc. are
                # technically dictionary words but in PDF context they're
                # almost always broken ligatures.  Merge if A+B is a word.
                elif _is_ligature_fragment(core_a) and is_word(core_a + core_b):
                    merged_word = core_a + core_b
                    log.debug(f"Ligature fragment merge: '{core_a} {core_b}' → '{merged_word}'")
                    result.append(pre_a + merged_word + suf_b)
                    i += 2
                    merged = True
                # Ligature-aware merge — handles longer fragments like
                # "profi" + "table" where neither side is ≤3 chars
                elif _has_ligature_boundary(core_a, core_b):
                    candidate = core_a + core_b
                    if is_word(candidate) and not is_word(core_a):
                        log.debug(f"Ligature merge: '{core_a} {core_b}' → '{candidate}'")
                        result.append(pre_a + candidate + suf_b)
                        i += 2
                        merged = True

        if not merged:
            result.append(tokens[i])
            i += 1

    return " ".join(result)


def clean_broken_words(text: str) -> str:
    """
    Repair words that were split by erroneous spaces during PDF extraction.

    Uses a token-based sliding-window scanner that tries merging 2–3 adjacent
    fragments and validates the result against an English dictionary.

    A merge is only applied when:
    - At least one fragment is short (≤3–4 chars), AND
    - The fragments are NOT all valid words on their own, AND
    - Their concatenation IS a recognized English word.

    This avoids falsely merging legitimate short words like "a way" or "I do".
    """
    # First normalize Unicode ligatures (ﬁ → fi, ﬂ → fl, etc.)
    text = normalize_ligatures(text)

    dictionary = get_dictionary()
    if not dictionary:
        log.warning("Dictionary is empty — skipping broken-word repair")
        return text

    lines = text.split("\n")
    cleaned_lines = [_clean_line(line) for line in lines]
    return "\n".join(cleaned_lines)


def clean_extra_whitespace(text: str) -> str:
    """Normalize runs of whitespace: collapse multiple spaces, trim lines."""
    # Collapse multiple spaces (but not newlines) into one
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove trailing whitespace on each line
    text = re.sub(r" +\n", "\n", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def extract_and_clean_pdf(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
) -> str:
    """
    One-step PDF → cleaned text.

    Extracts text from *pdf_path*, repairs broken words, and normalizes
    whitespace — ready for an audiobook TTS reader.
    """
    raw = extract_text_from_pdf(pdf_path, page_range=page_range)
    cleaned = clean_broken_words(raw)
    cleaned = clean_extra_whitespace(cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and clean up broken words.",
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "-p", "--pages",
        help="Page range (0-indexed), e.g. '0-9' for first 10 pages",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    page_range = None
    if args.pages:
        parts = args.pages.split("-")
        page_range = (int(parts[0]), int(parts[1]))

    cleaned = extract_and_clean_pdf(args.pdf, page_range=page_range)

    if args.output:
        out = Path(args.output)
        out.write_text(cleaned, encoding="utf-8")
        log.info(f"Cleaned text written to {out}")
    else:
        sys.stdout.write(cleaned)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
