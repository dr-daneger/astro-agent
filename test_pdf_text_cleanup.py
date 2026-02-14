"""Tests for pdf_text_cleanup module."""

import textwrap
from pdf_text_cleanup import (
    clean_broken_words, clean_extra_whitespace, is_word,
    normalize_ligatures,
)


# ---- Dictionary sanity check ----

def test_dictionary_loads():
    assert is_word("first")
    assert is_word("difficult")
    assert is_word("the")
    assert not is_word("xyzzyplugh")


# ---- Ligature-style broken words (short-left fragment) ----

def test_fi_ligature_break():
    assert clean_broken_words("fi rst") == "first"


def test_fl_ligature_break():
    assert clean_broken_words("fl ow") == "flow"


def test_ff_ligature_break():
    assert clean_broken_words("e ff ect") == "effect"


def test_fi_in_sentence():
    text = "The fi rst thing we need to fi nd is the fi le."
    cleaned = clean_broken_words(text)
    assert "first" in cleaned
    assert "find" in cleaned
    assert "file" in cleaned
    assert "fi rst" not in cleaned
    assert "fi nd" not in cleaned
    assert "fi le" not in cleaned


def test_ffi_ligature_triple():
    """Triple-fragment merge: 'di ffi cult' → 'difficult'."""
    result = clean_broken_words("This is di ffi cult to read.")
    assert "difficult" in result


def test_ffl_ligature():
    result = clean_broken_words("The ba ffl ing puzzle")
    assert "baffling" in result


# ---- Short-right fragment ----

def test_short_right_fragment():
    """e.g. 'wor ds' → 'words'"""
    result = clean_broken_words("These wor ds are broken")
    assert "words" in result


# ---- Should NOT merge real word pairs ----

def test_preserves_real_words_a_way():
    """'a way' should stay as 'a way', not become 'away'."""
    # Both 'a' and 'way' are real words, so no merge
    result = clean_broken_words("There is a way to do this.")
    assert "a way" in result


def test_preserves_real_words_an_other():
    """'an other' — both are real words, should not merge."""
    result = clean_broken_words("an other option")
    assert "an other" in result


def test_preserves_i_do():
    result = clean_broken_words("I do not agree.")
    assert "I do" in result


# ---- Whitespace cleanup ----

def test_collapse_multiple_spaces():
    result = clean_extra_whitespace("hello    world")
    assert result == "hello world"


def test_collapse_excessive_newlines():
    result = clean_extra_whitespace("para1\n\n\n\n\npara2")
    assert result == "para1\n\npara2"


def test_trim_trailing_spaces():
    result = clean_extra_whitespace("hello   \nworld   \n")
    assert result == "hello\nworld"


# ---- Full pipeline on realistic text block ----

def test_realistic_paragraph():
    broken = textwrap.dedent("""\
        The fi rst chapter of this book discusses the di ffi cult
        problem of fi nding e ffi cient solutions. It is important
        to understand the fl ow of information and how it a ff ects
        the fi nal outcome.""")

    cleaned = clean_broken_words(broken)

    assert "first" in cleaned
    assert "difficult" in cleaned
    assert "finding" in cleaned
    assert "efficient" in cleaned
    assert "flow" in cleaned
    assert "affects" in cleaned
    assert "final" in cleaned

    # None of the broken fragments should remain
    assert "fi rst" not in cleaned
    assert "di ffi cult" not in cleaned
    assert "fi nding" not in cleaned


# ---- Unicode ligature normalization ----

def test_normalize_fi_ligature():
    """ﬁ (U+FB01) should be replaced with 'fi'."""
    assert normalize_ligatures("pro\ufb01table") == "profitable"


def test_normalize_fl_ligature():
    """ﬂ (U+FB02) should be replaced with 'fl'."""
    assert normalize_ligatures("\ufb02ow") == "flow"


def test_normalize_ff_ligature():
    assert normalize_ligatures("e\ufb00ect") == "effect"


def test_normalize_ffi_ligature():
    assert normalize_ligatures("di\ufb03cult") == "difficult"


def test_normalize_preserves_normal_text():
    text = "This is normal text without ligatures."
    assert normalize_ligatures(text) == text


# ---- Ligature-aware merge (longer fragments) ----

def test_ligature_merge_profitable():
    """'profi table' — both fragments are >3 chars but it's a ligature break."""
    result = clean_broken_words("This was profi table for everyone.")
    assert "profitable" in result
    assert "profi table" not in result


def test_ligature_merge_field():
    """'fi eld' — 'fi' is technically a word but here it's a ligature."""
    result = clean_broken_words("cutting edge of their fi eld")
    assert "field" in result


def test_ligature_merge_figures():
    result = clean_broken_words("List of fi gures vii")
    assert "figures" in result


def test_ligature_merge_benefited():
    result = clean_broken_words("He benefi ted from the change.")
    assert "benefited" in result


def test_ligature_merge_with_punctuation():
    """'profi table.' should become 'profitable.'"""
    result = clean_broken_words("This was profi table.")
    assert "profitable." in result


# ---- Allen 2009 style text ----

def test_allen2009_paragraph():
    """Text modeled on actual output from Allen 2009 frontmatter PDF."""
    broken = (
        "were uniquely profi table to invent and use in Britain. "
        "The high wage economy of pre-industrial Britain also "
        "fostered industrial development. He benefi ted enormously "
        "from the cutting edge of their fi eld with an ability to "
        "write clearly. List of fi gures vii."
    )
    cleaned = clean_broken_words(broken)
    assert "profitable" in cleaned
    assert "benefited" in cleaned
    assert "field" in cleaned
    assert "figures" in cleaned
    assert "profi table" not in cleaned
    assert "fi eld" not in cleaned
    assert "fi gures" not in cleaned


# ---- Edge cases ----

def test_empty_string():
    assert clean_broken_words("") == ""


def test_no_broken_words():
    text = "This is a perfectly normal sentence."
    assert clean_broken_words(text) == text


def test_single_character_word():
    """Single chars like 'I' or 'a' followed by real words should be preserved."""
    result = clean_broken_words("I am a person.")
    assert "I am" in result
    assert "a person" in result


if __name__ == "__main__":
    import sys
    # Run tests manually if pytest isn't available
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
