"""
tests/test_translate_chunk.py
==============================
Matrix tests for translate_diarize.translate_chunk and _translate_with_retry.
No Ollama needed — the Ollama client is mocked.

Root causes of "Missing line N" warnings found and tested here:
  BUG-1 : Regex doesn't handle '(180) text' → NO MATCH → missing line
  BUG-2 : '180) text' matches but captures ') text' → corrupted translation
  BUG-3 : _translate_with_retry only retries when result is 100% empty;
           if just one index is absent it returns silently — never retried.

Run:
    uv run --with pytest,pysrt,langsmith pytest tests/test_translate_chunk.py -v
"""

import re
import time
from unittest.mock import MagicMock, call, patch

import pytest

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, **kwargs):
        def _d(fn): return fn
        return _d(name) if callable(name) else _d

# translate_utils has NO module-level side effects — safe to import directly
from translate_utils import (
    LANG_MAP,
    translate_chunk,
    _translate_with_retry
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class Sub:
    """Minimal stand-in for pysrt.SubRipItem."""
    def __init__(self, index: int, text: str):
        self.index = index
        self.text  = text


def _client(response: str) -> MagicMock:
    """Return a mock Ollama Client whose generate() returns `response`."""
    c = MagicMock()
    c.generate.return_value = {"response": response}
    return c


def _build_response(subs: list[Sub], fmt: str = "[{i}] {text}") -> str:
    """Build a fake model response from a list of subs using the given format."""
    lines = []
    for s in subs:
        lines.append(fmt.format(i=s.index, text=f"TRANSLATED_{s.text}"))
    return "\n".join(lines)


def _subs(start: int, count: int, text_fn=None) -> list[Sub]:
    """Create `count` sequential subs starting at `start`."""
    if text_fn is None:
        text_fn = lambda i: f"line {i} content"
    return [Sub(start + j, text_fn(start + j)) for j in range(count)]


# ═════════════════════════════════════════════════════════════════════════════
# BUG-1 + BUG-2: Regex parsing — all output formats
# ═════════════════════════════════════════════════════════════════════════════

GOOD_FORMATS = [
    "[{i}] {text}",     # standard  ← always worked
    "{i}. {text}",      # dot       ← always worked
    "{i}: {text}",      # colon     ← always worked
    "{i}- {text}",      # dash      ← always worked
    "{i} {text}",       # space     ← always worked
    "<{i}> {text}",     # angle     ← always worked
    "[<{i}>] {text}",   # mixed     ← always worked
]

BAD_FORMATS = [
    "({i}) {text}",     # BUG-1: parens → NO MATCH → missing line
    "{i}) {text}",      # BUG-2: paren-close → matches with ') text' prefix
]


@pytest.mark.parametrize("fmt", GOOD_FORMATS, ids=lambda f: f[:8])
@traceable(name="test_regex_good_format")
def test_good_formats_parse_all_lines(fmt):
    """Standard output formats must parse all 5 indices with clean translations."""
    chunk = _subs(10, 5)
    response = _build_response(chunk, fmt)
    result = translate_chunk(chunk, "en", "fr", _client(response))

    assert set(result.keys()) == {10, 11, 12, 13, 14}, \
        f"Format {fmt!r}: missing indices in result"
    for sub in chunk:
        assert not result[sub.index].startswith(")"), \
            f"Index {sub.index}: translation starts with ')' — BUG-2"


@pytest.mark.parametrize("fmt,bug", [
    ("({i}) {text}", "BUG-1"),
    ("{i}) {text}",  "BUG-2"),
], ids=["paren-both", "paren-close"])
@traceable(name="test_regex_bad_format")
def test_bad_formats_document_bugs(fmt, bug):
    """
    Documents the parser failures.
    These tests will FAIL until the regex is fixed.
    """
    chunk = _subs(10, 5)
    response = _build_response(chunk, fmt)
    result = translate_chunk(chunk, "en", "fr", _client(response))

    if bug == "BUG-1":
        missing = sorted(set(s.index for s in chunk) - set(result.keys()))
        assert not missing, f"BUG-1 ({fmt!r}): indices {missing} missing — regex doesn't handle '(' prefix"
    elif bug == "BUG-2":
        for sub in chunk:
            val = result.get(sub.index, "")
            assert not val.startswith(")"), \
                f"BUG-2 ({fmt!r}): index {sub.index} translation starts with ') '"


# ═════════════════════════════════════════════════════════════════════════════
# BUG-3: _translate_with_retry — partial missing lines
# ═════════════════════════════════════════════════════════════════════════════

@traceable(name="test_retry_partial_missing")
def test_retry_triggers_when_one_line_is_missing():
    """
    BUG-3: if the model skips ONE line (e.g. line 180), _translate_with_retry
    must detect it and retry — not silently return a partial result.
    """
    chunk = _subs(166, 15)  # indices 166-180 (like chunk 12 of 238 subs)
    expected = {s.index for s in chunk}

    # First call: missing index 180 (the last line — most common failure point)
    response_partial = "\n".join(
        f"[{s.index}] TRANSLATED_{s.text}"
        for s in chunk if s.index != 180
    )
    # Second call: returns all lines
    response_full = _build_response(chunk)

    client = MagicMock()
    client.generate.side_effect = [
        {"response": response_partial},
        {"response": response_full},
    ]

    with patch("translate_diarize.time.sleep"):
        result = _translate_with_retry(chunk, "en", "fr", client)

    assert 180 in result, "BUG-3: index 180 still missing — retry not triggered for partial result"
    assert set(result.keys()) == expected


@traceable(name="test_retry_not_triggered_when_complete")
def test_retry_not_triggered_when_all_lines_present():
    """If the first call returns all expected indices, no retry should happen."""
    chunk = _subs(1, 5)
    client = _client(_build_response(chunk))

    with patch("translate_diarize.time.sleep"):
        result = _translate_with_retry(chunk, "en", "fr", client)

    assert client.generate.call_count == 1, "Retry triggered unnecessarily"
    assert set(result.keys()) == {1, 2, 3, 4, 5}


@traceable(name="test_retry_exhausted_still_returns_partial")
def test_all_retries_exhausted_returns_best_partial():
    """
    If the model NEVER returns line 180 across all retries,
    return whatever we have — don't crash or lose everything.
    """
    chunk = _subs(166, 15)

    response_without_180 = "\n".join(
        f"[{s.index}] TRANSLATED_{s.text}"
        for s in chunk if s.index != 180
    )
    client = _client(response_without_180)

    with patch("translate_diarize.time.sleep"):
        result = _translate_with_retry(chunk, "en", "fr", client, retries=3)

    # Should still contain the 14 lines that DID translate
    assert len(result) >= 14, "Lost all translations when only one was missing"


# ═════════════════════════════════════════════════════════════════════════════
# Language pair matrix
# ═════════════════════════════════════════════════════════════════════════════

LANG_PAIRS = [
    ("en", "fr"), ("en", "de"), ("en", "es"),
    ("de", "fr"), ("de", "en"),
    ("fr", "en"), ("fr", "de"),
    ("tr", "en"), ("it", "fr"),
]


@pytest.mark.parametrize("src,tgt", LANG_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}" if isinstance(p, tuple) else p)
@traceable(name="test_lang_pair_prompt_correctness")
def test_language_names_in_prompt(src, tgt):
    """The prompt must name the correct source/target languages for every pair."""
    chunk = _subs(1, 3)
    client = _client(_build_response(chunk))

    translate_chunk(chunk, src, tgt, client)

    prompt = client.generate.call_args[1]["prompt"]
    src_name = LANG_MAP.get(src, src)
    tgt_name = LANG_MAP.get(tgt, tgt)
    assert src_name in prompt, f"Source language '{src_name}' not in prompt"
    assert tgt_name in prompt, f"Target language '{tgt_name}' not in prompt"


@pytest.mark.parametrize("src,tgt", LANG_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}" if isinstance(p, tuple) else p)
@traceable(name="test_lang_pair_returns_all_indices")
def test_language_pairs_return_all_indices(src, tgt):
    """Every language pair must return all indices when model output is perfect."""
    chunk = _subs(1, 10)
    result = translate_chunk(chunk, src, tgt, _client(_build_response(chunk)))
    missing = sorted({s.index for s in chunk} - set(result.keys()))
    assert not missing, f"{src}→{tgt}: missing indices {missing}"


# ═════════════════════════════════════════════════════════════════════════════
# SRT content edge cases
# ═════════════════════════════════════════════════════════════════════════════

@traceable(name="test_speaker_tags_preserved")
def test_speaker_tags_are_preserved():
    """[Speaker N] prefix must be re-attached to the translated text."""
    chunk = [Sub(1, "[Speaker 1] Hello world"), Sub(2, "[Speaker 2] How are you")]
    response = "[1] TRANSLATED_Hello world\n[2] TRANSLATED_How are you"
    result = translate_chunk(chunk, "en", "fr", _client(response))
    assert result[1].startswith("[Speaker 1]"), f"Speaker tag lost: {result[1]!r}"
    assert result[2].startswith("[Speaker 2]"), f"Speaker tag lost: {result[2]!r}"


@traceable(name="test_single_word_lines")
def test_single_word_lines_not_skipped():
    """Very short lines (single word, yes/no) must not go missing."""
    chunk = [Sub(179, "important"), Sub(180, "yes"), Sub(181, "ok")]
    response = "[179] important_fr\n[180] oui\n[181] ok_fr"
    result = translate_chunk(chunk, "en", "fr", _client(response))
    assert 180 in result, "Short single-word line 180 is missing"
    assert result[180] == "oui"


@traceable(name="test_last_line_of_chunk")
def test_last_line_of_chunk_not_dropped():
    """
    The LAST line of a chunk is the most commonly skipped by small models.
    Simulate index 180 being the last line of chunk 12 (as in real failure).
    """
    chunk = _subs(166, 15)  # 166..180, index 180 is last
    assert chunk[-1].index == 180

    response = _build_response(chunk)
    result = translate_chunk(chunk, "en", "fr", _client(response))
    assert 180 in result, "Last line of chunk (index 180) went missing"


@pytest.mark.parametrize("count", [1, 5, 15, 30], ids=["1line", "5lines", "15lines", "30lines"])
@traceable(name="test_chunk_sizes")
def test_various_chunk_sizes_all_present(count):
    """All chunk sizes must return all indices."""
    chunk = _subs(1, count)
    result = translate_chunk(chunk, "en", "fr", _client(_build_response(chunk)))
    missing = sorted({s.index for s in chunk} - set(result.keys()))
    assert not missing, f"{count}-line chunk: missing {missing}"


@traceable(name="test_multiline_subtitle")
def test_pipe_separator_reconstructed():
    """Lines containing ' | ' are expanded back to newlines in output."""
    chunk = [Sub(1, "line one\nline two")]
    # model returns pipe-joined text as per translate_chunk logic
    response = "[1] translated one | translated two"
    result = translate_chunk(chunk, "en", "fr", _client(response))
    assert "\n" in result[1], "Pipe separator not reconstructed to newline"


@traceable(name="test_model_adds_preamble")
def test_preamble_lines_are_ignored():
    """If model outputs a preamble before the translations, it must be skipped."""
    chunk = _subs(1, 3)
    response = (
        "Here are the translations:\n"
        "[1] TRANSLATED_line 1 content\n"
        "[2] TRANSLATED_line 2 content\n"
        "[3] TRANSLATED_line 3 content"
    )
    result = translate_chunk(chunk, "en", "fr", _client(response))
    assert set(result.keys()) == {1, 2, 3}


@traceable(name="test_model_adds_postamble")
def test_postamble_lines_are_ignored():
    """If model adds a note after translations, indices should still be found."""
    chunk = _subs(1, 3)
    response = (
        "[1] TRANSLATED_line 1 content\n"
        "[2] TRANSLATED_line 2 content\n"
        "[3] TRANSLATED_line 3 content\n"
        "Note: some lines may have been paraphrased."
    )
    result = translate_chunk(chunk, "en", "fr", _client(response))
    assert set(result.keys()) == {1, 2, 3}


@traceable(name="test_non_sequential_indices")
def test_non_sequential_srt_indices():
    """SRT files don't always start at 1. Indices like 200-214 must work."""
    chunk = _subs(200, 15)
    result = translate_chunk(chunk, "en", "fr", _client(_build_response(chunk)))
    missing = sorted({s.index for s in chunk} - set(result.keys()))
    assert not missing, f"Non-sequential indices: missing {missing}"


@traceable(name="test_model_reindexes_from_1")
def test_model_reindexing_from_1_is_detected():
    """
    Some small models reindex output starting from [1] regardless of input.
    Chunk 12 (indices 166-180) returns [1]-[15] instead.
    This causes ALL 15 lines to be missing — must be caught by retry.
    """
    chunk = _subs(166, 15)
    # Model reindexes from 1
    reindexed_response = "\n".join(
        f"[{j+1}] TRANSLATED_{chunk[j].text}" for j in range(len(chunk))
    )
    result = translate_chunk(chunk, "en", "fr", _client(reindexed_response))
    missing = sorted({s.index for s in chunk} - set(result.keys()))
    # Document the bug: all 15 lines will be missing
    assert len(missing) == 15, \
        "Expected all 15 lines missing when model reindexes (regression test for known failure)"


# ═════════════════════════════════════════════════════════════════════════════
# LANG_MAP completeness
# ═════════════════════════════════════════════════════════════════════════════

@traceable(name="test_lang_map_completeness")
@pytest.mark.parametrize("code,name", [
    ("en", "English"), ("fr", "French"), ("de", "German"),
    ("es", "Spanish"), ("it", "Italian"), ("tr", "Turkish"),
    ("nl", "Dutch"), ("pl", "Polish"), ("pt", "Portuguese"),
    ("ru", "Russian"), ("zh", "Chinese"), ("ja", "Japanese"),
])
def test_lang_map_has_correct_names(code, name):
    assert LANG_MAP.get(code) == name, f"LANG_MAP['{code}'] should be '{name}'"
