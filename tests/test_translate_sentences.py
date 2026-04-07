"""
tests/test_translate_sentences.py
===================================
Tests for _group_into_sentences and translate_chunk_sentences.
No Ollama needed — the Ollama client is mocked.

Run:
    uv run --with pytest,pysrt pytest tests/test_translate_sentences.py -v
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from translate_utils import (
    _group_into_sentences,
    translate_chunk_sentences,
    _translate_sentences_with_retry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Time:
    """Minimal stand-in for pysrt time objects (ordinal = ms since start)."""
    def __init__(self, ordinal_ms: int):
        self.ordinal = ordinal_ms


class Sub:
    """Minimal stand-in for pysrt.SubRipItem with timing info."""
    def __init__(self, index: int, text: str,
                 start_ms: int = 0, end_ms: int = 2000):
        self.index = index
        self.text  = text
        self.start = _Time(start_ms)
        self.end   = _Time(end_ms)


def _client(response: str) -> MagicMock:
    c = MagicMock()
    c.generate.return_value = {"response": response}
    return c


def _build_response(subs: list, fmt: str = "[{i}] {text}") -> str:
    lines = []
    for s in subs:
        lines.append(fmt.format(i=s.index, text=f"FR_{s.index}"))
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# _group_into_sentences
# ═════════════════════════════════════════════════════════════════════════════

class TestGroupIntoSentences:

    def test_single_sentence_one_sub(self):
        subs = [Sub(1, "Hello world.")]
        groups = _group_into_sentences(subs)
        assert len(groups) == 1
        assert groups[0] == subs

    def test_single_sentence_two_subs(self):
        subs = [Sub(1, "Hello"), Sub(2, "world.")]
        groups = _group_into_sentences(subs)
        assert len(groups) == 1
        assert groups[0] == subs

    def test_two_sentences(self):
        subs = [
            Sub(1, "Hello world."),
            Sub(2, "How are you?"),
        ]
        groups = _group_into_sentences(subs)
        assert len(groups) == 2
        assert groups[0] == [subs[0]]
        assert groups[1] == [subs[1]]

    def test_two_subs_per_sentence(self):
        subs = [
            Sub(1, "The cat sat"),
            Sub(2, "on the mat."),
            Sub(3, "The dog ran"),
            Sub(4, "down the road."),
        ]
        groups = _group_into_sentences(subs)
        assert len(groups) == 2
        assert [s.index for s in groups[0]] == [1, 2]
        assert [s.index for s in groups[1]] == [3, 4]

    def test_question_mark_closes_group(self):
        subs = [Sub(1, "Is this correct?"), Sub(2, "Yes it is.")]
        groups = _group_into_sentences(subs)
        assert len(groups) == 2

    def test_exclamation_closes_group(self):
        subs = [Sub(1, "Look out!"), Sub(2, "Too late.")]
        groups = _group_into_sentences(subs)
        assert len(groups) == 2

    def test_no_terminal_punctuation_one_group(self):
        """Fragments without terminal punctuation form one open group."""
        subs = [Sub(1, "Hello"), Sub(2, "world"), Sub(3, "how")]
        groups = _group_into_sentences(subs)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_speaker_tag_stripped_before_check(self):
        """Speaker tag should not affect punctuation detection."""
        subs = [
            Sub(1, "[Speaker 1] Hello there,"),
            Sub(2, "[Speaker 1] how are you?"),
        ]
        groups = _group_into_sentences(subs)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_trailing_open_group(self):
        """Subs after last terminal punct become a final open group."""
        subs = [
            Sub(1, "First sentence."),
            Sub(2, "Incomplete fragment"),
            Sub(3, "still going"),
        ]
        groups = _group_into_sentences(subs)
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 2

    def test_empty_text_no_crash(self):
        subs = [Sub(1, ""), Sub(2, "Normal text.")]
        groups = _group_into_sentences(subs)
        # empty text neither closes nor corrupts — both subs land somewhere
        total = sum(len(g) for g in groups)
        assert total == 2

    def test_mixed_sentence_ends(self):
        subs = [
            Sub(1, "One."),
            Sub(2, "Two"),
            Sub(3, "three?"),
            Sub(4, "Four"),
            Sub(5, "five!"),
        ]
        groups = _group_into_sentences(subs)
        assert len(groups) == 3
        assert [s.index for s in groups[0]] == [1]
        assert [s.index for s in groups[1]] == [2, 3]
        assert [s.index for s in groups[2]] == [4, 5]

    def test_newline_in_text_normalised(self):
        """Newline characters don't confuse the terminal-punct check."""
        subs = [Sub(1, "Line one\nline two."), Sub(2, "Continues here.")]
        groups = _group_into_sentences(subs)
        assert len(groups) == 2


# ═════════════════════════════════════════════════════════════════════════════
# translate_chunk_sentences
# ═════════════════════════════════════════════════════════════════════════════

class TestTranslateChunkSentences:

    def test_single_sub_calls_client(self):
        subs = [Sub(1, "Hello world.", start_ms=0, end_ms=2000)]
        client = _client("[1] Bonjour monde.")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert 1 in result
        assert result[1] == "Bonjour monde."
        client.generate.assert_called_once()

    def test_duration_mentioned_in_prompt(self):
        """Single sub spanning 3s — prompt must mention 3.0 seconds."""
        subs = [Sub(1, "Hello.", start_ms=0, end_ms=3000)]
        client = _client("[1] Bonjour.")
        translate_chunk_sentences(subs, "en", "fr", client)
        prompt = client.generate.call_args[1]["prompt"]
        assert "3.0" in prompt

    def test_multi_sub_sentence_duration(self):
        """Two subs forming one sentence — duration = end_of_last - start_of_first."""
        subs = [
            Sub(1, "The cat sat", start_ms=0,    end_ms=1500),
            Sub(2, "on the mat.", start_ms=1600,  end_ms=3000),
        ]
        # duration = (3000 - 0) / 1000 = 3.0s
        client = _client("[1] Le chat\n[2] était assis.")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert set(result.keys()) == {1, 2}
        # Only one generate call (one sentence group)
        assert client.generate.call_count == 1
        prompt = client.generate.call_args[1]["prompt"]
        assert "3.0" in prompt

    def test_two_sentences_two_llm_calls(self):
        """Two complete sentences → two separate generate() calls."""
        subs = [
            Sub(1, "First sentence.", start_ms=0,    end_ms=2000),
            Sub(2, "Second sentence.", start_ms=2500, end_ms=4500),
        ]
        responses = ["[1] Première phrase.", "[2] Deuxième phrase."]
        client = MagicMock()
        client.generate.side_effect = [{"response": r} for r in responses]
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert client.generate.call_count == 2
        assert set(result.keys()) == {1, 2}

    def test_speaker_tag_preserved(self):
        subs = [Sub(1, "[Speaker 1] Hello world.", start_ms=0, end_ms=2000)]
        client = _client("[1] Bonjour monde.")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert result[1] == "[Speaker 1] Bonjour monde."

    def test_speaker_tag_not_in_prompt_text(self):
        """Speaker tag stripped from text sent to LLM (rule 5 in prompt)."""
        subs = [Sub(1, "[Speaker 2] How are you?", start_ms=0, end_ms=2000)]
        client = _client("[1] Comment allez-vous ?")
        translate_chunk_sentences(subs, "en", "fr", client)
        prompt = client.generate.call_args[1]["prompt"]
        # The indexed line should NOT contain the speaker tag in the text body
        assert "[Speaker 2] How are you?" not in prompt
        assert "How are you?" in prompt

    def test_returns_empty_on_llm_exception(self):
        subs = [Sub(1, "Hello.", start_ms=0, end_ms=2000)]
        client = MagicMock()
        client.generate.side_effect = RuntimeError("connection refused")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert result == {}

    def test_partial_parse_returns_available(self):
        """If model skips index 2, only index 1 is returned."""
        subs = [
            Sub(1, "Part one."),
            Sub(2, "Part two that was skipped."),
        ]
        client = _client("[1] Partie un.")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert 1 in result
        assert 2 not in result

    def test_all_index_formats_parsed(self):
        """_LINE_RE supports multiple index formats — tested via sentence path too."""
        subs = [Sub(10, "Hello world.", start_ms=0, end_ms=2000)]
        for resp in ["[10] Bonjour.", "10. Bonjour.", "10: Bonjour.", "(10) Bonjour."]:
            client = _client(resp)
            result = translate_chunk_sentences(subs, "en", "fr", client)
            assert 10 in result, f"Failed to parse: {resp!r}"

    def test_pipe_join_in_response_expanded(self):
        """' | ' in model response is expanded back to newline."""
        subs = [Sub(1, "Line one.", start_ms=0, end_ms=2000)]
        client = _client("[1] Ligne un | suite.")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert "\n" in result[1]

    def test_open_group_still_translated(self):
        """Fragments without terminal punct form an open group — still sent to LLM."""
        subs = [Sub(1, "No punct here", start_ms=0, end_ms=2000)]
        client = _client("[1] Pas de ponctuation ici")
        result = translate_chunk_sentences(subs, "en", "fr", client)
        assert 1 in result


# ═════════════════════════════════════════════════════════════════════════════
# _translate_sentences_with_retry
# ═════════════════════════════════════════════════════════════════════════════

class TestTranslateSentencesWithRetry:

    def test_success_first_attempt(self):
        subs = [Sub(1, "Hello.", start_ms=0, end_ms=2000)]
        client = _client("[1] Bonjour.")
        result = _translate_sentences_with_retry(subs, "en", "fr", client)
        assert result == {1: "Bonjour."}
        assert client.generate.call_count == 1

    def test_retries_missing_indices(self):
        """First call misses index 2; second call fills it in."""
        subs = [
            Sub(1, "Part one."),
            Sub(2, "Part two."),
        ]
        responses = [
            {"response": "[1] Partie un."},       # misses index 2
            {"response": "[2] Partie deux."},      # fills index 2
        ]
        client = MagicMock()
        client.generate.side_effect = responses

        with patch("translate_utils.time.sleep"):
            result = _translate_sentences_with_retry(subs, "en", "fr", client, retries=3)

        assert result == {1: "Partie un.", 2: "Partie deux."}

    def test_fallback_to_fragment_mode(self):
        """After all sentence-mode retries fail, falls back to translate_chunk for missing."""
        subs = [
            Sub(1, "Sentence one."),
            Sub(2, "Sentence two."),
        ]
        # sentence-mode always returns only index 1
        # fragment fallback returns index 2
        sentence_resp = {"response": "[1] Phrase un."}
        fragment_resp = {"response": "[2] Phrase deux."}

        client = MagicMock()
        client.generate.side_effect = [
            sentence_resp,  # attempt 1 (sentence mode)
            sentence_resp,  # attempt 2 (sentence mode, missing=[2])
            sentence_resp,  # attempt 3 (sentence mode, missing=[2])
            fragment_resp,  # fallback translate_chunk for [2]
            fragment_resp,  # fallback retry
        ]

        with patch("translate_utils.time.sleep"):
            result = _translate_sentences_with_retry(subs, "en", "fr", client, retries=3)

        assert 1 in result
        assert 2 in result

    def test_all_missing_returns_empty_best_effort(self):
        """If everything fails, returns whatever partial result was obtained."""
        subs = [Sub(1, "Test.", start_ms=0, end_ms=1000)]
        client = MagicMock()
        client.generate.return_value = {"response": "garbage output no index"}

        with patch("translate_utils.time.sleep"):
            result = _translate_sentences_with_retry(subs, "en", "fr", client, retries=2)

        # Should not raise — returns empty dict as best effort
        assert isinstance(result, dict)
