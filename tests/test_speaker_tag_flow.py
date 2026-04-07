"""
tests/test_speaker_tag_flow.py — End-to-end speaker tag integrity
==================================================================

Integration test that traces speaker tags through every pipeline step:

  NeMo (_segs_to_srt) → translate (translate_chunk) → dub (parse_srt) → clean (clean_srt_files)

This catches the user's exact bug: "speaker tags not being removed properly"
and verifies the full contract at each boundary.

Run:
    uv run --with "pytest,pysrt" pytest tests/test_speaker_tag_flow.py -v
"""

import re
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pysrt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen3-tts"))

from nemo_audio import _segs_to_srt, _words_to_segs
from translate_utils import translate_chunk
from dub_srt import parse_srt


# ── Helpers ───────────────────────────────────────────────────────────────────

class Sub:
    """Minimal pysrt.SubRipItem stand-in."""
    def __init__(self, index: int, text: str):
        self.index = index
        self.text  = text


def _client(response: str) -> MagicMock:
    c = MagicMock()
    c.generate.return_value = {"response": response}
    return c


def _write_srt_file(path: Path, srt_text: str) -> Path:
    path.write_text(srt_text, encoding="utf-8")
    return path


def _parse_srt_text(srt_text: str) -> list[dict]:
    """Parse SRT text into subtitle entries (index, start, end, text)."""
    entries = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0])
        except ValueError:
            continue
        entries.append({"index": idx, "text": " ".join(lines[2:])})
    return entries


# ═════════════════════════════════════════════════════════════════════════════
# 1. NeMo output format — _segs_to_srt produces correct [Speaker N] tags
# ═════════════════════════════════════════════════════════════════════════════

class TestNemoOutput:

    def test_diarized_srt_has_speaker_tags(self):
        """NeMo diarized output must have [Speaker N] prefix on every line."""
        segs = [
            {"text": "Das ist gut.", "start": 0.0, "end": 2.0, "speaker": "spk_0"},
            {"text": "Ja, genau.",   "start": 2.5, "end": 4.0, "speaker": "spk_1"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        entries = _parse_srt_text(srt)

        assert len(entries) == 2
        for entry in entries:
            assert re.match(r"\[Speaker \d+\]", entry["text"]), \
                f"Missing speaker tag: {entry['text']!r}"

    def test_non_diarized_srt_has_no_tags(self):
        """Non-diarized NeMo output must NOT have speaker tags."""
        segs = [
            {"text": "Hello world.", "start": 0.0, "end": 2.0},
        ]
        srt = _segs_to_srt(segs, diarized=False)
        entries = _parse_srt_text(srt)

        for entry in entries:
            assert "[Speaker" not in entry["text"]

    def test_speaker_numbers_are_sequential(self):
        """Speaker numbers in [Speaker N] must start at 1 and be sequential."""
        segs = [
            {"text": "First.", "start": 0.0, "end": 1.0, "speaker": "spk_2"},
            {"text": "Second.", "start": 1.0, "end": 2.0, "speaker": "spk_0"},
            {"text": "Third.", "start": 2.0, "end": 3.0, "speaker": "spk_1"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        entries = _parse_srt_text(srt)

        speakers = set()
        for entry in entries:
            m = re.match(r"\[Speaker (\d+)\]", entry["text"])
            assert m, f"No speaker tag in: {entry['text']}"
            speakers.add(int(m.group(1)))

        assert 1 in speakers  # Must start at 1
        assert max(speakers) == len(speakers)  # Sequential


# ═════════════════════════════════════════════════════════════════════════════
# 2. Translate step — speaker tags preserved through translation
# ═════════════════════════════════════════════════════════════════════════════

class TestTranslatePreservesTags:

    def test_speaker_tags_reattached_after_translation(self):
        """translate_chunk must strip tags before LLM and re-attach after."""
        chunk = [
            Sub(1, "[Speaker 1] Das ist gut für die Gesundheit."),
            Sub(2, "[Speaker 2] Ja, genau so ist es."),
            Sub(3, "[Speaker 1] Und deshalb ist es wichtig."),
        ]

        # LLM returns clean translations (no speaker tags)
        response = (
            "[1] C'est bien pour la santé.\n"
            "[2] Oui, c'est exactement ça.\n"
            "[3] Et c'est pourquoi c'est important."
        )

        result = translate_chunk(chunk, "de", "fr", _client(response))

        assert result[1].startswith("[Speaker 1]"), f"Tag lost: {result[1]!r}"
        assert result[2].startswith("[Speaker 2]"), f"Tag lost: {result[2]!r}"
        assert result[3].startswith("[Speaker 1]"), f"Tag lost: {result[3]!r}"

        # Verify the translated text is also present
        assert "santé" in result[1]
        assert "exactement" in result[2]

    def test_no_tags_in_input_no_tags_in_output(self):
        """When input has no speaker tags, output should be clean text."""
        chunk = [Sub(1, "Hello world"), Sub(2, "How are you")]
        response = "[1] Bonjour le monde\n[2] Comment allez-vous"

        result = translate_chunk(chunk, "en", "fr", _client(response))

        assert not result[1].startswith("[Speaker")
        assert result[1] == "Bonjour le monde"

    def test_mixed_tagged_and_untagged(self):
        """Mix of tagged and untagged lines — tags preserved where they exist."""
        chunk = [
            Sub(1, "[Speaker 1] Tagged line"),
            Sub(2, "Untagged line"),
            Sub(3, "[Speaker 2] Another tagged"),
        ]
        response = "[1] Ligne taguée\n[2] Ligne non taguée\n[3] Autre taguée"

        result = translate_chunk(chunk, "en", "fr", _client(response))

        assert result[1].startswith("[Speaker 1]")
        assert not result[2].startswith("[Speaker")
        assert result[3].startswith("[Speaker 2]")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Dub step — parse_srt extracts speaker and text correctly
# ═════════════════════════════════════════════════════════════════════════════

class TestDubParsesTranslatedSrt:

    def test_parses_diarized_translated_srt(self, tmp_path):
        """The exact output of translate_chunk fed to parse_srt."""
        srt_content = textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1] C'est bien pour la santé.

            2
            00:00:03,500 --> 00:00:05,000
            [Speaker 2] Oui, exactement.

            3
            00:00:05,500 --> 00:00:07,000
            [Speaker 1] Et c'est pourquoi c'est important.
        """).strip()

        srt_path = _write_srt_file(tmp_path / "video.diarize_fr.srt", srt_content)
        segments = parse_srt(srt_path)

        assert len(segments) == 3

        # Speaker tags extracted, not in text
        assert segments[0]["speaker"] == "Speaker 1"
        assert segments[0]["text"] == "C'est bien pour la santé."
        assert "[Speaker" not in segments[0]["text"]

        assert segments[1]["speaker"] == "Speaker 2"
        assert segments[1]["text"] == "Oui, exactement."

        assert segments[2]["speaker"] == "Speaker 1"
        assert segments[2]["text"] == "Et c'est pourquoi c'est important."

    def test_parse_srt_handles_missing_tags(self, tmp_path):
        """Non-diarized SRT — all segments default to Speaker 1."""
        srt_content = textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:02,000
            Bonjour.

            2
            00:00:02,000 --> 00:00:03,000
            Au revoir.
        """).strip()

        srt_path = _write_srt_file(tmp_path / "test.srt", srt_content)
        segments = parse_srt(srt_path)

        assert all(s["speaker"] == "Speaker 1" for s in segments)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Clean step — speaker tags stripped from final SRTs
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanRemovesTags:

    def test_clean_removes_all_speaker_tags(self, tmp_path):
        """clean_srt_files() must remove ALL [Speaker N] tags."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        # Write a diarized translated SRT
        srt_path = nemo_dir / "video.nemo.de.diarize_fr.srt"
        srt_path.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1] C'est bien pour la santé.

            2
            00:00:03,500 --> 00:00:05,000
            [Speaker 2] Oui, exactement.
        """).strip(), encoding="utf-8")

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        clean_path = nemo_dir / "video.nemo.de.diarize_fr_clean.srt"
        assert clean_path.exists(), "_clean.srt not created"

        content = pysrt.open(str(clean_path))
        for sub in content:
            assert "[Speaker" not in sub.text, \
                f"Speaker tag still present in clean SRT: {sub.text!r}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Full flow — NeMo → translate → dub → clean
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndSpeakerTagFlow:
    """
    Traces speaker tags through the entire pipeline:
    NeMo segments → SRT string → translate_chunk → file → parse_srt → clean.
    """

    def test_full_flow_two_speakers(self, tmp_path):
        # Step 1: NeMo produces diarized segments
        nemo_segs = [
            {"text": "Das ist gut für die Gesundheit.",
             "start": 0.0, "end": 2.5, "speaker": "spk_0"},
            {"text": "Ja, genau so ist es.",
             "start": 2.8, "end": 4.5, "speaker": "spk_1"},
            {"text": "Und deshalb ist es wichtig.",
             "start": 5.0, "end": 7.0, "speaker": "spk_0"},
        ]
        nemo_srt = _segs_to_srt(nemo_segs, diarized=True)

        # Verify NeMo output has speaker tags
        nemo_entries = _parse_srt_text(nemo_srt)
        assert all("[Speaker" in e["text"] for e in nemo_entries), \
            "NeMo SRT missing speaker tags"

        # Step 2: translate_chunk receives pysrt-parsed entries
        srt_path = tmp_path / "video.nemo.de.diarize.srt"
        srt_path.write_text(nemo_srt, encoding="utf-8")
        subs = pysrt.open(str(srt_path))

        chunk = [Sub(s.index, s.text) for s in subs]

        # Build a mock translation response
        response_lines = []
        for s in chunk:
            # Extract text after speaker tag
            m = re.match(r"\[Speaker \d+\]\s*(.*)", s.text)
            text = m.group(1) if m else s.text
            response_lines.append(f"[{s.index}] TRANSLATED_{text}")
        response = "\n".join(response_lines)

        translated = translate_chunk(chunk, "de", "fr", _client(response))

        # Verify speaker tags are preserved in translation output
        for idx, text in translated.items():
            assert "[Speaker" in text, \
                f"Speaker tag lost after translation for index {idx}: {text!r}"

        # Step 3: Write translated SRT and parse for dubbing
        translated_srt_path = tmp_path / "video.nemo.de.diarize_fr.srt"
        for s in subs:
            if s.index in translated:
                s.text = translated[s.index]
        subs.save(str(translated_srt_path), encoding="utf-8")

        dub_segments = parse_srt(translated_srt_path)

        # Verify parse_srt correctly separates speaker from text
        assert len(dub_segments) == 3
        for seg in dub_segments:
            assert seg["speaker"] in ("Speaker 1", "Speaker 2"), \
                f"Bad speaker: {seg['speaker']}"
            assert "[Speaker" not in seg["text"], \
                f"Speaker tag leaked into text: {seg['text']!r}"
            assert "TRANSLATED_" in seg["text"], \
                f"Translation content lost: {seg['text']!r}"

        # Step 4: Clean step removes tags for human-readable SRT
        nemo_dir = tmp_path
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        # Check _clean versions
        for f in nemo_dir.glob("*_clean.srt"):
            content = pysrt.open(str(f))
            for sub in content:
                assert "[Speaker" not in sub.text, \
                    f"Speaker tag in clean SRT: {sub.text!r}"

    def test_words_to_segs_to_srt_to_translate_round_trip(self):
        """
        Full round trip: raw words → _words_to_segs → _segs_to_srt → pysrt → translate_chunk

        This is the exact code path that runs in production.
        """
        # Simulate NeMo word-level output with diarization
        words = []
        for i, (word, spk) in enumerate([
            ("Hallo", "spk_0"), ("Welt.", "spk_0"),
            ("Wie", "spk_1"), ("geht", "spk_1"), ("es?", "spk_1"),
            ("Mir", "spk_0"), ("geht", "spk_0"), ("es", "spk_0"), ("gut.", "spk_0"),
        ]):
            words.append({
                "word": word,
                "start": float(i) * 0.5,
                "end": float(i) * 0.5 + 0.4,
                "speaker": spk,
            })

        # Step 1a: words → segments
        segs = _words_to_segs(words, diarized=True)
        assert len(segs) >= 2, f"Expected at least 2 segments, got {len(segs)}"

        # Step 1b: segments → SRT
        srt = _segs_to_srt(segs, diarized=True)
        entries = _parse_srt_text(srt)
        assert len(entries) >= 2

        # Every entry must have [Speaker N] tag
        for e in entries:
            assert re.match(r"\[Speaker \d+\]", e["text"]), \
                f"Missing tag: {e['text']!r}"

        # Step 2: translate
        chunk = [Sub(i + 1, e["text"]) for i, e in enumerate(entries)]
        response = "\n".join(
            f"[{s.index}] TRANSLATED"
            for s in chunk
        )
        result = translate_chunk(chunk, "de", "fr", _client(response))

        # Speaker tags must survive
        for idx, text in result.items():
            assert "[Speaker" in text, f"Tag lost at index {idx}: {text!r}"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Edge cases that cause real pipeline failures
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_speaker_tag_regex_consistency(self):
        """
        The three regexes used at different pipeline stages must all handle
        the same [Speaker N] format consistently.

        NeMo produces:   [Speaker 1] text
        translate strips: \\[Speaker\\s+\\d+\\]\\s*(.*)
        dub parses:       \\[([^\\]]+)\\]\\s*(.*)
        clean strips:     \\[Speaker\\s+\\d+\\]\\s*
        """
        test_lines = [
            "[Speaker 1] Hello world",
            "[Speaker 10] Multi digit",
            "[Speaker 1]NoSpace",  # edge case
            "[Speaker 99] With numbers 123",
        ]

        # Translate regex
        translate_re = re.compile(r'(\[Speaker\s+\d+\])\s*(.*)', re.DOTALL)
        # Dub regex
        dub_re = re.compile(r"\[([^\]]+)\]\s*(.*)", re.DOTALL)
        # Clean regex
        clean_re = re.compile(r'\[Speaker\s+\d+\]\s*')

        for line in test_lines:
            # Translate must extract tag and content
            t_match = translate_re.match(line)
            assert t_match, f"Translate regex failed on: {line!r}"
            tag, content = t_match.group(1), t_match.group(2)
            assert tag.startswith("[Speaker")
            assert "[Speaker" not in content

            # Dub must extract speaker name and text
            d_match = dub_re.match(line)
            assert d_match, f"Dub regex failed on: {line!r}"
            speaker = d_match.group(1).strip()
            text = d_match.group(2).strip()
            assert "Speaker" in speaker
            assert "[" not in text

            # Clean must strip the tag completely
            cleaned = clean_re.sub("", line).strip()
            assert "[Speaker" not in cleaned, \
                f"Clean regex failed on: {line!r} → {cleaned!r}"

    def test_srt_with_utf8_characters_in_translation(self, tmp_path):
        """Unicode in translated text must survive the full flow."""
        srt = tmp_path / "test.srt"
        srt.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1] C'est génial — très bien !

            2
            00:00:03,000 --> 00:00:05,000
            [Speaker 2] Ça marche à merveille.
        """).strip(), encoding="utf-8")

        segments = parse_srt(srt)
        assert segments[0]["text"] == "C'est génial — très bien !"
        assert segments[1]["text"] == "Ça marche à merveille."

    def test_empty_speaker_tag_in_nemo_output(self):
        """
        If NeMo produces a segment with speaker 'unknown', it should still
        get a valid [Speaker N] label in the SRT.
        """
        segs = [
            {"text": "Unknown speaker text.", "start": 0.0, "end": 2.0, "speaker": "unknown"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        entries = _parse_srt_text(srt)
        assert len(entries) == 1
        assert "[Speaker" in entries[0]["text"]
