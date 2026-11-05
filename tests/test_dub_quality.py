"""Tests for the dub timing check (L4) and the meaning check (L3).

Both work on artefacts, not models: synthetic WAVs and SRTs here, the real
run's artefacts in tools/check_dubbing.py.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nemo"))
sys.path.insert(0, str(ROOT / "qwen3-tts"))

from dub_meaning import (  # noqa: E402
    check_sentences,
    compare_back_translation,
    facts,
    missing_facts,
    normalize,
    similarity,
    word_error_rate,
    write_report,
)
from dub_timing import MIN_GAP, ONSET_TOLERANCE, check_dub  # noqa: E402


def _stamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, cues: list[tuple[float, float, str]]) -> Path:
    blocks = [f"{i}\n{_stamp(a)} --> {_stamp(b)}\n{text}\n"
              for i, (a, b, text) in enumerate(cues, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def write_wav(path: Path, seconds: float, speech_after: float = 0.0,
              rate: int = 24000) -> Path:
    """A WAV with silence up to `speech_after`, then a tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    silence = int(rate * speech_after)
    tone = int(rate * (seconds - speech_after))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * silence)
        handle.writeframes(b"\x20\x00" * tone)
    return path


# ── L4: timing ───────────────────────────────────────────────────────────────

def _dub_fixture(tmp_path: Path, speech_seconds: float, onset: float = 0.0,
                 slot_seconds: float | None = None):
    """A dub with one synthesised segment.

    `slot_seconds` is the subtitle slot it had to fit into. By default the
    slot is longer than the speech, so the segment fits.
    """
    slot = slot_seconds if slot_seconds is not None else speech_seconds + 1.0
    dub = tmp_path / "dub"
    temp = dub / "cache" / "qwen" / "abc" / "temp"
    temp.mkdir(parents=True)
    clip = write_wav(temp / "seg_0001_natural_fit_0123456789abcdef.wav",
                     speech_seconds, onset)
    (temp / "seg_0001_natural_fit_0123456789abcdef.json").write_text(json.dumps({
        "configuration": {"target_seconds": slot,
                          "max_speed": 1.0, "min_speed": 1.0,
                          "output_sample_rate": 24000},
        "edges": {"raw_seconds": speech_seconds, "retained_seconds": speech_seconds},
    }))
    (temp.parent / "checkpoint.json").write_text(json.dumps([
        {"clip": str(clip), "start": 1.0, "end": 1.0 + slot}]))
    script = write_srt(dub / "clip_dub.srt", [(1.0, 1.0 + slot, "Ein Satz.")])
    return dub, script


def test_timing_check_passes_when_the_speech_fits(tmp_path):
    dub, script = _dub_fixture(tmp_path, speech_seconds=2.0)
    report = check_dub(dub, script=script)
    assert report.segments, "no segments found"
    assert not report.too_long
    assert report.ok, report.summary()


def test_timing_check_flags_speech_that_overruns(tmp_path):
    """The voice speaks longer than the slot it was given."""
    dub, script = _dub_fixture(tmp_path, speech_seconds=6.0, slot_seconds=4.0)
    report = check_dub(dub, script=script)
    assert report.too_long, "overlong speech was not flagged"
    assert not report.ok


def test_timing_check_flags_a_late_onset(tmp_path):
    """The voice must not start long after the cue, or the first words are cut."""
    dub, script = _dub_fixture(tmp_path, speech_seconds=2.0, onset=0.4)
    report = check_dub(dub, script=script)
    assert report.late_onsets, "a late onset was not flagged"
    assert report.segments[0].onset > ONSET_TOLERANCE


def test_timing_check_accepts_an_onset_inside_the_tolerance(tmp_path):
    dub, script = _dub_fixture(tmp_path, speech_seconds=2.0, onset=0.02)
    report = check_dub(dub, script=script)
    assert not report.late_onsets


def test_timing_check_reports_nothing_when_there_is_no_dub(tmp_path):
    report = check_dub(tmp_path / "empty")
    assert report.warnings
    assert not report.ok
    assert not report.segments


def test_timing_check_warns_when_the_srt_is_missing(tmp_path):
    """Clips but no SRT: it must say so instead of claiming success."""
    dub, script = _dub_fixture(tmp_path, speech_seconds=2.0)
    script.unlink()
    report = check_dub(dub)
    assert report.warnings
    assert not report.ok


# ── L3: meaning ──────────────────────────────────────────────────────────────

def test_normalize_strips_accents_and_punctuation():
    assert normalize("Ça ne marche pas.") == ["ca", "ne", "marche", "pas"]
    assert normalize("Straße, groß.") == ["strasse", "gross"]


def test_normalize_keeps_every_word():
    """No stopword list: nothing is dropped, so it works for any language."""
    assert normalize("Das ist nicht wichtig") == ["das", "ist", "nicht", "wichtig"]


def test_word_error_rate_is_zero_for_identical_text():
    text = "Es sind vierzehn Jahre später und es bleibt wichtig."
    assert word_error_rate(text, text) == 0.0


def test_word_error_rate_grows_with_differences():
    reference = "Es sind vierzehn Jahre später und es bleibt wichtig."
    near = "Es sind vierzehn Jahre später und es bleibt wichtig."
    far = "Das Wetter ist heute schön."
    assert word_error_rate(reference, near) < word_error_rate(reference, far)


def test_facts_find_numbers_and_acronyms():
    found = facts("CRISPR ist 14 Jahre alt und DNA hilft.")
    assert "14" in found
    assert "crispr" in found
    assert "dna" in found


def test_facts_do_not_guess_at_names():
    """A capitalised word may be an ordinary noun in any language."""
    found = facts("Das Leben der Menschen in Zukunft verändert sich.")
    assert found == set()


def test_facts_work_without_being_told_the_language():
    assert "14" in facts("Cela fait 14 ans")
    assert "crispr" in facts("Le CRISPR est une technologie")


def test_missing_facts_reports_a_dropped_number():
    assert "14" in missing_facts("Es sind 14 Jahre später.", "Es sind später.")


def test_missing_facts_reports_a_dropped_acronym():
    assert "crispr" in missing_facts("CRISPR ist eine Technologie.",
                                     "Das ist eine Technologie.")


def test_missing_facts_uses_the_document_when_a_word_straddles_a_window():
    """A word spoken just after the window is not lost."""
    assert missing_facts("Es sind 14 Jahre.", "Es sind.",
                         document="Es sind. 14 Jahre später.") == []


def test_check_sentences_passes_a_clean_dub():
    pairs = [(1, 0.0, 2.0, "Es sind 14 Jahre später.", "Es sind 14 Jahre später.")]
    report = check_sentences(pairs)
    assert report.ok
    assert report.sentences[0].wer == 0.0


def test_check_sentences_flags_a_lost_number():
    pairs = [(1, 0.0, 2.0, "Es sind 14 Jahre später.", "Es sind später.")]
    report = check_sentences(pairs)
    assert not report.ok
    assert "14" in report.sentences[0].missing


def test_check_sentences_flags_a_meaning_change():
    pairs = [(1, 0.0, 2.0, "Die Forschung war erfolgreich.",
              "Die Forschung war ein Fehlschlag.")]
    report = check_sentences(pairs, max_wer=0.35)
    assert not report.ok


def test_back_translation_comparison():
    """Both texts are in the same language (English -> German -> English)."""
    score, problem = compare_back_translation(
        "When we first published our research on CRISPR it was clear.",
        "When we first published our research on CRISPR it was clear.")
    assert problem is None, problem
    assert score > 0.9


def test_back_translation_detects_a_changed_meaning():
    score, problem = compare_back_translation(
        "The research was successful and changed medicine.",
        "The research failed and changed nothing.")
    assert problem is not None


def test_back_translation_flags_a_changed_meaning():
    score, problem = compare_back_translation(
        "The research was successful and changed medicine.",
        "Die Forschung war ein Fehlschlag und hat nichts bewirkt.")
    assert problem is not None


def test_meaning_report_is_written(tmp_path):
    pairs = [(1, 0.0, 2.0, "Ein Satz.", "Ein Satz.")]
    report = check_sentences(pairs)
    path = tmp_path / "meaning.json"
    write_report(report, path)
    payload = json.loads(path.read_text())
    assert payload["sentences"] == 1
    assert payload["ok"] is True


if __name__ == "__main__":
    import tempfile

    failures = 0
    for test in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            if test.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as tmp:
                    test(Path(tmp))
            else:
                test()
            print("ok", test.__name__)
        except AssertionError as error:
            failures += 1
            print("FAIL", test.__name__, "->", error)
    print(f"\n{'failed' if failures else 'all passed'}: {failures} failure(s)")
    raise SystemExit(1 if failures else 0)
