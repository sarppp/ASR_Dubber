"""Tests for the measured speech rate and the natural-pace gate.

The bug these lock down: the gate used a hard-coded 11.6 chars/s for German
while the real Qwen voice speaks ~17.6 chars/s. The gate then believed every
line was 1.4-1.6x too long and shortened text that already fitted, losing
meaning for nothing. See nemo/dub_rate_findings.md.

No models, no GPU: everything here is arithmetic over measured constants and
synthetic fixtures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nemo"))
sys.path.insert(0, str(ROOT / "qwen3-tts"))

import pipeline_subtitle as ps  # noqa: E402
from speech_rate import (  # noqa: E402
    DEFAULT_RATES,
    MEASURED_RATES,
    calibrate_rate,
    character_budget,
    engine_key,
    measured_rate,
    word_budget,
)

# The measured evidence from the real 120 s CRISPR dub
# (qwen3-tts/output/dub/CRISPR-120s). Every one of these fitted its slot.
REAL_DUB_EVIDENCE = [
    # (gate estimate with the old 11.6 constant, measured speech/slot)
    (1.41, 0.885),
    (1.59, 0.925),
    (1.15, 0.882),
    (1.34, 0.903),
    (1.16, 0.731),
    (1.10, 0.710),
    (0.94, 0.915),
]


# ── the rate table itself ────────────────────────────────────────────────────

def test_every_engine_has_a_rate_for_every_language():
    for engine, rates in MEASURED_RATES.items():
        for language in ("en", "de", "fr"):
            assert language in rates, f"{engine} has no {language} rate"
            assert rates[language] > 0


def test_qwen_german_rate_is_not_the_old_legacy_constant():
    """The old constant was 11.6 chars/s. The measured Qwen rate is ~1.9x that."""
    assert MEASURED_RATES["qwen"]["de"] > 16.0
    assert MEASURED_RATES["qwen"]["de"] > 1.4 * 11.6


def test_engine_key_resolves_names_and_aliases():
    assert engine_key("qwen") == "qwen"
    assert engine_key("supertonic") == "supertonic"
    assert engine_key("Qwen3-TTS") == "qwen"
    assert engine_key("supertonic-3") == "supertonic"
    assert engine_key("") == ""
    assert engine_key(None) == ""


def test_measured_rate_falls_back_safely():
    # Unknown engine, known language -> the conservative default.
    assert measured_rate("de", "something-else") == DEFAULT_RATES["de"]
    # Known engine -> its own measured rate.
    assert measured_rate("de", "qwen") == MEASURED_RATES["qwen"]["de"]
    # Unknown language -> a usable number, never zero.
    assert measured_rate("xx", "qwen") > 0


def test_env_override_still_wins(monkeypatch=None):
    import os
    os.environ["SUBTITLE_TTS_CPS"] = "12.5"
    try:
        assert measured_rate("de", "qwen") == 12.5
    finally:
        os.environ.pop("SUBTITLE_TTS_CPS", None)


# ── budgets derived from the rate ───────────────────────────────────────────

def test_character_budget_grows_with_the_span():
    short = character_budget(4.0, "de", "qwen")
    long = character_budget(12.0, "de", "qwen")
    assert 0 < short < long


def test_character_budget_is_about_the_measured_rate():
    """8.8 s at the measured Qwen German rate allows ~145 characters."""
    budget = character_budget(8.8, "de", "qwen")
    assert 120 <= budget <= 165, budget


def test_character_budget_is_larger_with_the_fast_engine():
    """Qwen speaks German faster than Supertonic, so it allows more text."""
    assert (character_budget(8.8, "de", "qwen")
            > character_budget(8.8, "de", "supertonic"))


def test_word_budget_scales_with_characters_and_language():
    assert word_budget(143, "de") > word_budget(70, "de")
    # German is longer per word than English.
    assert word_budget(100, "de") <= word_budget(100, "en")


# ── the gate uses the measured rate ──────────────────────────────────────────

def test_gate_rate_is_the_engine_rate():
    assert ps._rate("de", "qwen") == MEASURED_RATES["qwen"]["de"]
    assert ps._rate("de", "supertonic") == MEASURED_RATES["supertonic"]["de"]


def test_gate_no_longer_flags_lines_that_fit_in_reality():
    """The regression that broke the shipped German.

    With the old 11.6 constant every one of these blocks was flagged as
    1.1-1.6x too long. With the measured rate they are not, so the gate keeps
    the wording and the meaning.
    """
    # (span, chars) of the real blocks, and the measured speech/slot ratio.
    blocks = [(8.80, 152, 0.885), (9.84, 161, 0.925), (6.52, 122, 0.903),
              (9.44, 175, 0.731), (8.88, 132, 0.882), (7.80, 131, 0.710),
              (9.04, 125, 0.915)]
    cps = MEASURED_RATES["qwen"]["de"]
    flagged_old = 0
    flagged_new = 0
    for span, chars, _measured in blocks:
        if chars / 11.6 / span > 1.0:
            flagged_old += 1
        if chars / cps / span > 1.0:
            flagged_new += 1
    assert flagged_old == len(blocks), "the old constant should have flagged all"
    assert flagged_new < len(blocks) / 2, (
        f"the measured rate should flag few blocks, flagged {flagged_new}")


def test_gate_accepts_a_line_that_the_old_rate_rejected():
    """A block that fitted in reality must not be shortened by the gate."""
    span, chars = 8.80, 152
    assert chars / 11.6 / span > 1.0          # old rate: too long
    assert chars / MEASURED_RATES["qwen"]["de"] / span <= 1.0   # measured: fits


# ── calibration from a previous dub ──────────────────────────────────────────

def test_calibration_reads_the_measured_rate_from_sidecars(tmp_path):
    """A previous dub measures the voice; the next gate uses that number."""
    import wave

    dub = tmp_path / "dub"
    cache = dub / "cache" / "qwen" / "abc"
    temp = cache / "temp"
    temp.mkdir(parents=True)

    # One synthesised clip: 100 characters spoken in 5 s -> 20 chars/s.
    clip = temp / "seg_0001_natural_fit_abcdef0123456789.wav"
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * (24000 * 5))
    (temp / "seg_0001_natural_fit_abcdef0123456789.json").write_text(json.dumps({
        "configuration": {"policy": "natural-fit-v1", "target_seconds": 6.0,
                          "max_speed": 1.0, "min_speed": 1.0,
                          "output_sample_rate": 24000},
        "edges": {"raw_seconds": 5.0, "retained_seconds": 5.0,
                  "start_sample": 0, "end_sample": 120000},
    }))
    (cache / "checkpoint.json").write_text(json.dumps([
        {"clip": str(clip), "start": 1.0, "end": 7.0}]))

    # The SRT that was spoken: 100 characters in the covered window.
    script = dub / "clip_dub.srt"
    script.write_text(
        "1\n00:00:01,000 --> 00:00:07,000\n" + "Wort " * 20 + "\n",
        encoding="utf-8")

    result = calibrate_rate(dub, "de", "qwen")
    assert result is not None
    rate, evidence = result
    assert evidence["samples"] == 1
    assert 19.0 <= rate <= 21.0, rate


def test_calibration_returns_none_without_a_dub(tmp_path):
    assert calibrate_rate(tmp_path / "nothing", "de", "qwen") is None


def test_calibration_returns_none_without_a_script(tmp_path):
    """Sidecars but no SRT -> no measurement, and no crash."""
    import wave

    dub = tmp_path / "dub"
    temp = dub / "cache" / "qwen" / "abc" / "temp"
    temp.mkdir(parents=True)
    clip = temp / "seg_0001_natural_fit_0123456789abcdef.wav"
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 24000)
    (temp / "seg_0001_natural_fit_0123456789abcdef.json").write_text(json.dumps({
        "configuration": {"target_seconds": 1.0},
        "edges": {"raw_seconds": 1.0},
    }))
    assert calibrate_rate(dub, "de", "qwen") is None


# ── the word-budget control the translator now uses ──────────────────────────

def test_json_text_is_extracted_from_a_json_answer():
    from local_mt import _extract_json_text
    assert _extract_json_text('{"text": "Ein kurzer Satz."}') == "Ein kurzer Satz."


def test_json_text_survives_a_code_fence():
    from local_mt import _extract_json_text
    fenced = '```json\n{"text": "Ein kurzer Satz."}\n```'
    assert _extract_json_text(fenced) == "Ein kurzer Satz."


def test_json_text_falls_back_to_the_raw_answer():
    from local_mt import _extract_json_text
    assert _extract_json_text("Ein kurzer Satz.") == "Ein kurzer Satz."
    assert _extract_json_text("") == ""


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
