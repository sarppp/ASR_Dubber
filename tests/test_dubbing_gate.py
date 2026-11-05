"""L0/L1 tests for the subtitle-dubbing gate and adaptation (no models, no GPU).

L0: guards, sentence grouping, SRT integrity, merge boundaries, report shape.
L1: fake TTS stub — gate -> adapt -> gate must converge or report honestly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nemo"))
sys.path.insert(0, str(ROOT / "qwen3-tts"))

import pipeline_subtitle as ps  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

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


class FakeTranslator:
    """Returns text of exactly `max_chars`, varied enough to dodge the loop guard."""

    def __init__(self, ignore_budget: bool = False):
        self.ignore_budget = ignore_budget
        self.calls = 0

    def translate_sentence(self, text, source, target, max_chars=None,
                           max_words=None):
        self.calls += 1
        if self.ignore_budget:
            return text
        budget = max_chars or 40
        numbers = re.findall(r"\d+", text)
        out = list(numbers)
        i = 0
        while len(" ".join(out)) < budget:
            out.append(f"wort{i}")
            i += 1
        line = " ".join(out)
        # Keep every number the original had: the meaning guard rejects a
        # candidate that drops one.
        kept = " ".join(numbers) if numbers else ""
        return f"{kept} {line[:budget]}".strip()[:budget + len(kept)]


# ── L0: guards ───────────────────────────────────────────────────────────────

def test_guard_rejects_number_loss():
    assert ps._preservation_problem("Es sind 14 Jahre später.", "Es ist viel später.", "de")
    assert ps._preservation_problem("Es sind 14 Jahre später.", "Es sind 14 Jahre danach.", "de") is None


def test_guard_rejects_negation_loss():
    assert ps._preservation_problem("Das ist nicht wichtig.", "Das ist wichtig.", "de")
    assert ps._preservation_problem("Ce n'est pas vrai.", "C'est vrai.", "fr")
    assert ps._preservation_problem("Das ist nicht wichtig.", "Das ist nicht wichtig.", "de") is None


def test_guard_rejects_shrink():
    long_text = "Ein ziemlich langer deutscher Satz mit vielen Wörtern und Details."
    assert ps._preservation_problem(long_text, "Kurz.", "de")


def test_guard_rejects_repetition_loop():
    loop = "und manchmal " * 12
    assert ps._preservation_problem("Ein Satz.", loop.strip(), "de")
    assert ps._preservation_problem("Ein Satz.", "Zwei Wörter hier.", "de") is None


# ── L0: sentence grouping and SRT integrity ──────────────────────────────────

def test_sentence_groups_end_on_punctuation_or_caps():
    cues = [{"start": i * 2.0, "end": i * 2.0 + 1.9,
             "text": t} for i, t in enumerate(
        ["Erster Teil,", "zweiter Teil.", "Dritter Teil,", "vierter Teil!"])]
    spans = ps._sentence_spans(cues)
    assert spans[0] == (0, 1)          # closed on '.'
    assert spans[1] == (2, 3)          # closed on '!'
    assert len(spans) == 2


def test_sentence_group_respects_max_cues():
    from local_mt import MAX_SENTENCE_CUES
    cues = [{"start": i * 1.0, "end": i * 1.0 + 0.9, "text": "weiter und weiter,"}
            for i in range(MAX_SENTENCE_CUES + 3)]
    spans = ps._sentence_spans(cues)
    assert spans[0][1] - spans[0][0] + 1 == MAX_SENTENCE_CUES


def test_render_srt_is_monotonic_and_preserves_text():
    cues = [(0.0, 1.0, "eins"), (1.0, 2.5, "zwei"), (2.5, 4.0, "drei")]
    rendered = ps._render_srt(cues)
    blocks = [b.splitlines() for b in rendered.strip().split("\n\n")]
    assert [b[2] for b in blocks] == ["eins", "zwei", "drei"]
    starts, ends = [], []
    for block in blocks:
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", block[1])
        starts.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000)
        ends.append(int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000)
    assert starts == sorted(starts)
    assert all(ends[i] <= starts[i + 1] for i in range(len(starts) - 1))


def test_merge_never_crosses_a_finished_sentence():
    from dub_srt import merge_segments
    segments = [
        {"index": 1, "start": 0.0, "end": 2.0, "speaker": "A", "text": "Erster Satz."},
        {"index": 2, "start": 2.0, "end": 4.0, "speaker": "A", "text": "Zweiter Satz."},
    ]
    merged = merge_segments(segments, gap_sec=1.0, max_dur=10.0)
    assert len(merged) == 2, merged


# ── L1: fake TTS convergence ─────────────────────────────────────────────────

def _adapt_case(tmp_path, translator, cues, source_cues=None):
    target = write_srt(tmp_path / "clip.nemo.de.srt", cues)
    source = write_srt(tmp_path / "clip.nemo.en.srt", source_cues or cues)
    adapted, rows, dub = ps._adapt(target, source, "en", "de", translator,
                                   target=1.0, max_sentences=None)
    return adapted, rows, dub


def test_adapt_converges_when_rate_matches(tmp_path):
    cues = [(4.3, 13.1, "Ich denke, einer der Gründe ist, dass es hin und wieder selten ist.")]
    adapted, rows, dub = _adapt_case(tmp_path, FakeTranslator(), cues)
    assert rows and all(row["ratio_after"] <= 1.0 for row in rows), rows
    assert adapted.exists() and dub.exists()


def test_adapt_reports_when_the_voice_ignores_the_budget(tmp_path):
    cues = [(0.0, 4.0, "Ein Satz, der eindeutig viel zu lang für seinen Platz ist und bleibt.")]
    adapted, rows, dub = _adapt_case(tmp_path, FakeTranslator(ignore_budget=True), cues)
    # It cannot shrink anything; the report must say so instead of faking a fit.
    assert rows[0]["ratio_after"] > 1.0
    assert rows[0]["text_after"] == rows[0]["text_before"]


def test_adapt_keeps_original_when_candidate_loses_a_number(tmp_path):
    class NumberDropper:
        def translate_sentence(self, text, source, target, max_chars=None,
                               max_words=None):
            return "Es ist später."

    # A line that is genuinely too long for its slot, so the guard is reached.
    # (The measured rate is ~17.6 chars/s for German, so 130 characters in
    # 4 seconds is about 1.8x over.)
    cues = [(0.0, 4.0, "Es sind 14 Jahre später und es bleibt wichtig, "
                       "denn die Forschung ging weiter und die Ergebnisse "
                       "waren sehr wichtig für uns alle.")]
    _adapted, rows, _dub = _adapt_case(tmp_path, NumberDropper(), cues)
    assert rows[0].get("kept_original")
    assert rows[0]["text_after"] == rows[0]["text_before"]


# ── L2: measure-then-adapt using the dub's fit report ───────────────────────

def _rows_and_files(tmp_path, count=10):
    cues = [(i * 2.0, i * 2.0 + 1.9, f"Deutscher Satz Nummer {i + 1} bleibt hier.")
            for i in range(count)]
    dub = write_srt(tmp_path / "clip.nemo.de.dub.srt", cues)
    source = write_srt(tmp_path / "clip.nemo.en.srt",
                       [(a, b, f"English sentence number {i + 1}.") for i, (a, b, _) in enumerate(cues)])
    rows = [{"index": i + 1, "start": a, "end": b, "text_after": t,
             "text_before": t, "span": b - a, "ratio_before": 1.0, "ratio_after": 1.0,
             "chars_before": len(t), "chars_after": len(t)}
            for i, (a, b, t) in enumerate(cues)]
    return dub, source, rows


def test_report_readapts_only_the_failing_sentences(tmp_path):
    import json
    dub, source, rows = _rows_and_files(tmp_path)
    # The dub reports the segments it synthesised, matched to the adaptation
    # rows by TIME (dub.py merges cues, so its indices do not line up).
    report = tmp_path / "fit_report.json"
    report.write_text(json.dumps({"max_speed": 1.0, "segments": [
        {"index": 2, "start": rows[1]["start"], "tempo": 1.15, "span": 1.9},
        {"index": 5, "start": rows[4]["start"], "tempo": 1.08, "span": 1.9},
        {"index": 9, "start": rows[8]["start"], "tempo": 1.19, "span": 1.9},
    ]}), encoding="utf-8")

    changed = ps._adapt_from_report(dub, source, "en", "de", rows, report,
                                    FakeTranslator(), target=1.0)
    assert len(changed) == 3, changed
    touched = [row["index"] for row in rows if row.get("ratio_measured")]
    assert touched == [2, 5, 9], touched
    untouched = [row for row in rows if row["index"] not in (2, 5, 9)]
    assert all(row["text_after"] == row["text_before"] for row in untouched)
    assert len(ps._parse_srt(dub)) == 10


def test_report_matches_the_failing_segment_by_time(tmp_path):
    """A dub segment number that does not match the row number is fine:
    the segment is matched to the row that covers its time."""
    import json
    dub, source, rows = _rows_and_files(tmp_path)
    report = tmp_path / "fit_report.json"
    # Segment "1" is reported, but it starts on the fifth row's time.
    report.write_text(json.dumps({"max_speed": 1.0, "segments": [
        {"index": 1, "start": rows[4]["start"], "tempo": 1.2, "span": 1.9},
    ]}), encoding="utf-8")
    changed = ps._adapt_from_report(dub, source, "en", "de", rows, report,
                                    FakeTranslator(), target=1.0)
    assert len(changed) == 1, changed
    assert rows[4].get("ratio_measured"), rows[4]
    assert not any(row.get("ratio_measured") for row in rows[:4])


def test_report_without_failures_changes_nothing(tmp_path):
    import json
    dub, source, rows = _rows_and_files(tmp_path)
    report = tmp_path / "fit_report.json"
    report.write_text(json.dumps({"max_speed": 1.0, "segments": [
        {"index": 3, "start": rows[2]["start"], "tempo": 1.0, "span": 1.9}]}),
        encoding="utf-8")
    assert ps._adapt_from_report(dub, source, "en", "de", rows, report,
                                 FakeTranslator(), target=1.0) == []


if __name__ == "__main__":
    import tempfile

    for test in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        with tempfile.TemporaryDirectory() as tmp:
            if test.__code__.co_argcount:
                test(Path(tmp))
            else:
                test()
        print("ok", test.__name__)
