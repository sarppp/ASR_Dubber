#!/usr/bin/env python3
"""Check that the compressor can shorten the lines the gate cannot fit.

Uses the real failing sentences from the 120 s CRISPR run. For each one it
reports what the compressor produced and whether the result would fit the
slot under the measured speech model.

    python3 tools/check_compressor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nemo"))

import pipeline_subtitle as ps  # noqa: E402
from speech_rate import speech_seconds  # noqa: E402

# (segment, slot seconds, german text, english source) — the four that did not fit.
FAILING = [
    (2, 6.08, "Sie wissen, es ist ein ziemlich viskales Gefühl, dass Sie etwas "
              "von großer Bedeutung entdeckt haben.",
     "You know, it is kind of a visceral sense that you've uncovered something "
     "of great import."),
    (3, 3.72, "Als wir unsere Forschung zu CRISPR erstmals veröffentlichten.",
     "When we first published our research on CRISPR."),
    (4, 8.24, "Es war klar, dass es sich um eine transformative Technologie für "
              "die Forschung handeln würde, und wir hofften natürlich, dass sie "
              "das Leben der Menschen in Zukunft beeinflussen würde.",
     "It was clear that it was going to be a transformative technology for "
     "research, and we hoped, of course, for impacting people's lives in the "
     "future."),
    (9, 3.28, "CRISPR ist ein großartiges Akronym, aber was bedeutet es?",
     "CRISPR, an acronym that is very promising, but what does it mean?"),
]


def main() -> int:
    compressor = ps._make_compressor("de")
    if compressor is None:
        print("no compressor available")
        return 2

    fitted = 0
    for index, slot, text, source in FAILING:
        need = speech_seconds(len(text), "de", "qwen")
        print(f"\n── segment {index}: slot {slot:.2f}s, {len(text)} chars, "
              f"needs {need:.2f}s ({need / slot:.2f}x)")
        words = max(4, int(len(text.split()) * (slot / need) * 0.95))
        shortened, report = compressor.shorten(
            text, words,
            guard=lambda before, after: ps._preservation_problem(before, after, "de"),
            source_text=source,
            fits=lambda candidate: speech_seconds(len(candidate), "de", "qwen") <= slot)
        if shortened == text:
            print("   ✗ no valid shortening found")
            for rejection in report.get("rejected", []):
                print(f"       rejected {rejection['engine']}/{rejection['form']}: "
                      f"{rejection['reason']}")
            continue
        new_need = speech_seconds(len(shortened), "de", "qwen")
        fits = new_need <= slot
        fitted += fits
        print(f"   ✓ {report.get('engine')} -> {len(shortened)} chars, "
              f"needs {new_need:.2f}s ({new_need / slot:.2f}x) "
              f"{'FITS' if fits else 'still too long'}")
        print(f"     {shortened}")
    print(f"\nfitted {fitted}/{len(FAILING)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
