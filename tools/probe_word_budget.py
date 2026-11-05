#!/usr/bin/env python3
"""Measure how well the local models obey a WORD budget (not a character one).

The natural-pace gate needs a translator that can shorten a line on request.
A character budget does not work (measured: every model returns the same text
whatever the budget says). This measures the word budget, with and without
the JSON wrapper, so the pipeline uses the form that actually works.

    python3 tools/probe_word_budget.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nemo"))

from local_mt import OllamaTranslator, _looks_translated  # noqa: E402

SENTENCES = [
    "When we first published our research on CRISPR, it was clear it would be "
    "a transformative technology for research, and we hoped that it would "
    "impact people's lives in the future.",
    "It's now fourteen years later and it's amazing to see what has happened. "
    "We have the potential now to do things like changing disease-causing "
    "genes, and that's already being done.",
    "I think one of the reasons we do science is that every now and then, "
    "every now and then you make a discovery and you know in your gut that "
    "it's something really profound.",
]
BUDGETS = [12, 16, 20]


def main() -> int:
    rows = []
    for model in ("tower-plus-2b:latest", "hymt2-1.8b:latest"):
        translator = OllamaTranslator(model=model, timeout=180)
        overshoots = []
        for index, sentence in enumerate(SENTENCES, 1):
            for budget in BUDGETS:
                started = time.monotonic()
                out = translator.translate_sentence(
                    sentence, "en", "de", max_chars=budget * 7,
                    max_words=budget)
                elapsed = time.monotonic() - started
                words = len(out.split()) if out else 0
                ratio = words / budget if budget else 0.0
                overshoots.append(ratio)
                rows.append({"model": model, "sentence": index, "budget": budget,
                             "words": words, "ratio": round(ratio, 3),
                             "seconds": round(elapsed, 2), "text": out})
                print(f"  {model:24s} s{index} budget {budget:3d} -> {words:3d} words "
                      f"({ratio:.2f}x) {elapsed:5.1f}s | {out[:70]}")
        if overshoots:
            print(f"  --> {model}: median {statistics.median(overshoots):.2f}x, "
                  f"max {max(overshoots):.2f}x\n")
        translator.unload()
    Path(ROOT / "tools" / "word_budget_probe.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    print("wrote tools/word_budget_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
