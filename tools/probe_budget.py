#!/usr/bin/env python3
"""Probe which local MT model can honour a character budget.

The dub loop needs a translator that returns a line at most `max_chars` long.
Tower ignores a prompt-only budget, so this measures the real adherence of
every locally available model against the same sentence and the same budget.

    python3 tools/probe_budget.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nemo"))

from local_mt import OllamaTranslator  # noqa: E402

# The sentence that failed in the real 120 s run (segment 4, budget 143).
SOURCE = (
    "for the research handeln würde, und wir hofften natürlich, dass sie das Leben "
    "der Menschen in Zukunft beeinflussen würde. Es ist jetzt, 14 Jahre später, "
    "erstaunlich zu sehen"
)
BUDGETS = [143, 100, 80]

MODELS = [
    "tower-plus-2b:latest",
    "tg-4b:latest",
    "hymt2-1.8b:latest",
    "eurollm1.7b:latest",
    "salamandra2b:latest",
]


def main() -> int:
    results = []
    for model in MODELS:
        translator = OllamaTranslator(model=model, timeout=180)
        for budget in BUDGETS:
            started = time.monotonic()
            try:
                text = translator.translate_sentence(
                    SOURCE, "en", "de", max_chars=budget)
            except Exception as error:  # noqa: BLE001
                results.append({"model": model, "budget": budget,
                                "error": f"{type(error).__name__}: {error}"})
                print(f"  {model:28s} budget {budget:4d}  ERROR {error}")
                continue
            elapsed = time.monotonic() - started
            length = len(text)
            results.append({"model": model, "budget": budget, "chars": length,
                            "over": round(length / budget, 3), "seconds": round(elapsed, 2),
                            "text": text})
            print(f"  {model:28s} budget {budget:4d} -> {length:4d} chars "
                  f"({length / budget:.2f}x)  {elapsed:5.1f}s")
        translator.unload()
    Path(ROOT / "tools" / "budget_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\nwrote tools/budget_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
