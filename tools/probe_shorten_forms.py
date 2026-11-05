#!/usr/bin/env python3
"""Find the prompt form that makes the LOCAL model actually shorten a line.

The gate needs a translator that can be told to compress a sentence. The
character budget does not work: Tower, Ling, Gemini and Haiku all return the
same text whatever the budget says.

This sweeps prompt shapes against the local models and reports which ones
actually produce shorter text, so the pipeline uses a form that works instead
of one that looks right.

    python3 tools/probe_shorten_forms.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nemo"))

from local_mt import OllamaTranslator, _bigrams, _extract_json_text  # noqa: E402

SOURCE = (
    "Es war klar, dass es sich um eine transformative Technologie für die "
    "Forschung handeln würde, und wir hofften natürlich, dass sie das Leben "
    "der Menschen in Zukunft beeinflussen würde."
)
SOURCE_EN = (
    "It was clear that it would be a transformative technology for research, "
    "and we hoped that it would impact people's lives in the future."
)
TARGET_CHARS = 142          # what fits the slot
BUDGET_WORDS = 22

FORMS = {
    "1 json words": (
        f"Translate the German sentence into German (de).\n"
        f"Answer as JSON: {{\"text\": \"<the German translation>\"}}\n"
        f"Use at most {BUDGET_WORDS} words. Keep every name, number, negation "
        f"and fact; compress the wording, never drop information."),
    "2 json words tight": (
        f"Translate the German sentence into German (de).\n"
        f"Answer as JSON: {{\"text\": \"<the German translation>\"}}\n"
        f"Use at most {BUDGET_WORDS} words. Keep every name, number and fact. "
        f"Remove filler words and redundant clauses. Never drop a fact."),
    "3 rewrite json": (
        f"Rewrite the German sentence so it is shorter but keeps every fact.\n"
        f"Use at most {BUDGET_WORDS} words. Compress the wording only.\n"
        f"Answer as JSON: {{\"text\": \"<the rewritten German>\"}}"),
    "4 english then german": (
        f"First shorten the English sentence to at most {BUDGET_WORDS} words, "
        f"keeping every fact. Then translate it into German (de) and answer as "
        f"JSON: {{\"text\": \"<the German translation>\"}}"),
    "5 json + chars": (
        f"Translate the German sentence into German (de).\n"
        f"Answer as JSON: {{\"text\": \"<the German translation>\"}}\n"
        f"The translation must be at most {TARGET_CHARS} characters. Keep every "
        f"name, number and fact; compress the wording, never drop information."),
}


def main() -> int:
    rows = []
    for model in ("tower-plus-2b:latest", "hymt2-1.8b:latest"):
        translator = OllamaTranslator(model=model, timeout=240)
        for name, prompt in FORMS.items():
            body = SOURCE_EN if "english" in name else SOURCE
            raw = translator.generate(prompt + f"\n\n{body}", num_predict=500)
            text = _extract_json_text(raw)
            length = len(text)
            words = len(text.split())
            rows.append({"model": model, "form": name, "chars": length,
                         "words": words, "text": text,
                         "fits": length <= TARGET_CHARS})
            flag = "FITS " if length <= TARGET_CHARS else "over "
            print(f"  {model:22s} {name:22s} {flag} {length:4d} chars "
                  f"{words:3d} words | {text[:70]}")
        translator.unload()
        print()

    fitting = [r for r in rows if r["fits"]]
    print(f"forms that fit the {TARGET_CHARS}-char slot: {len(fitting)}/{len(rows)}")
    for r in fitting:
        print(f"   {r['model']} / {r['form']}")
    Path(ROOT / "tools" / "shorten_forms.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    print("\nwrote tools/shorten_forms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
