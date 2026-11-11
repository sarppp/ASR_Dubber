#!/usr/bin/env python3
"""Validate the JSON+word-count control across sentences and budgets.

The single-probe result was promising (15 words for a 14-word budget) but one
sample proves nothing. This runs several real sentences through the control and
reports the overshoot distribution, for both Ling (OpenRouter) and the local
Tower model, so we can pick the mechanism that actually holds.

    python3 tools/probe_control_matrix.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "nemo" / ".env"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
sys.path.insert(0, str(ROOT / "nemo"))

from local_mt import OllamaTranslator  # noqa: E402

# Real sentences from the CRISPR 120 s clip (the actual failing run).
SENTENCES = [
    "I think one of the reasons we do science is that every now and then, "
    "every now and then you make a discovery and you know in your gut that "
    "it's something really profound.",
    "When we first published our research on CRISPR, it was clear it would be "
    "a transformative technology for research, and we hoped that it would "
    "impact people's lives in the future.",
    "It's now fourteen years later and it's amazing to see what has happened. "
    "We have the potential now to do things like changing disease-causing "
    "genes, and that's already being done.",
]
WORD_BUDGETS = [10, 14, 18]


def api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip().upper().replace(" ", "")
        if name.replace("_", "") in {"OPENROUTERAPIKEY", "OPENROUTER"}:
            return value.strip().strip("\"'")
    raise SystemExit("OpenRouter key missing from nemo/.env")


def ling_json(source: str, words: int) -> tuple[str, float]:
    body = json.dumps({
        "model": "inclusionai/ling-3.0-flash", "temperature": 0, "max_tokens": 2000,
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content":
                "You are a professional German subtitle translator. Keep every name, "
                "number, negation and fact; never drop information. Answer as JSON "
                f"with exactly one key \"text\" whose value is a German translation "
                f"of at most {words} words."},
            {"role": "user", "content": source},
        ],
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": "Bearer " + api_key(),
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/asr-pipeline",
                 "X-Title": "control-matrix"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}", time.monotonic() - started
    raw = payload["choices"][0]["message"]["content"] or ""
    try:
        return json.loads(raw).get("text", ""), time.monotonic() - started
    except Exception:
        return raw, time.monotonic() - started


def tower_json(source: str, words: int) -> tuple[str, float]:
    """The same JSON+word-count control against the local Tower model."""
    prompt = (
        "You are a professional German subtitle translator. Keep every name, number, "
        "negation and fact; never drop information. Answer as JSON with exactly one "
        f"key \"text\" whose value is a German translation of at most {words} words.\n\n"
        f"{source}")
    translator = OllamaTranslator(model="tower-plus-2b:latest", timeout=120)
    started = time.monotonic()
    raw = translator.generate(prompt, num_predict=400)
    elapsed = time.monotonic() - started
    try:
        return json.loads(raw).get("text", ""), elapsed
    except Exception:
        return raw, elapsed


def main() -> int:
    rows = []
    for name, fn in (("ling", ling_json), ("tower", tower_json)):
        overshoots = []
        for index, sentence in enumerate(SENTENCES, 1):
            for budget in WORD_BUDGETS:
                text, seconds = fn(sentence, budget)
                words = len(text.split())
                overshoot = words / budget
                overshoots.append(overshoot)
                rows.append({"engine": name, "sentence": index, "budget": budget,
                             "words": words, "overshoot": round(overshoot, 3),
                             "seconds": round(seconds, 2), "text": text})
                print(f"  {name:5s} s{index} budget {budget:3d} -> {words:3d} words "
                      f"({overshoot:.2f}x) {seconds:5.1f}s | {text[:90]}")
        print(f"  --> {name}: median overshoot {statistics.median(overshoots):.2f}x, "
              f"max {max(overshoots):.2f}x, within budget "
              f"{sum(1 for o in overshoots if o <= 1.0)}/{len(overshoots)}\n")

    Path(ROOT / "tools" / "control_matrix.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    print("wrote tools/control_matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
