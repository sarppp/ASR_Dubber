#!/usr/bin/env python3
"""Probe whether Ling (OpenRouter) honours a character budget, with clean input.

The first probe fed a mixed DE/EN fragment and Ling analysed it instead of
translating. This uses a clean English sentence and an explicit user turn,
and compares several models so we can pick one that actually shortens.

    python3 tools/probe_ling_budget2.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "nemo" / ".env"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Clean English source sentence (what ASR would produce).
SOURCE = ("When we first published our research on CRISPR, it was clear it would be "
          "a transformative technology for research, and we hoped that it would "
          "impact people's lives in the future.")
BUDGETS = [143, 100, 80]

SYSTEM = (
    "You are a professional German subtitle translator. You output ONLY the "
    "German translation — no analysis, no preamble, no explanation, no "
    "markdown. Keep every name, number, negation and fact. The line must fit "
    "a subtitle slot, so keep it short and compress the wording: use at most "
    "{budget} characters. Never drop a fact to save characters. Output one "
    "single line of German text and nothing else."
)


def api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip().upper().replace(" ", "")
        if name.replace("_", "") in {"OPENROUTERAPIKEY", "OPENROUTER"}:
            return value.strip().strip("\"'")
    raise SystemExit("OpenRouter key missing from nemo/.env")


def call(model: str, budget: int) -> tuple[str, float]:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": SYSTEM.format(budget=budget)},
            {"role": "user", "content": SOURCE},
        ],
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": "Bearer " + api_key(),
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/asr-pipeline",
                 "X-Title": "budget-probe"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return f"HTTP {exc.code}: {detail}", time.monotonic() - started
    text = payload["choices"][0]["message"]["content"] or ""
    return text.strip(), time.monotonic() - started


MODELS = [
    "inclusionai/ling-3.0-flash",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
]


def main() -> int:
    results = []
    for model in MODELS:
        for budget in BUDGETS:
            text, seconds = call(model, budget)
            length = len(text)
            results.append({"model": model, "budget": budget, "chars": length,
                            "over": round(length / budget, 3),
                            "seconds": round(seconds, 2), "text": text})
            flag = "OK " if length <= budget else "OVER"
            print(f"  {model:34s} budget {budget:4d} {flag} {length:4d} chars "
                  f"({length / budget:.2f}x) {seconds:5.1f}s")
            if length <= budget * 1.5:
                print(f"      {text[:160]}")
    Path(ROOT / "tools" / "ling_budget_probe2.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\nwrote tools/ling_budget_probe2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
