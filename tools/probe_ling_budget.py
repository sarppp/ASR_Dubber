#!/usr/bin/env python3
"""Probe whether Ling (OpenRouter) honours a character budget.

Tower ignores the budget entirely, which blocks the natural-pace dub loop.
This measures whether Ling does better, before wiring it into the pipeline.

    python3 tools/probe_ling_budget.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "nemo" / ".env"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SOURCE = (
    "for the research handeln würde, und wir hofften natürlich, dass sie das Leben "
    "der Menschen in Zukunft beeinflussen würde. Es ist jetzt, 14 Jahre später, "
    "erstaunlich zu sehen"
)
BUDGETS = [143, 100, 80]


def api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip().upper().replace(" ", "")
        if name.replace("_", "") in {"OPENROUTERAPIKEY", "OPENROUTER"}:
            return value.strip().strip("\"'")
    raise SystemExit("OpenRouter key missing from nemo/.env")


def call(prompt: str, model: str) -> tuple[str, dict]:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"enabled": False},
        "messages": [{"role": "system", "content": prompt}],
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
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    text = payload["choices"][0]["message"]["content"] or ""
    return text.strip(), {"seconds": round(time.monotonic() - started, 2),
                          "usage": payload.get("usage")}


SYSTEM = (
    "You are a subtitle translator from English to German. Translate the "
    "sentence the user sends. Keep every name, number, negation and fact. "
    "The translation must be spoken aloud in the same time as the original, "
    "so keep it SHORT: use at most {budget} characters. Compress the wording, "
    "never drop information. Answer with the translation only, one line."
)


def main() -> int:
    model = os.getenv("LING_MODEL", "inclusionai/ling-3.0-flash")
    results = []
    for budget in BUDGETS:
        prompt = SYSTEM.format(budget=budget)
        text, meta = call(f"{SOURCE}", model)
        length = len(text)
        results.append({"budget": budget, "chars": length,
                        "over": round(length / budget, 3), "text": text, **meta})
        print(f"  budget {budget:4d} -> {length:4d} chars "
              f"({length / budget:.2f}x)  {meta['seconds']}s")
        print(f"      {text[:150]}")
    Path(ROOT / "tools" / "ling_budget_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\nwrote tools/ling_budget_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
