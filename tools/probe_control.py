#!/usr/bin/env python3
"""Find a control that actually makes the model shorten the output.

All three probes so far failed: Tower, Ling, Gemini and Haiku all returned the
SAME text for budgets 143/100/80. "Use at most N characters" is not an
instruction these models obey.

This probe tests techniques that are known to work with instruction models:

  A  characters          (baseline, known to fail)
  B  word count         "at most N words"
  C  word count + JSON  {"text": "..."} so the model counts inside a structure
  D  word count, twice  ask, then re-ask the model to shorten its own answer
  E  local truncation    model output + hard cut at a word boundary

Technique D (self-shortening) and E (hard cut) are the interesting ones:
D lets the model do the semantic compression, E is the guaranteed fallback.

    python3 tools/probe_control.py
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "nemo" / ".env"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SOURCE = ("When we first published our research on CRISPR, it was clear it would be "
          "a transformative technology for research, and we hoped that it would "
          "impact people's lives in the future.")


def api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip().upper().replace(" ", "")
        if name.replace("_", "") in {"OPENROUTERAPIKEY", "OPENROUTER"}:
            return value.strip().strip("\"'")
    raise SystemExit("OpenRouter key missing from nemo/.env")


def call(messages: list[dict], model: str = "inclusionai/ling-3.0-flash",
         max_tokens: int = 2000) -> tuple[str, float]:
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "reasoning": {"enabled": False}, "messages": messages,
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": "Bearer " + api_key(),
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/asr-pipeline",
                 "X-Title": "control-probe"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return f"HTTP {exc.code}: {detail}", time.monotonic() - started
    return (payload["choices"][0]["message"]["content"] or "").strip(), \
        time.monotonic() - started


BASE = ("You are a professional German subtitle translator. Output ONLY the German "
        "translation. Keep every name, number, negation and fact. Never drop "
        "information. Output one line, no preamble, no markdown.")

WORD_BUDGET = 14   # ~14 German words is roughly 90 characters


def technique_b() -> tuple[str, float]:
    prompt = BASE + (f" The translation must be at most {WORD_BUDGET} WORDS long. "
                     "Compress the wording, keep all facts.")
    return call([{"role": "system", "content": prompt},
                 {"role": "user", "content": SOURCE}])


def technique_c() -> tuple[str, float]:
    prompt = (BASE + " Answer as JSON with exactly one key \"text\". The value must "
              f"contain at most {WORD_BUDGET} words.")
    text, seconds = call([{"role": "system", "content": prompt},
                          {"role": "user", "content": SOURCE}])
    try:
        return json.loads(text).get("text", ""), seconds
    except Exception:
        return text, seconds


def technique_d() -> tuple[str, float]:
    """Ask once, then ask the model to shorten its own answer."""
    prompt = BASE + " Output one line, no preamble, no markdown."
    first, seconds = call([{"role": "system", "content": prompt},
                           {"role": "user", "content": SOURCE}])
    shorten = (f"Shorten this German subtitle to at most {WORD_BUDGET} words. "
               "Keep every name, number and negation. Compress the wording only; "
               "never drop a fact. Output one line, nothing else.\n\n" + first)
    second, extra = call([{"role": "system", "content": BASE},
                          {"role": "user", "content": shorten}])
    return second, seconds + extra


def technique_e(first: str) -> str:
    """Hard cut at a word boundary, as a guaranteed last resort."""
    words = first.split()
    return " ".join(words[:WORD_BUDGET])


def main() -> int:
    results = []
    label, (text, seconds) = "B words", technique_b()
    results.append({"technique": label, "text": text, "chars": len(text),
                    "words": len(text.split()), "seconds": round(seconds, 2)})
    print(f"  B words      : {len(text):4d} chars, {len(text.split()):3d} words "
          f"| {text[:110]}")

    label, (text, seconds) = "C words+JSON", technique_c()
    results.append({"technique": label, "text": text, "chars": len(text),
                    "words": len(text.split()), "seconds": round(seconds, 2)})
    print(f"  C words+JSON : {len(text):4d} chars, {len(text.split()):3d} words "
          f"| {text[:110]}")

    label, (text, seconds) = "D self-shorten", technique_d()
    results.append({"technique": label, "text": text, "chars": len(text),
                    "words": len(text.split()), "seconds": round(seconds, 2)})
    print(f"  D self-short : {len(text):4d} chars, {len(text.split()):3d} words "
          f"| {text[:110]}")

    cut = technique_e(text)
    results.append({"technique": "E hard cut", "text": cut, "chars": len(cut),
                    "words": len(cut.split()), "seconds": 0.0})
    print(f"  E hard cut   : {len(cut):4d} chars, {len(cut.split()):3d} words "
          f"| {cut[:110]}")

    Path(ROOT / "tools" / "control_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\nwrote tools/control_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
