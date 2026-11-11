#!/usr/bin/env python3
"""MAI-Voice (Microsoft, via OpenRouter) worker using the dubbing JSON-lines protocol.

Opt-in vendor engine (``--tts-engine mai``), approved by the user on 2026-11-11;
never a default. Preset voices only, no cloning. The OpenRouter key is read from
``OPENROUTER_API_KEY`` / ``OPEN_ROUTER`` or ``nemo/.env`` and is never printed.

Per request (``text``, ``voice``, ``language``, ``output`` and optional
``target_seconds``):

1. Synthesize at MAI's native pace (``speed`` 1.0).
2. Silence an isolated non-speech tail (MAI sometimes appends short blips after a
   long silence following the last word); the removed span is recorded.
3. When ``target_seconds`` is given and the speech does not land inside
   [LOW x target, max_end_seconds] (the finisher's own limits), ask MAI itself to speak faster or slower (its own
   ``speed`` parameter, bounded to [MIN_SPEED, MAX_SPEED]) and keep the closest
   take. Audio is never time-stretched here; the shared finisher still applies
   its unchanged 0.90-1.00 pitch-preserving rule and fails for review otherwise.

Writes 24 kHz mono PCM16 WAV plus ``<output>.mai.json`` evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

API = "https://openrouter.ai/api/v1/audio/speech"   # MAI_TTS_API_URL overrides (proxy / offline test)
DEFAULT_MODEL = "microsoft/mai-voice-2.1-flash"
PRICE_PER_MILLION_CHARS = 15.0
LOW, AIM = 0.91, 0.96                    # fraction of the target: accept floor, retry aim
END_SLACK = 0.10                         # default max end when the caller gives only a target
MIN_SPEED, MAX_SPEED = 0.85, 1.15
MAX_ATTEMPTS = 3
ACTIVE_DB = -55.0                        # same activity threshold as the finisher
TAIL_MIN_GAP, TAIL_MAX_RUN, TAIL_MAX_TOTAL = 0.30, 0.25, 0.60

# MAI-Voice-2.1 locales; default voice per language (female where available).
LOCALES = {
    "cs": "cs-CZ", "da": "da-DK", "de": "de-DE", "en": "en-US", "es": "es-MX", "fi": "fi-FI",
    "fr": "fr-FR", "hi": "hi-IN", "hu": "hu-HU", "id": "id-ID", "it": "it-IT", "ko": "ko-KR",
    "nb": "nb-NO", "nl": "nl-NL", "pl": "pl-PL", "pt": "pt-BR", "ro": "ro-RO", "ru": "ru-RU",
    "sv": "sv-SE", "th": "th-TH", "tr": "tr-TR", "vi": "vi-VN", "zh": "zh-CN",
}
DEFAULT_VOICES = {"de": "Mia", "fr": "Soleil", "en": "Olivia", "es": "Valeria", "it": "Rosa",
                  "pt": "Luana", "ko": "Haena", "hi": "Kavya", "zh": "Mei", "tr": "Elif", "ru": "Masha"}
NAMES = {"german": "de", "english": "en", "french": "fr", "spanish": "es", "italian": "it",
         "portuguese": "pt", "dutch": "nl", "polish": "pl", "russian": "ru", "turkish": "tr",
         "chinese": "zh", "korean": "ko", "hindi": "hi", "japanese": "ja"}


def language_code(value: str) -> str:
    code = NAMES.get(value.strip().lower(), value.strip().lower()).split("-")[0]
    if code not in LOCALES:
        raise ValueError(f"Unsupported MAI-Voice language: {value!r}; supported: {' '.join(sorted(LOCALES))}")
    return code


def voice_id(voice: str | None, language: str, model: str = DEFAULT_MODEL) -> str:
    """'Mia' -> 'de-DE-Mia:MAI-Voice-2.1-Flash'; a full voice id is kept as given."""
    code = language_code(language)
    suffix = "MAI-Voice-2" if model.endswith("mai-voice-2-flash") else "MAI-Voice-2.1-Flash"
    name = voice or DEFAULT_VOICES.get(code, "Harper")
    if ":" in name:
        return name
    if not name.count("-") >= 2:
        name = f"{LOCALES[code]}-{name}"
    return f"{name}:{suffix}"


def api_key() -> str:
    for variable in ("OPENROUTER_API_KEY", "OPEN_ROUTER"):
        if os.environ.get(variable):
            return os.environ[variable].strip()
    env = Path(__file__).resolve().parent.parent / "nemo" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("OPENROUTER_API_KEY", "OPEN_ROUTER") and value.strip():
                return value.strip().strip('"').strip("'")
    raise RuntimeError("No OpenRouter key: set OPENROUTER_API_KEY or OPEN_ROUTER in nemo/.env")


def read_wav(path):
    with wave.open(str(path)) as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy(), wav.getframerate()


def write_wav(path, samples, rate):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())


def active_runs(samples, rate, frame=0.02, threshold_db=ACTIVE_DB):
    """[(start_s, end_s)] of contiguous 20 ms frames whose peak exceeds the threshold."""
    step = max(1, int(frame * rate))
    limit = 32768 * 10 ** (threshold_db / 20)
    peaks = [np.max(np.abs(samples[i:i + step].astype(np.int32))) for i in range(0, len(samples), step)]
    runs, begin = [], None
    for k, peak in enumerate(peaks):
        if peak >= limit and begin is None:
            begin = k
        elif peak < limit and begin is not None:
            runs.append((begin * frame, k * frame)); begin = None
    if begin is not None:
        runs.append((begin * frame, len(peaks) * frame))
    return runs


def clean_tail(samples, rate):
    """Silence trailing short bursts isolated from the speech by a long silence.

    Removed only when every trailing burst is <= TAIL_MAX_RUN, they total
    <= TAIL_MAX_TOTAL, and the gap after the last speech run is >= TAIL_MIN_GAP.
    Returns (samples, evidence or None).
    """
    runs = active_runs(samples, rate)
    if len(runs) < 2:
        return samples, None
    tail = []
    for k in range(len(runs) - 1, 0, -1):
        start, end = runs[k]
        if end - start > TAIL_MAX_RUN:
            break
        tail.insert(0, runs[k])
        if start - runs[k - 1][1] >= TAIL_MIN_GAP:
            speech_end = runs[k - 1][1]
            if sum(e - s for s, e in tail) > TAIL_MAX_TOTAL:
                return samples, None
            cut = int(round((speech_end + .05) * rate))
            cleaned = samples.copy(); cleaned[cut:] = 0
            peak = float(np.max(np.abs(samples[cut:].astype(np.int32)))) / 32768
            return cleaned, dict(silenced_from_seconds=round(speech_end + .05, 3), bursts=[[round(s, 3), round(e, 3)] for s, e in tail],
                                 removed_peak=round(peak, 4))
    return samples, None


def speech_end(samples, rate):
    runs = active_runs(samples, rate)
    return runs[-1][1] if runs else 0.0


def synthesize_once(session, key, model, voice, text, speed, tmp):
    body = {"model": model, "input": text, "voice": voice, "response_format": "mp3"}
    if abs(speed - 1.0) > 1e-9:
        body["speed"] = round(speed, 4)
    began = time.perf_counter()
    response = session.post(os.environ.get("MAI_TTS_API_URL", API), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                            json=body, timeout=180)
    seconds = time.perf_counter() - began
    if response.status_code != 200:
        raise RuntimeError(f"MAI API error {response.status_code}: {response.text[:300]}")
    mp3 = Path(tmp) / "take.mp3"; wav = Path(tmp) / "take.wav"
    mp3.write_bytes(response.content)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "24000",
                    "-c:a", "pcm_s16le", str(wav)], check=True, timeout=120)
    samples, rate = read_wav(wav)
    return samples, rate, dict(speed=body.get("speed", 1.0), api_seconds=round(seconds, 3),
                               generation_id=response.headers.get("X-Generation-Id"))


def next_speed(speed, end, target):
    """Speed that should move the end onto AIM x target, using the observed ratio."""
    wanted = speed * end / (AIM * target)
    return min(MAX_SPEED, max(MIN_SPEED, wanted))


def synthesise(session, key, model, request):
    text = request.get("text", "").strip()
    if not text or not request.get("output"):
        raise ValueError("missing text or output")
    language = language_code(request.get("language", "en"))
    voice = voice_id(request.get("voice"), language, model)
    target = request.get("target_seconds")
    max_end = request.get("max_end_seconds") or ((target + END_SLACK) if target else None)
    attempts, best, speed = [], None, 1.0
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(MAX_ATTEMPTS):
            samples, rate, info = synthesize_once(session, key, model, voice, text, speed, tmp)
            samples, tail = clean_tail(samples, rate)
            end = speech_end(samples, rate)
            info.update(speech_end_seconds=round(end, 3), tail_cleaned=tail,
                        est_cost_usd=round(len(text) * PRICE_PER_MILLION_CHARS / 1e6, 6))
            attempts.append(info)
            score = abs(end / target - AIM) if target else 0.0
            if best is None or score < best[0]:
                best = (score, samples, rate, len(attempts) - 1)
            if not target or LOW * target <= end <= max_end:
                break
            proposed = next_speed(speed, end, target)
            if abs(proposed - speed) < 0.005:
                break   # already at a bound; the finisher reports any remaining misfit
            speed = proposed
    _, samples, rate, chosen = best
    output = Path(request["output"])
    write_wav(output, samples, rate)
    evidence = dict(engine="mai", model=model, voice=voice, language=language, characters=len(text),
                    target_seconds=target, max_end_seconds=max_end, accept_floor=LOW, aim=AIM, speed_bounds=[MIN_SPEED, MAX_SPEED],
                    chosen_attempt=chosen, attempts=attempts,
                    est_cost_usd=round(sum(a["est_cost_usd"] for a in attempts), 6))
    output.with_suffix(".mai.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["custom"], default="custom")
    parser.parse_args()
    model = os.environ.get("MAI_TTS_MODEL", DEFAULT_MODEL)
    try:
        import requests
        key = api_key()
        session = requests.Session()
    except Exception as exc:
        print(f"LOAD_ERROR: {exc}", flush=True)
        return 1
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("quit"):
                print(json.dumps({"ok": True}), flush=True)
                break
            evidence = synthesise(session, key, model, request)
            chosen = evidence["attempts"][evidence["chosen_attempt"]]
            print(f"[mai] {evidence['voice']} speed {chosen['speed']} end {chosen['speech_end_seconds']}s "
                  f"target {evidence['target_seconds']} attempts {len(evidence['attempts'])}", file=sys.stderr, flush=True)
            response = {"ok": True}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
