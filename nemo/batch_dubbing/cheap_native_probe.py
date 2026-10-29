"""Four-call bilingual native-speed smoke experiment; no provider imports.

This establishes an explicit 1.0 synthesis baseline, not duration-conditioned
dubbing. Existing subtitle text is test material, never acoustic timing evidence.
"""
from copy import deepcopy
import hashlib
import math

MODEL = "Supertone/supertonic-3"
REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"
IMAGE_ID = "im-40qUTZlx5ttgX1nwrQdvyw"
PACKAGE = "supertonic==1.3.1"
MAX_BYTES = 8 * 1024**2


def validate_input(data):
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise ValueError("Expected probe schema 1")
    for name, expected in (("model", MODEL), ("revision", REVISION), ("package", PACKAGE),
                           ("image_id", IMAGE_ID)):
        if data.get(name) != expected:
            raise ValueError(f"Changed pinned {name}")
    if (type(data.get("speed")) not in (int, float) or data["speed"] != 1.0
            or data.get("steps") != 8 or type(data.get("steps")) is not int):
        raise ValueError("Probe requires speed 1.0 and eight synthesis steps")
    rows = data.get("samples")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Exactly four native samples are reviewed")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each sample must be an object")
    if [row.get("language") for row in rows] != ["en", "de", "en", "de"]:
        raise ValueError("Expected two English/German text pairs")
    for index, row in enumerate(rows):
        if row.get("id") != f"sample_{index}" or row.get("voice") not in ("M1", "M2"):
            raise ValueError("Invalid ordered sample ID or voice")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 80:
            raise ValueError("Each sample must contain 1..80 characters")
    return deepcopy(data)


def run_probe(data, smoke, *, synthesize):
    """One smoke call followed, only after success, by three workload calls.

    Callback returns encoded native WAV bytes and basic measurements. Rejected
    results stop immediately; nothing retries or substitutes a backend.
    """
    data = validate_input(data)
    if type(smoke) is not bool:
        raise ValueError("smoke must be Boolean")
    rows = data["samples"][:1] if smoke else data["samples"][1:]
    result = {"status": "running", "stage": "smoke" if smoke else "workload",
              "input": data, "calls": 0, "samples": [], "audio": {},
              "synthesis_speed": 1.0, "playback_rate": 1.0, "steps": 8,
              "duration_conditioning": False, "pause_conditioning": False,
              "timing_verified": False, "naturalness_verified": False,
              "processing": "Untrimmed native WAV; no fit, stretch, silence insertion or video packaging",
              "quality_limit": "Technical speed-baseline probe only. Subtitle timestamps are not speech evidence; translation and original pauses remain unvalidated."}
    total_bytes = 0
    def measure(row):
        value = synthesize(deepcopy(row))
        if not isinstance(value, dict):
            raise ValueError("Provider must return a measured WAV object")
        content = value.get("audio")
        count, rate = value.get("sample_count"), value.get("sample_rate")
        if (not isinstance(content, bytes) or len(content) < 44
                or not content.startswith(b"RIFF") or content[8:12] != b"WAVE"):
            raise ValueError("Expected native WAV bytes")
        if type(count) is not int or not 0 < count <= 12 * 44100 or type(rate) is not int or rate != 44100:
            raise ValueError("Generated clip exceeds 12 seconds or has wrong sample rate")
        peak = value.get("peak")
        if type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0:
            raise ValueError("Generated clip is silent or nonfinite")
        if total_bytes + len(content) > MAX_BYTES:
            raise ValueError("Artifact transfer exceeds 8 MiB per stage")
        filename = row["id"] + ".wav"
        result["audio"][filename] = content
        result["samples"].append({**row, "file": filename, "sha256": hashlib.sha256(content).hexdigest(),
                                  "sample_rate": rate, "sample_count": count,
                                  "raw_duration_seconds": count / rate, "peak": peak})
        return len(content)
    for row in rows:
        result["calls"] += 1
        try:
            total_bytes += measure(row)
        except Exception as exc:
            result.update(status="failed", failed_sample_id=row["id"], error=str(exc),
                          total_audio_bytes=total_bytes)
            return result
    result.update(status="complete", total_audio_bytes=total_bytes)
    return result
