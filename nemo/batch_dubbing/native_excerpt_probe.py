"""Bounded native speech experiment using caption gaps as listening scaffolding.

Pure stdlib. No model, cloud or audio-library imports. Caption boundaries never
establish acoustic alignment; sample preservation never establishes pronunciation.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from batch_dubbing.cheap_native_probe import MODEL, REVISION, PACKAGE, IMAGE_ID

SAMPLE_RATE = 44100
MAX_AUDIO_BYTES = 63 * 1024**2
MAX_METADATA_BYTES = 1024**2
MAX_CLIP_SECONDS = 120
MAX_STAGE_SECONDS = 300
MAX_API_CALLS = 16
MAX_INTERNAL_CALLS = 24


def digest(content):
    return hashlib.sha256(content).hexdigest()


def _json(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def input_digest(data):
    return digest(_json(data))


def _number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _frame(seconds):
    return round(seconds * SAMPLE_RATE)


def validate_scope(data):
    """Structural preflight without importing the synthesizer's text chunker."""
    if not isinstance(data, dict):
        raise ValueError("Input must be an object")
    for key, expected in (("model", MODEL), ("revision", REVISION), ("package", PACKAGE), ("image_id", IMAGE_ID)):
        if data.get(key) != expected:
            raise ValueError(f"Unexpected {key}")
    if type(data.get("schema")) is not int or data["schema"] != 1:
        raise ValueError("Expected schema 1")
    for key, expected in (("speed", 1.), ("silence_duration", .3)):
        if _number(data.get(key), key) != expected:
            raise ValueError(f"Unexpected {key}")
    for key, expected in (("steps", 8), ("max_chunk_length", 300)):
        if type(data.get(key)) is not int or data[key] != expected:
            raise ValueError(f"Unexpected {key}")
    start, end = (_number(data.get(key), key) for key in ("excerpt_start", "excerpt_end"))
    if not 0 <= start < end <= 86400 or end - start > 120:
        raise ValueError("Excerpt must be positive, within one day, and at most120 seconds")
    if not isinstance(data.get("source_metadata"), dict) or not data["source_metadata"]:
        raise ValueError("Explicit source metadata is required")
    rows = data.get("samples")
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_API_CALLS:
        raise ValueError("Expected two to16 source-linked samples")
    ids, previous, indices = set(), {}, {"en": set(), "de": set()}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["id"]):
            raise ValueError("Invalid sample ID")
        if row["id"] in ids or row["id"] in {"native_en", "native_de", "aligned_en", "aligned_de"}:
            raise ValueError("Duplicate or reserved sample ID")
        ids.add(row["id"])
        language = row.get("language")
        if language not in ("en", "de") or not isinstance(row.get("voice"), str) or not re.fullmatch(r"[FM][1-5]", row["voice"]):
            raise ValueError("Expected English/German and an explicit supported voice preset")
        if type(row.get("seed")) is not int or not 0 <= row["seed"] < 2**32:
            raise ValueError("Explicit seed must be an integer in [0, 2**32)")
        if not isinstance(row.get("speaker"), str) or not 0 < len(row["speaker"].strip()) <= 128:
            raise ValueError("Explicit source speaker is required")
        if not isinstance(row.get("text"), str) or not row["text"].strip() or len(row["text"]) > 5000:
            raise ValueError("Invalid complete source text")
        a, b = (_number(row.get(key), key) for key in ("srt_start", "srt_end"))
        if not start <= a < b <= end or _frame(b - start) <= _frame(a - start):
            raise ValueError("Caption slot must lie inside excerpt and contain at least one frame")
        if a < previous.get(language, start):
            raise ValueError("Caption groups must be ordered and nonoverlapping within each language")
        previous[language] = b
        source_indices = row.get("source_indices")
        if (not isinstance(source_indices, list) or not 1 <= len(source_indices) <= 1000
                or any(type(index) is not int or index < 0 for index in source_indices)
                or len(set(source_indices)) != len(source_indices)
                or indices[language].intersection(source_indices)):
            raise ValueError("Source indices must be unique nonnegative integers within each language")
        indices[language].update(source_indices)
    if rows[0]["language"] != "en" or set(previous) != {"en", "de"}:
        raise ValueError("Smoke must be the first English sample; both languages are required")
    try:
        encoded = _json(data)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Input must be finite JSON") from exc
    if len(encoded) > 128 * 1024:
        raise ValueError("Input metadata exceeds128 KiB")
    return deepcopy(data)


def validate_input(data, *, chunker):
    data = validate_scope(data)
    if not callable(chunker):
        raise ValueError("Actual pinned-package chunker is required")
    count = 0
    for row in data["samples"]:
        chunks = chunker(row["text"], 300)
        if (not isinstance(chunks, (list, tuple)) or not chunks
                or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks)
                or " ".join(chunks) != row["text"]):
            raise ValueError("Chunking changed or dropped source text")
        count += len(chunks)
    if count > MAX_INTERNAL_CALLS:
        raise ValueError("Internal chunk budget exceeds24 calls")
    return data


def plan_jobs(data, smoke, *, chunker):
    if type(smoke) is not bool:
        raise ValueError("smoke must be Boolean")
    data = validate_input(data, chunker=chunker)
    return data["samples"][:1] if smoke else data["samples"][1:]


def validate_samples(payload, sample_rate):
    if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
        raise ValueError("Expected44100 Hz")
    if not isinstance(payload, bytes) or not payload or len(payload) % 4 or len(payload) > MAX_CLIP_SECONDS * SAMPLE_RATE * 4:
        raise ValueError("Invalid native samples or clip exceeds120 seconds")
    peak = 0.
    for (sample,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(sample):
            raise ValueError("Nonfinite native samples")
        peak = max(peak, abs(sample))
    if not peak:
        raise ValueError("Silent native waveform")
    return len(payload) // 4, peak


def encode_wave(payload):
    fmt = struct.pack("<HHIIHH", 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
    body = (b"WAVEfmt " + struct.pack("<I", 16) + fmt
            + b"fact" + struct.pack("<II", 4, len(payload) // 4)
            + b"data" + struct.pack("<I", len(payload)) + payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def decode_wave(wav):
    if not isinstance(wav, bytes) or not 60 <= len(wav) <= MAX_CLIP_SECONDS * SAMPLE_RATE * 4 + 56:
        raise ValueError("Invalid smoke WAV bounds")
    payload = wav[56:]
    validate_samples(payload, SAMPLE_RATE)
    if wav != encode_wave(payload):
        raise ValueError("Smoke WAV format or contents changed")
    return payload


def validate_smoke(artifact, data):
    validate_scope(data)
    if not isinstance(artifact, dict) or set(artifact) != {"report", "report_sha256", "wav"}:
        raise ValueError("Hash-bound smoke artifact required")
    raw = artifact["report"]
    if not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES or digest(raw) != artifact["report_sha256"]:
        raise ValueError("Smoke report integrity failure")
    report = json.loads(raw)
    if (not isinstance(report, dict) or report.get("status") != "complete" or report.get("stage") != "smoke"
            or report.get("input") != data or report.get("input_sha256") != input_digest(data)
            or type(report.get("calls")) is not int or report["calls"] != 1
            or type(report.get("internal_calls")) is not int or not 1 <= report["internal_calls"] <= MAX_INTERNAL_CALLS
            or type(report.get("synthesis_speed")) not in (int, float) or report.get("synthesis_speed") != 1.
            or type(report.get("playback_rate")) not in (int, float) or report.get("playback_rate") != 1.
            or report.get("speech_trimmed") is not False or report.get("waveform_time_stretched") is not False
            or report.get("steps") != 8 or report.get("max_chunk_length") != 300
            or report.get("internal_chunk_silence_seconds") != .3
            or len(report.get("samples", [])) != 1 or report.get("sequences") != []):
        raise ValueError("Successful matching smoke report required")
    record = report["samples"][0]
    row = data["samples"][0]
    if (any(record.get(key) != value for key, value in row.items())
            or record.get("file") != row["id"] + ".wav" or record.get("sha256") != digest(artifact["wav"])):
        raise ValueError("Smoke sample identity/hash mismatch")
    payload = decode_wave(artifact["wav"])
    if (type(record.get("native_frames")) is not int or record["native_frames"] != len(payload) // 4
            or record.get("samples_sha256") != digest(payload)):
        raise ValueError("Smoke sample frames/hash mismatch")
    return report, payload


def load_smoke(path, data):
    validate_scope(data)
    path = Path(path)
    wavpath = path.parent / (data["samples"][0]["id"] + ".wav")
    if path.stat().st_size > MAX_METADATA_BYTES or wavpath.stat().st_size > MAX_CLIP_SECONDS * SAMPLE_RATE * 4 + 56:
        raise ValueError("Oversized smoke artifact")
    raw = path.read_bytes()
    artifact = {"report": raw, "report_sha256": digest(raw), "wav": wavpath.read_bytes()}
    validate_smoke(artifact, data)
    return artifact


def strict_caption_placement(data, samples):
    """Exact frame-count diagnostic; never treats caption slots as speech truth."""
    rows, collisions = [], []
    measured = {row["id"]: row for row in samples}
    previous = {"en": [], "de": []}
    for row in data["samples"]:
        start = _frame(row["srt_start"] - data["excerpt_start"])
        end = _frame(row["srt_end"] - data["excerpt_start"])
        frames = measured[row["id"]]["native_frames"]
        difference = frames - (end - start)
        item = {"id": row["id"], "language": row["language"], "start_frame": start,
                "caption_end_frame": end, "native_end_frame": start + frames,
                "target_frames": end - start, "native_frames": frames,
                "overfill_frames": max(0, difference), "underfill_frames": max(0, -difference),
                "matches_caption_slot_exactly": difference == 0}
        for earlier in previous[row["language"]]:
            overlap = min(earlier["native_end_frame"], start + frames) - start
            if overlap > 0:
                collisions.append({"earlier_id": earlier["id"], "later_id": row["id"],
                                   "overlap_frames": overlap})
        previous[row["language"]].append(item)
        rows.append(item)
    matches = all(row["matches_caption_slot_exactly"] for row in rows)
    return {"status": "exact_caption_slot_match" if matches else "caption_slot_mismatch",
            "all_slots_match_exactly": matches, "rows": rows, "collisions": collisions,
            "timing_verified": False, "evidence_kind": "caption_windows_not_acoustic_speech"}


def assemble_native(data, language, samples, payloads):
    """Copy full raw clips with reference-caption gaps, allowing duration drift."""
    source = [row for row in data["samples"] if row["language"] == language]
    measured = {row["id"]: row for row in samples}
    cursor, previous_end, placements, silence = 0, 0, [], []
    for row in source:
        reference_start = _frame(row["srt_start"] - data["excerpt_start"])
        reference_end = _frame(row["srt_end"] - data["excerpt_start"])
        gap = reference_start - previous_end
        silence.append({"start_frame": cursor, "end_frame": cursor + gap, "basis": "reference_caption_gap"})
        cursor += gap
        frames = measured[row["id"]]["native_frames"]
        placements.append({"id": row["id"], "start_frame": cursor, "end_frame": cursor + frames, "native_frames": frames})
        cursor += frames
        previous_end = reference_end
    tail = _frame(data["excerpt_end"] - data["excerpt_start"]) - previous_end
    silence.append({"start_frame": cursor, "end_frame": cursor + tail, "basis": "reference_caption_tail"})
    cursor += tail
    if cursor * 4 + 56 > MAX_AUDIO_BYTES:
        raise ValueError("Native sequence exceeds audio byte budget")
    output = bytearray(cursor * 4)
    for row in placements:
        payload = payloads[row["id"]]
        a, b = row["start_frame"] * 4, row["end_frame"] * 4
        if len(payload) != b - a:
            raise ValueError("Native sample count changed")
        output[a:b] = payload
        row["source_samples_sha256"] = digest(payload)
        row["sequence_slice_sha256"] = digest(output[a:b])
        if row["source_samples_sha256"] != row["sequence_slice_sha256"]:
            raise ValueError("Native sample contents changed")
    return bytes(output), {"language": language, "kind": "native_listening_sequence",
                          "placements": placements, "silence": silence, "total_frames": cursor,
                          "duration_seconds": cursor / SAMPLE_RATE,
                          "reference_duration_seconds": data["excerpt_end"] - data["excerpt_start"],
                          "sample_content_retention_verified": True, "timing_verified": False,
                          "evidence_kind": "caption_gap_scaffolding_not_acoustic_speech"}


def run_excerpt(data, smoke, *, chunker, synthesize, smoke_artifact=None):
    jobs = plan_jobs(data, smoke, chunker=chunker)
    data = deepcopy(data)
    if smoke and smoke_artifact is not None:
        raise ValueError("Smoke cannot reuse a prior artifact")
    prior = None if smoke else validate_smoke(smoke_artifact, data)
    if prior is not None and prior[0]["internal_calls"] != len(chunker(data["samples"][0]["text"], 300)):
        raise ValueError("Smoke internal call count differs from pinned chunker")
    result = {"status": "running", "stage": "smoke" if smoke else "workload", "input": data,
              "input_sha256": input_digest(data), "calls": 0, "internal_calls": 0,
              "internal_calls_evidence": "expected_chunks_for_dispatched_api_calls_not_observed_completions",
              "samples": [], "sequences": [], "audio": {}, "generated_seconds": 0.,
              "synthesis_speed": 1., "playback_rate": 1., "speech_trimmed": False,
              "waveform_time_stretched": False, "duration_conditioning": False, "pause_conditioning": False,
              "timing_verified": False, "pronunciation_verified": False, "naturalness_verified": False,
              "steps": 8, "max_chunk_length": 300, "internal_chunk_silence_seconds": .3,
              "gap_basis": "existing_caption_lead_interblock_tail_gaps_only_not_intrablock_or_acoustic_speech",
              "aligned_timeline_files": {},
              "strict_caption_placement": None,
              "quality_limit": "Native waveform retention does not prove pronounced endings, meaning fidelity, naturalness or acoustic source alignment."}
    payloads = {}
    if prior is not None:
        report, payload = prior
        record = deepcopy(report["samples"][0])
        record["reused_from_smoke"] = True
        result["samples"].append(record)
        result["audio"][record["file"]] = smoke_artifact["wav"]
        result["smoke_report_sha256"] = smoke_artifact["report_sha256"]
        payloads[record["id"]] = payload
    generated_frames = 0
    try:
        for row in jobs:
            result["active_sample_id"] = row["id"]
            result["calls"] += 1
            result["internal_calls"] += len(chunker(row["text"], 300))
            measured = synthesize(deepcopy(row))
            if not isinstance(measured, dict):
                raise ValueError("Invalid callback output")
            payload = measured.get("samples")
            frames, peak = validate_samples(payload, measured.get("sample_rate"))
            if generated_frames + frames > MAX_STAGE_SECONDS * SAMPLE_RATE:
                raise ValueError("Generated stage duration exceeds300 seconds")
            wav = encode_wave(payload)
            if sum(map(len, result["audio"].values())) + len(wav) > MAX_AUDIO_BYTES:
                raise ValueError("Stage audio transfer exceeds63 MiB")
            generated_frames += frames
            filename = row["id"] + ".wav"
            result["audio"][filename] = wav
            payloads[row["id"]] = payload
            result["samples"].append({**deepcopy(row), "file": filename, "sha256": digest(wav),
                                      "samples_sha256": digest(payload), "native_frames": frames,
                                      "sample_rate": SAMPLE_RATE, "raw_duration_seconds": frames / SAMPLE_RATE,
                                      "peak": peak, "reused_from_smoke": False})
            result["generated_seconds"] = generated_frames / SAMPLE_RATE
        result.pop("active_sample_id", None)
        if not smoke:
            strict = strict_caption_placement(data, result["samples"])
            result["strict_caption_placement"] = strict
            for language in ("en", "de"):
                payload, sequence = assemble_native(data, language, result["samples"], payloads)
                wav = encode_wave(payload)
                if sum(map(len, result["audio"].values())) + len(wav) > MAX_AUDIO_BYTES:
                    raise ValueError("Sequences exceed63 MiB audio transfer bound")
                filename = f"native_{language}.wav"
                result["audio"][filename] = wav
                result["sequences"].append({**sequence, "file": filename, "sha256": digest(wav)})
                if strict["all_slots_match_exactly"]:
                    # The native sequence already occupies exactly these slots.
                    # Refer to that artifact instead of duplicating its bytes.
                    result["aligned_timeline_files"][language] = filename
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["audio_transfer_bytes"] = sum(map(len, result["audio"].values()))
    result["metadata_reserve_bytes"] = MAX_METADATA_BYTES
    if len(_json({key: value for key, value in result.items() if key != "audio"})) > MAX_METADATA_BYTES:
        raise ValueError("Report exceeds1 MiB metadata bound")
    return result
