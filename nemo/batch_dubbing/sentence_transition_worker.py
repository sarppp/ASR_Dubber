"""Prepared, serial cached-CPU transition probe. Importing this declares a Modal app.

Pure helpers accept synthetic float32 sample bytes for tests; no model is loaded
until the explicitly released remote function runs. Audio is never fitted.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import modal

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dubbing_naturalness_2026-10-28/sentence_transition_trial"
image = (modal.Image.from_id("im-40qUTZlx5ttgX1nwrQdvyw")
         .add_local_file(ROOT / "batch_dubbing/cheap_native_probe.py", "/work/batch_dubbing/cheap_native_probe.py")
         .add_local_file(ROOT / "batch_dubbing/sentence_transition_probe.py", "/work/batch_dubbing/sentence_transition_probe.py"))
volume = modal.Volume.from_name("speech-model-benchmark-20261027")
app = modal.App("sentence-transition-native-probe")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def fixture_digest(data):
    return digest(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def validate_samples(payload, sample_rate=44100):
    """Validate full mono little-endian float32 samples, without changing them."""
    if type(sample_rate) is not int or sample_rate != 44100:
        raise ValueError("Expected 44100 Hz")
    if not isinstance(payload, bytes) or not payload or len(payload) % 4 or len(payload) > 90 * 44100 * 4:
        raise ValueError("Invalid samples or native clip exceeds 90 seconds")
    peak = 0.0
    for (value,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(value):
            raise ValueError("Nonfinite native audio")
        peak = max(peak, abs(value))
    if peak == 0:
        raise ValueError("Silent native clip")
    return len(payload) // 4, peak


def encode_wave(payload):
    """Lossless IEEE float WAV; no resampling, normalization, fades or trimming."""
    count = len(payload) // 4
    fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
    body = (b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
            + b"fact" + struct.pack("<II", 4, count)
            + b"data" + struct.pack("<I", len(payload)) + payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def decode_wave(content):
    """Accept only the worker's exact float WAV representation, including bounds."""
    if not isinstance(content, bytes) or len(content) < 60 or len(content) > 90 * 44100 * 4 + 56:
        raise ValueError("Invalid or overlong smoke WAV")
    payload = content[56:]
    validate_samples(payload)
    if content != encode_wave(payload):
        raise ValueError("Smoke WAV must be intact 44100 Hz mono float32 output")
    return payload


def validate_smoke(artifact, data):
    """Verify scope, report binding and WAV hashes before any new synthesis."""
    if not isinstance(artifact, dict) or set(artifact) != {"report", "report_sha256", "wav"}:
        raise ValueError("Matching smoke report and WAV are required")
    report_bytes, wav = artifact["report"], artifact["wav"]
    if not isinstance(report_bytes, bytes) or len(report_bytes) > 1024 * 1024 or digest(report_bytes) != artifact["report_sha256"]:
        raise ValueError("Smoke report integrity failure")
    report = json.loads(report_bytes)
    if (not isinstance(report, dict) or report.get("status") != "complete" or report.get("stage") != "smoke"
            or report.get("input") != data or report.get("input_sha256") != fixture_digest(data)
            or type(report.get("calls")) is not int or report.get("calls") != 1
            or type(report.get("internal_calls")) is not int or report.get("internal_calls") != 1
            or type(report.get("synthesis_speed")) not in (int, float) or report.get("synthesis_speed") != 1.
            or type(report.get("playback_rate")) not in (int, float) or report.get("playback_rate") != 1.
            or report.get("waveform_time_stretched") is not False or report.get("speech_trimmed") is not False
            or report.get("steps") != 8 or report.get("max_chunk_length") != 300
            or report.get("internal_chunk_silence_seconds") != .3
            or report.get("synthetic_join_silence_seconds") != .4
            or len(report.get("samples", [])) != 1 or report.get("sequences") != []):
        raise ValueError("Smoke report is not a successful matching single-call trial")
    record = report["samples"][0]
    expected = data["samples"][0]
    if (any(record.get(key) != value for key, value in expected.items())
            or record.get("file") != "sample_0.wav" or record.get("sha256") != digest(wav)):
        raise ValueError("Smoke sample identity or WAV hash mismatch")
    payload = decode_wave(wav)
    if record.get("native_frames") != len(payload) // 4 or record.get("samples_sha256") != digest(payload):
        raise ValueError("Smoke frame count or sample hash mismatch")
    return report, payload


def load_smoke(path, data):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Oversized smoke report")
    report_bytes = path.read_bytes()
    wav_path = path.parent / "sample_0.wav"
    if wav_path.stat().st_size > 90 * 44100 * 4 + 56:
        raise ValueError("Oversized smoke WAV")
    artifact = {"report": report_bytes, "report_sha256": digest(report_bytes), "wav": wav_path.read_bytes()}
    validate_smoke(artifact, data)
    return artifact


def assemble_sequence(rows, payloads):
    from batch_dubbing.sentence_transition_probe import plan_sequence
    plan = plan_sequence(rows)
    if plan["status"] != "ready":
        raise ValueError("Native sequence cannot be placed without loss")
    # Placement is an exact copy into zeros. Full waveforms include their own
    # native leading/trailing silence; fixture gaps are additional silence.
    result = bytearray(plan["total_frames"] * 4)
    for row in plan["schedule"]:
        source = payloads[row["id"]]
        start, end = row["start_frame"] * 4, row["end_frame"] * 4
        if len(source) != end - start:
            raise ValueError("Native sample count changed before assembly")
        result[start:end] = source
        if bytes(result[start:end]) != source:
            raise ValueError("Native samples changed during assembly")
        row["source_samples_sha256"] = digest(source)
        row["sequence_slice_sha256"] = digest(result[start:end])
    plan["sample_content_retention_verified"] = True
    return bytes(result), plan


def run_stage(data, smoke, *, chunker, synthesize, smoke_artifact=None):
    """Serial bounded callback runner; retain completed attempts on any failure."""
    from copy import deepcopy
    from batch_dubbing.sentence_transition_probe import MAX_BYTES, plan_jobs
    # Reserve one MiB for report/transport metadata within the 64 MiB ceiling.
    audio_budget = MAX_BYTES - 1024 * 1024
    jobs = plan_jobs(data, smoke, chunker=chunker)
    if smoke and smoke_artifact is not None:
        raise ValueError("Smoke cannot consume a prior artifact")
    prior = None if smoke else validate_smoke(smoke_artifact, data)
    result = {"status": "running", "stage": "smoke" if smoke else "workload",
              "input": deepcopy(data), "input_sha256": fixture_digest(data),
              "calls": 0, "internal_calls": 0,
              "internal_calls_evidence": "Expected chunks dispatched through synthesis calls; actual internal completion is unknown if a call fails.",
              "samples": [], "sequences": [], "audio": {},
              "generated_seconds": 0., "synthesis_speed": 1., "playback_rate": 1.,
              "waveform_time_stretched": False, "speech_trimmed": False,
              "steps": 8, "max_chunk_length": 300, "internal_chunk_silence_seconds": .3,
              "synthetic_join_silence_seconds": .4, "duration_conditioning": False,
              "pause_conditioning": False, "timing_verified": False,
              "pronunciation_verified": False, "naturalness_verified": False,
              "evidence_kind": "synthetic_fixture", "quality_limit": data["quality_limit"]}
    payloads = {}
    if prior is not None:
        prior_report, payload = prior
        result["samples"].append(deepcopy(prior_report["samples"][0]))
        result["samples"][0]["reused_from_smoke"] = True
        result["audio"]["sample_0.wav"] = smoke_artifact["wav"]
        result["smoke_report_sha256"] = smoke_artifact["report_sha256"]
        payloads["sample_0"] = payload
    frames = 0
    try:
        for row in jobs:
            result["active_sample_id"] = row["id"]
            result["calls"] += 1
            result["internal_calls"] += row["expected_chunks"]
            measured = synthesize(deepcopy(row))
            if not isinstance(measured, dict):
                raise ValueError("Synthesis callback returned invalid output")
            payload = measured.get("samples")
            count, peak = validate_samples(payload, measured.get("sample_rate"))
            if frames + count > 300 * 44100:
                raise ValueError("Stage generated audio exceeds 300 seconds")
            wav = encode_wave(payload)
            if sum(map(len, result["audio"].values())) + len(wav) > audio_budget:
                raise ValueError("Stage transfer exceeds 64 MiB")
            frames += count
            filename = row["id"] + ".wav"
            result["audio"][filename] = wav
            payloads[row["id"]] = payload
            result["samples"].append({**deepcopy(row), "file": filename, "sha256": digest(wav),
                                      "samples_sha256": digest(payload), "native_frames": count,
                                      "sample_rate": 44100, "raw_duration_seconds": count / 44100,
                                      "peak": peak, "reused_from_smoke": False})
            result["generated_seconds"] = frames / 44100
        result.pop("active_sample_id", None)
        if not smoke:
            for language in data["language_order"]:
                rows = [{"id": row["id"], "native_frames": row["native_frames"]}
                        for row in result["samples"] if row["language"] == language]
                # Check bytes before allocating combined output; retain raw clips
                # even if their duplication in sequences exceeds transfer budget.
                expected_bytes = (sum(row["native_frames"] for row in rows) + 2 * 17640) * 4 + 56
                if sum(map(len, result["audio"].values())) + expected_bytes > audio_budget:
                    raise ValueError("Sequences would exceed 64 MiB transfer bound")
                samples, plan = assemble_sequence(rows, payloads)
                wav = encode_wave(samples)
                filename = f"sequence_{language}.wav"
                result["audio"][filename] = wav
                result["sequences"].append({"language": language, "file": filename,
                                             "sha256": digest(wav), "samples_sha256": digest(samples), **plan})
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["audio_transfer_bytes"] = sum(map(len, result["audio"].values()))
    result["metadata_reserve_bytes"] = 1024 * 1024
    result["transfer_limit_bytes"] = MAX_BYTES
    return result


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=150,
              startup_timeout=30, scaledown_window=2, min_containers=0,
              max_containers=1, retries=0, volumes={"/cache": volume})
def probe(data, smoke=False, smoke_artifact=None):
    import os
    import sys
    import time
    from importlib.metadata import version
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    sys.path.insert(0, "/work")
    from batch_dubbing.cheap_native_probe import MODEL, REVISION
    from batch_dubbing.sentence_transition_probe import plan_jobs
    if version("supertonic") != "1.3.1":
        raise ValueError("Cached image has unexpected Supertonic version")
    import numpy as np
    from huggingface_hub import snapshot_download
    from supertonic import TTS
    from supertonic.utils import chunk_text
    plan_jobs(data, smoke, chunker=chunk_text)
    if not smoke:
        validate_smoke(smoke_artifact, data)
    print('MC_EVENT ' + json.dumps({"kind": "worker_ready", "hardware": "CPU", "max_containers": 1}), flush=True)
    started = time.monotonic()
    model_dir = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    model = TTS(model="supertonic-3", model_dir=model_dir, auto_download=False,
                intra_op_num_threads=2, inter_op_num_threads=1)

    def synthesize(row):
        np.random.seed(row["seed"])
        audio, _ = model.synthesize(row["text"], voice_style=model.get_voice_style(row["voice"]),
                                    lang=row["language"], speed=1.0, total_steps=8,
                                    max_chunk_length=300, silence_duration=0.3)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1 or not 0 < audio.size <= 90 * 44100:
            raise ValueError("Invalid or overlong native audio")
        print('MC_EVENT ' + json.dumps({"kind": "segment_complete", "id": row["id"]}), flush=True)
        return {"samples": audio.astype("<f4", copy=False).tobytes(), "sample_rate": model.sample_rate}

    result = run_stage(data, smoke, chunker=chunk_text, synthesize=synthesize, smoke_artifact=smoke_artifact)
    result["worker_seconds"] = time.monotonic() - started
    return result


def save_result(result, destination):
    """Persist valid partial attempts before surfacing a failed stage."""
    from batch_dubbing.sentence_transition_probe import MAX_BYTES
    audio = result.pop("audio")
    report_json = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    expected = {row["file"]: row["sha256"] for row in result["samples"] + result["sequences"]}
    if (not isinstance(audio, dict) or set(audio) != set(expected)
            or sum(map(len, audio.values())) > MAX_BYTES - 1024 * 1024
            or len(report_json.encode("utf-8")) > 1024 * 1024
            or any(Path(name).name != name or not name.endswith(".wav")
                   or digest(content) != expected[name] for name, content in audio.items())):
        raise ValueError("Returned audio integrity failure")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in audio.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    with (destination / "report.json").open("x") as handle:
        handle.write(report_json)
    print("RESULT", destination, flush=True)
    if result.get("status") != "complete":
        raise RuntimeError("Probe failed; partial audio and report retained; no retry")


@app.local_entrypoint()
def main(smoke: bool = False, smoke_result: str = ""):
    import sys
    sys.path.insert(0, str(ROOT))
    from batch_dubbing.sentence_transition_probe import build_fixture
    data = build_fixture()
    destination = OUTPUT / ("smoke" if smoke else "workload")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if smoke and smoke_result:
        raise ValueError("Smoke cannot consume another smoke result")
    if not smoke and not smoke_result:
        raise ValueError("Workload requires --smoke-result report.json")
    artifact = None if smoke else load_smoke(smoke_result, data)
    result = probe.remote(data, smoke, artifact)
    save_result(result, destination)
