"""OpenRouter Nemotron probe; network execution requires explicit hash confirmation.

This evaluates recognized text and timestamps, not whether TTS actually omitted
speech, speaker attribution, naturalness, or exact acoustic silence boundaries.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import difflib
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import struct
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request

MODEL = "nvidia/nemotron-3.5-transcribe"
ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
# 16 MB becomes ~21.34 MB of base64, leaving room below the 25 MB inline cap.
MAX_FILE_BYTES = 16_000_000
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULTS = dict(version=1, model=MODEL, endpoint=ENDPOINT, budget_usd=.10,
                price_per_second_usd=.0000333, max_requests=2, max_audio_seconds=180,
                max_file_bytes=MAX_FILE_BYTES, timeout_seconds=60, max_retries=0,
                temperature=0, response_format="verbose_json",
                timestamp_granularities=["word", "segment"])
LIMITATION = ("ASR differences may be recognition errors; they do not establish actual speech deletion. "
              "Matching text does not establish naturalness, exact pauses, or speaker correctness.")


def _number(value, name):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be positive and finite")
    return value


def normalize_words(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", unicodedata.normalize("NFKC", text).casefold())


def prepare_manifest(inputs):
    """Freeze measured metadata; input files need not exist at preparation time."""
    return validate_manifest(dict(deepcopy(DEFAULTS), inputs=deepcopy(inputs)))


def validate_manifest(manifest):
    """Validate the complete bounded scope without filesystem or network access."""
    if not isinstance(manifest, dict) or set(manifest) != set(DEFAULTS) | {"inputs"}:
        raise ValueError("Unexpected or missing manifest fields")
    for name, expected in DEFAULTS.items():
        value = manifest[name]
        if isinstance(expected, (int, float)):
            if name in ("max_retries", "temperature"):
                if type(value) is not int or value != 0:
                    raise ValueError(f"{name} must be 0")
            else:
                _number(value, name)
        if value != expected:
            raise ValueError(f"Frozen {name} changed")
    inputs = manifest["inputs"]
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 2:
        raise ValueError("One or two inputs are required")
    ids, paths, duration = set(), set(), 0.0
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"id", "path", "sha256", "duration_seconds", "language", "expected_text"}:
            raise ValueError("Unexpected or missing input fields")
        identifier = item["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier) or identifier in ids:
            raise ValueError("Input IDs must be unique safe names")
        ids.add(identifier)
        path = item["path"]
        if (not isinstance(path, str) or not path or len(path) > 4096 or "\x00" in path
                or "://" in path or Path(path).suffix.lower() != ".wav" or path in paths):
            raise ValueError("Input paths must be distinct local WAV files")
        paths.add(path)
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Input SHA256 must be pinned")
        duration += _number(item["duration_seconds"], "duration_seconds")
        if item["language"] not in ("en", "de"):
            raise ValueError("Only English and German inputs are authorized")
        expected = item["expected_text"]
        if not isinstance(expected, str) or len(expected) > 50000 or not 1 <= len(normalize_words(expected)) <= 2048:
            raise ValueError("Expected text must contain 1 to 2048 words")
    if duration > 180:
        raise ValueError("Total audio exceeds 180 seconds")
    if duration * manifest["price_per_second_usd"] > manifest["budget_usd"]:
        raise ValueError("Estimated cost exceeds reviewed budget")
    return deepcopy(manifest)


def manifest_sha256(manifest):
    validated = validate_manifest(manifest)
    return hashlib.sha256(json.dumps(validated, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _wav_metadata(data):
    """Inspect bounded RIFF PCM/float data, including truncated-chunk detection."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Input is not a RIFF WAV")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("WAV RIFF length mismatch")
    offset, fmt, data_size = 12, None, None
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Truncated WAV chunk header")
        tag, size = struct.unpack_from("<4sI", data, offset)
        start, end = offset + 8, offset + 8 + size
        if end + (size % 2) > len(data):
            raise ValueError("Truncated WAV chunk")
        if tag == b"fmt ":
            if fmt is not None or size < 16:
                raise ValueError("Invalid WAV format chunk")
            fmt = struct.unpack_from("<HHIIHH", data, start)
        elif tag == b"data":
            if data_size is not None:
                raise ValueError("Multiple WAV data chunks")
            data_size = size
        offset = end + (size % 2)
    if fmt is None or not data_size:
        raise ValueError("WAV is missing format or nonempty audio data")
    encoding, channels, rate, byte_rate, block, bits = fmt
    if (encoding not in (1, 3) or channels not in (1, 2) or not 8000 <= rate <= 192000
            or bits not in ((16, 24, 32) if encoding == 1 else (32,))
            or block != channels * bits // 8 or byte_rate != rate * block or data_size % block):
        raise ValueError("Unsupported or inconsistent WAV encoding")
    return dict(duration_seconds=data_size / block / rate, sample_rate=rate,
                channels=channels, sample_bits=bits, encoding=encoding)


def _read_input(item):
    with Path(item["path"]).open("rb") as handle:
        if os.fstat(handle.fileno()).st_size > MAX_FILE_BYTES:
            raise ValueError("Input exceeds 16 MB")
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Input exceeds 16 MB")
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("Input hash changed")
    measured = _wav_metadata(data)
    if abs(measured["duration_seconds"] - item["duration_seconds"]) > 1e-9:
        raise ValueError("Input duration changed")
    return data, measured


def prepare_requests(manifest):
    """Preflight ALL files, then build serial JSON requests from verified bytes.

    This does not contact OpenRouter, read credentials, or execute anything.
    The caller must enforce manifest timeout/retry/request/budget limits and
    persist responses. Provider order/only/ignore are not supported by this API.
    """
    manifest = validate_manifest(manifest)
    verified = []
    paths = set()
    for item in manifest["inputs"]:
        resolved = Path(item["path"]).resolve()
        if resolved in paths:
            raise ValueError("Multiple inputs refer to the same file")
        paths.add(resolved)
        verified.append(_read_input(item))
    measured_total = sum(metadata["duration_seconds"] for _, metadata in verified)
    if measured_total > manifest["max_audio_seconds"]:
        raise ValueError("Measured total audio exceeds 180 seconds")
    return [dict(id=item["id"], endpoint=ENDPOINT, input_metadata=metadata,
                 input_sha256=item["sha256"],
                 estimated_cost_usd=metadata["duration_seconds"] * manifest["price_per_second_usd"],
                 json=dict(model=MODEL, input_audio=dict(data=base64.b64encode(data).decode("ascii"), format="wav"),
                           language=item["language"], temperature=0, response_format="verbose_json",
                           timestamp_granularities=["word", "segment"]))
            for item, (data, metadata) in zip(manifest["inputs"], verified)]


def prepare_transition_manifest(workload_report_path):
    """Verify saved transition output and derive the exact English/German scope.

    No Modal worker imports, synthesis, audio rewrites, credentials, or network.
    The sequence must contain each full native sample, in fixture order, with
    exactly the fixture's zero-filled joins. Existing audio is only inspected.
    """
    from batch_dubbing.sentence_transition_probe import build_fixture, plan_sequence
    path = Path(workload_report_path).resolve()
    with path.open("rb") as handle:
        encoded = handle.read(1024 * 1024 + 1)
    if len(encoded) > 1024 * 1024:
        raise ValueError("Transition report exceeds one MiB")
    report, fixture = json.loads(encoded), build_fixture()
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if (not isinstance(report, dict) or canonical(report.get("input")) != canonical(fixture)
            or report.get("input_sha256") != hashlib.sha256(canonical(fixture).encode()).hexdigest()):
        raise ValueError("Transition fixture or fixture hash changed")
    required = dict(status="complete", stage="workload", calls=5, internal_calls=7,
                    synthesis_speed=1.0, playback_rate=1.0, waveform_time_stretched=False,
                    speech_trimmed=False, steps=8, max_chunk_length=300,
                    internal_chunk_silence_seconds=.3, synthetic_join_silence_seconds=.4)
    if any(canonical(report.get(key)) != canonical(value) for key, value in required.items()):
        raise ValueError("Transition report is not a matching successful native workload")
    records, sequences = report.get("samples"), report.get("sequences")
    if (not isinstance(records, list) or len(records) != 6
            or not isinstance(sequences, list) or len(sequences) != 2):
        raise ValueError("Expected six raw samples and two language sequences")
    total_bytes = 0

    def read_audio(row, filename, frame_key):
        nonlocal total_bytes
        if not isinstance(row, dict) or row.get("file") != filename:
            raise ValueError("Unexpected transition audio filename")
        audio_path = (path.parent / filename).resolve()
        if not audio_path.is_relative_to(path.parent):
            raise ValueError("Transition audio escapes its result directory")
        frames = row.get(frame_key)
        if type(frames) is not int or frames <= 0:
            raise ValueError("Invalid transition native frame count")
        item = dict(path=str(audio_path), sha256=row.get("sha256"), duration_seconds=frames / 44100)
        data, metadata = _read_input(item)
        total_bytes += len(data)
        if total_bytes > 63 * 1024**2 or any(metadata[key] != value for key, value in
                (("encoding", 3), ("sample_rate", 44100), ("channels", 1), ("sample_bits", 32))):
            raise ValueError("Unexpected transition WAV format or aggregate size")
        # Worker writes fmt/fact/data in this exact representation; no extra data may hide at the end.
        fmt = struct.pack("<HHIIHH", 3, 1, 44100, 176400, 4, 32)
        header = (b"RIFF" + struct.pack("<I", frames * 4 + 48) + b"WAVEfmt " + struct.pack("<I", 16)
                  + fmt + b"fact" + struct.pack("<II", 4, frames) + b"data" + struct.pack("<I", frames * 4))
        payload = data[56:]
        if data[:56] != header or hashlib.sha256(payload).hexdigest() != row.get("samples_sha256"):
            raise ValueError("Transition sample bytes or WAV header changed")
        return item, payload

    payloads = {}
    for index, (row, expected) in enumerate(zip(records, fixture["samples"])):
        if (not isinstance(row, dict) or any(canonical(row.get(k)) != canonical(v) for k, v in expected.items())
                or row.get("reused_from_smoke") is not (index == 0)):
            raise ValueError("Raw samples differ from ordered fixture identities")
        _, payload = read_audio(row, expected["id"] + ".wav", "native_frames")
        if row.get("sample_rate") != 44100 or row.get("raw_duration_seconds") != len(payload) / 176400:
            raise ValueError("Raw sample duration metadata changed")
        payloads[expected["id"]] = payload
    inputs = []
    for language, sequence in zip(fixture["language_order"], sequences):
        rows = [row for row in records if row["language"] == language]
        expected = plan_sequence([dict(id=row["id"], native_frames=row["native_frames"]) for row in rows])
        expected["sample_content_retention_verified"] = True
        for entry in expected["schedule"]:
            sample_hash = hashlib.sha256(payloads[entry["id"]]).hexdigest()
            entry.update(source_samples_sha256=sample_hash, sequence_slice_sha256=sample_hash)
        if (not isinstance(sequence, dict) or sequence.get("language") != language
                or any(canonical(sequence.get(k)) != canonical(v) for k, v in expected.items())):
            raise ValueError("Sequence schedule, order or retention evidence changed")
        item, payload = read_audio(sequence, f"sequence_{language}.wav", "total_frames")
        cursor = 0
        for entry in expected["schedule"]:
            start, end = entry["start_frame"] * 4, entry["end_frame"] * 4
            if any(payload[cursor:start]) or payload[start:end] != payloads[entry["id"]]:
                raise ValueError("Sequence changed native samples or fixture silence")
            cursor = end
        inputs.append(dict(item, id=f"sequence_{language}", language=language,
                           expected_text=" ".join(row["text"] for row in rows)))
    return prepare_manifest(inputs)


def _text_comparison(expected, observed):
    wanted, heard = normalize_words(expected), normalize_words(observed)
    if len(heard) > 4096:
        raise ValueError("Response exceeds text comparison limit")
    edits = []
    for kind, a, b, c, d in difflib.SequenceMatcher(None, wanted, heard, autojunk=False).get_opcodes():
        if kind != "equal":
            edits.append(dict(kind=kind, expected=wanted[a:b], recognized=heard[c:d],
                              expected_range=[a, b], recognized_range=[c, d]))
    tail = wanted[-5:]
    return dict(normalized_exact_match=wanted == heard, terminal_expected_words=tail,
                terminal_words_match=heard[-len(tail):] == tail, differences=edits,
                interpretation=LIMITATION)


def inspect_response(response, item, *, headers=None):
    """Flag malformed/overlapping timestamps instead of repairing or clamping."""
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("Response requires transcript text")
    if not response["text"].strip() or len(response["text"]) > 50000:
        raise ValueError("Response transcript is empty or oversized")
    errors, billing_errors = [], []
    usage_cost = None
    usage = response.get("usage")
    if usage is not None and not isinstance(usage, dict):
        billing_errors.append("usage must be an object")
    elif isinstance(usage, dict) and "cost" in usage:
        cost = usage["cost"]
        try:
            valid_cost = not isinstance(cost, bool) and isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0
        except OverflowError:
            valid_cost = False
        if valid_cost:
            usage_cost = cost
        else:
            billing_errors.append("usage.cost must be finite and nonnegative")
    generation_id = None
    if headers is not None:
        if not hasattr(headers, "items"):
            billing_errors.append("Response headers must be a mapping")
        else:
            identifiers = [value for key, value in headers.items() if str(key).lower() == "x-generation-id"]
            if identifiers:
                candidate = identifiers[0]
                if (len(identifiers) == 1 and isinstance(candidate, str)
                        and re.fullmatch(r"[A-Za-z0-9._:-]{1,256}", candidate)):
                    generation_id = candidate
                else:
                    billing_errors.append("Invalid or duplicate X-Generation-Id header")
    for kind, text_field in (("words", "word"), ("segments", "text")):
        entries = response.get(kind)
        if not isinstance(entries, list) or not 1 <= len(entries) <= 8192:
            errors.append(f"{kind}: missing, empty, or oversized")
            continue
        previous_end = 0.0
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get(text_field), str) or not entry[text_field].strip():
                errors.append(f"{kind}[{i}]: invalid text")
                continue
            start, end = entry.get("start"), entry.get("end")
            try:
                finite = all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end))
            except OverflowError:
                finite = False
            if not finite or not 0 <= start < end <= item["duration_seconds"]:
                errors.append(f"{kind}[{i}]: invalid timestamp range")
                continue
            if start < previous_end:
                errors.append(f"{kind}[{i}]: overlapping or unordered timestamps")
            previous_end = max(previous_end, end)
    comparison = _text_comparison(item["expected_text"], response["text"])
    for kind, field, label in (("words", "word", "word"), ("segments", "text", "segment")):
        if isinstance(response.get(kind), list):
            combined = " ".join(e.get(field, "") for e in response[kind]
                                if isinstance(e, dict) and isinstance(e.get(field), str))
            if normalize_words(combined) != normalize_words(response["text"]):
                errors.append(f"{label} text differs from transcript text")
    return dict(status="invalid_response" if errors or billing_errors else "pending_review",
                timestamp_errors=errors, text_comparison=comparison,
                usage_cost_usd=usage_cost, generation_id=generation_id,
                billing_errors=billing_errors,
                accounting_complete=usage_cost is not None and generation_id is not None,
                naturalness_verified=False, speech_deletion_verified=False,
                speaker_attribution_available=False)


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, dict):
        return {_redact(str(k), secret): _redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secret) for v in value]
    return value


def _save(path, value, secret):
    temporary = path.with_suffix(".tmp")
    encoded = json.dumps(_redact(value, secret), ensure_ascii=False, indent=2, allow_nan=False)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    os.replace(temporary, path)


def execute_manifest(manifest, *, transport, api_key, output_dir, confirmation):
    """Call a supplied transport serially after confirmation and complete preflight.

    transport(request, *, api_key, timeout_seconds) returns an envelope with
    status_code, headers, and body (a JSON string). Only request['json'] is sent.
    Injected transports must honor timeout_seconds; http_transport enforces it.
    """
    manifest = validate_manifest(manifest)
    identity = manifest_sha256(manifest)
    if confirmation != identity:
        raise ValueError("Exact manifest SHA256 confirmation is required")
    if not isinstance(api_key, str) or not api_key or not api_key.isascii() or any(c.isspace() for c in api_key):
        raise ValueError("A nonempty API key is required")
    if not callable(transport):
        raise ValueError("An explicit transport is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version=1, status="preflight", manifest_sha256=identity,
                  manifest=manifest, attempts=[], naturalness_verified=False,
                  speech_deletion_verified=False, limitation=LIMITATION,
                  estimated_cost_usd=sum(i["duration_seconds"] for i in manifest["inputs"]) * manifest["price_per_second_usd"],
                  known_usage_cost_usd=0.0, actual_cost_complete=True, receipts_complete=True,
                  provider_note="OpenRouter transcription ignores provider order/only/ignore; the backend cannot be pinned.",
                  budget_note="Review ceiling is not a provider-enforced billing cap; failed or timed-out calls may still be billed.")
    _save(output / "report.json", report, api_key)
    try:
        requests = prepare_requests(manifest)
        report["status"] = "running"
        _save(output / "report.json", report, api_key)
        for item, request in zip(manifest["inputs"], requests):
            remaining_estimate = sum(r["estimated_cost_usd"] for r in requests[len(report["attempts"]):])
            if report["known_usage_cost_usd"] + remaining_estimate > manifest["budget_usd"]:
                report.update(status="blocked", stop_reason="Remaining estimate exceeds reviewed budget")
                break
            attempt = dict(id=item["id"], status="started", input_metadata=request["input_metadata"],
                           input_sha256=item["sha256"], started_at=time.time())
            report["attempts"].append(attempt)
            report["actual_cost_complete"] = False
            report["receipts_complete"] = False
            _save(output / "report.json", report, api_key)
            started = time.monotonic()
            response = transport(request, api_key=api_key, timeout_seconds=manifest["timeout_seconds"])
            attempt["elapsed_seconds"] = time.monotonic() - started
            if not isinstance(response, dict) or not isinstance(response.get("body"), str):
                raise ValueError("Transport must return a response body string")
            body = response["body"]
            if len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise ValueError("Response exceeds 4 MiB")
            response_path = output / (item["id"] + ".response.txt")
            with response_path.open("x", encoding="utf-8") as handle:
                handle.write(body.replace(api_key, "[REDACTED]"))
            attempt["response_file"] = response_path.name
            attempt["response_sha256"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
            attempt["http_status"] = response.get("status_code")
            attempt["response_headers"] = response.get("headers", {})
            _save(output / "report.json", report, api_key)
            if type(response.get("status_code")) is not int or response["status_code"] != 200:
                raise ValueError("Hosted request failed; response persisted; no retry")
            decoded = json.loads(body)
            analysis = inspect_response(decoded, item, headers=response.get("headers"))
            attempt.update(analysis)
            if isinstance(decoded, dict):
                # Actual provider/model identifiers are recorded when present, never inferred.
                attempt["returned_route"] = {k: decoded[k] for k in ("provider", "model", "id")
                                             if isinstance(decoded.get(k), str)}
            cost = analysis["usage_cost_usd"]
            if cost is not None:
                report["known_usage_cost_usd"] += cost
                report["actual_cost_complete"] = True
            report["receipts_complete"] = analysis["accounting_complete"]
            _save(output / "report.json", report, api_key)
            if analysis["status"] == "invalid_response" or not analysis["accounting_complete"]:
                report.update(status="blocked", stop_reason="Invalid timestamps/metadata or missing cost/generation receipt; remaining requests stopped")
                break
            if report["known_usage_cost_usd"] > manifest["budget_usd"]:
                report.update(status="blocked", stop_reason="Reported cost exceeds reviewed budget; remaining requests stopped")
                break
        else:
            report["status"] = "pending_review"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
        if report["attempts"] and report["attempts"][-1]["status"] == "started":
            report["attempts"][-1]["status"] = "failed"
    _save(output / "report.json", report, api_key)
    return _redact(report, api_key)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Returning None makes urllib surface the original HTTP response as an error.
        return None


def _http_child(result_path, request, api_key, timeout_seconds):
    try:
        if request["endpoint"] != ENDPOINT or request["json"]["model"] != MODEL:
            raise ValueError("Unexpected hosted endpoint or model")
        encoded = json.dumps(request["json"], separators=(",", ":"), allow_nan=False).encode()
        http_request = urllib.request.Request(ENDPOINT, data=encoded, method="POST",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            response = opener.open(http_request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Response exceeds 4 MiB")
            headers = {k: v for k, v in response.headers.items()
                       if k.lower() in ("x-generation-id", "x-request-id", "content-type", "retry-after",
                                        "x-provider", "x-model", "x-openrouter-provider")}
            result = dict(status_code=response.status, headers=headers, body=body.decode("utf-8", errors="replace"))
    except Exception as error:
        result = dict(transport_error=type(error).__name__ + ": request failed; no retry")
    with Path(result_path).open("x", encoding="utf-8") as handle:
        json.dump(result, handle)


def http_transport(request, *, api_key, timeout_seconds):
    """Dormant unless explicitly selected; hard wall timeout, no retry or redirect."""
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="nemotron-response-") as folder:
        path = Path(folder) / "response.json"
        process = context.Process(target=_http_child, args=(str(path), request, api_key, timeout_seconds))
        started = time.monotonic()
        try:
            process.start()
            process.join(timeout=max(0, timeout_seconds - (time.monotonic() - started)))
            if process.is_alive():
                raise TimeoutError("Hosted request exceeded wall deadline; not retried and may still be billed")
            if process.exitcode != 0 or not path.is_file():
                raise RuntimeError("Hosted request subprocess failed; no retry")
            # JSON escapes can expand a UTF-8 response sixfold.
            if path.stat().st_size > MAX_RESPONSE_BYTES * 6 + 65536:
                raise ValueError("Response envelope exceeds bounded size")
            result = json.loads(path.read_text())
            if "transport_error" in result:
                raise RuntimeError(result["transport_error"])
            return result
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--transition-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (args.manifest is None) == (args.transition_report is None):
        parser.error("Supply either a manifest or --transition-report")
    if args.transition_report:
        manifest = prepare_transition_manifest(args.transition_report)
    else:
        if args.manifest.stat().st_size > 256 * 1024:
            parser.error("Manifest exceeds 256 KiB")
        manifest = validate_manifest(json.loads(args.manifest.read_text()))
    if not args.execute:
        print(json.dumps(dict(status="prepared_only", manifest_sha256=manifest_sha256(manifest)), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--execute requires a new --output-dir")
    report = execute_manifest(manifest, transport=http_transport,
        api_key=os.environ.get("OPENROUTER_API_KEY"), output_dir=args.output_dir,
        confirmation=args.confirm_sha256)
    print(json.dumps(dict(status=report["status"], report=str(args.output_dir / "report.json"))))
    return 0 if report["status"] == "pending_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
