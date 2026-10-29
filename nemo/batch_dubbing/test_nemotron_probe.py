"""Pure metadata and synthetic WAV tests; no HTTP, models, or real media."""
from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import struct
import unittest
import wave
from unittest.mock import Mock, patch

from batch_dubbing.nemotron_probe import (
    _read_input, inspect_response, manifest_sha256, MODEL, ENDPOINT, MAX_FILE_BYTES,
    normalize_words, prepare_manifest, prepare_requests, validate_manifest,
    execute_manifest, http_transport, _http_child, _NoRedirect,
    prepare_transition_manifest,
)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(id="english", path="not-created-yet.wav", sha256="a" * 64,
                         duration_seconds=2.0, language="en", expected_text="Hello world.")

    def response(self):
        return dict(text="Hello world.", words=[dict(word="Hello", start=0, end=.5),
                                               dict(word="world", start=.5, end=1.0)],
                    segments=[dict(text="Hello world.", start=0, end=1.0)])

    def test_preparation_needs_no_files_and_preserves_input(self):
        original = deepcopy(self.item)
        manifest = prepare_manifest([self.item])
        self.assertEqual(self.item, original)
        manifest["inputs"][0]["expected_text"] = "Changed"
        self.assertEqual(self.item, original)

    def test_identity_changes_with_reviewed_text_and_bytes(self):
        manifest = prepare_manifest([self.item])
        identity = manifest_sha256(manifest)
        for field, value in (("expected_text", "Other text"), ("sha256", "b" * 64)):
            changed = deepcopy(manifest)
            changed["inputs"][0][field] = value
            self.assertNotEqual(identity, manifest_sha256(changed))

    def test_scope_cannot_expand_or_redirect(self):
        manifest = prepare_manifest([self.item])
        for field, value in (("endpoint", "https://other.invalid"), ("model", "other"),
                             ("max_requests", 3), ("max_retries", True),
                             ("max_retries", 1), ("timeout_seconds", 61),
                             ("budget_usd", .11), ("max_audio_seconds", 181)):
            with self.subTest(field=field):
                changed = deepcopy(manifest)
                changed[field] = value
                with self.assertRaises(ValueError):
                    validate_manifest(changed)
        with self.assertRaises(ValueError):
            prepare_manifest([self.item] * 3)

    def test_total_duration_and_nonfinite_values(self):
        second = dict(self.item, id="german", path="de.wav", language="de")
        self.assertEqual(len(prepare_manifest([self.item, second])["inputs"]), 2)
        for value in (181, float("nan"), float("inf"), True, 0, -1, 10**400):
            with self.subTest(duration=value):
                with self.assertRaises(ValueError):
                    prepare_manifest([dict(self.item, duration_seconds=value)])
        with self.assertRaises(ValueError):
            prepare_manifest([dict(self.item, duration_seconds=91), dict(second, duration_seconds=90)])

    def test_actual_wav_hash_duration_and_truncation(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
            wav.writeframes(b"\x00\x00" * 8000)
        data = buffer.getvalue()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.wav"
            path.write_bytes(data)
            item = dict(self.item, path=str(path), sha256=hashlib.sha256(data).hexdigest(), duration_seconds=1)
            self.assertEqual(_read_input(item)[1]["sample_rate"], 8000)
            with self.assertRaisesRegex(ValueError, "duration"):
                _read_input(dict(item, duration_seconds=2))
            with self.assertRaisesRegex(ValueError, "hash"):
                _read_input(dict(item, sha256="b" * 64))
            truncated = data[:-2]
            path.write_bytes(truncated)
            with self.assertRaisesRegex(ValueError, "RIFF length"):
                _read_input(dict(item, sha256=hashlib.sha256(truncated).hexdigest()))

    def test_matching_text_is_only_pending_review(self):
        result = inspect_response(self.response(), self.item)
        self.assertEqual(result["status"], "pending_review")
        self.assertTrue(result["text_comparison"]["normalized_exact_match"])
        self.assertFalse(result["naturalness_verified"])
        self.assertFalse(result["speech_deletion_verified"])
        self.assertFalse(result["speaker_attribution_available"])

    def test_terminal_omission_is_asr_difference_not_verified_audio_deletion(self):
        response = self.response()
        response["text"] = "Hello"
        response["words"] = response["words"][:1]
        result = inspect_response(response, self.item)
        self.assertFalse(result["text_comparison"]["terminal_words_match"])
        self.assertEqual(result["text_comparison"]["differences"][0]["kind"], "delete")
        self.assertFalse(result["speech_deletion_verified"])

    def test_invalid_timestamps_are_flagged_not_repaired(self):
        for start, end in ((float("nan"), .8), (True, .8), (-.1, .8), (.4, .8),
                           (.5, .5), (.5, 2.1), (10**400, 10**401)):
            with self.subTest(start=start, end=end):
                response = self.response()
                response["words"][1].update(start=start, end=end)
                result = inspect_response(response, self.item)
                self.assertEqual(result["status"], "invalid_response")
                self.assertTrue(result["timestamp_errors"])

    def test_both_timestamp_products_and_text_consistency_required(self):
        for field in ("words", "segments"):
            response = self.response()
            del response[field]
            self.assertEqual(inspect_response(response, self.item)["status"], "invalid_response")
        response = self.response()
        response["text"] = "Other words"
        self.assertIn("word text differs from transcript text", inspect_response(response, self.item)["timestamp_errors"])
        response = self.response(); response["segments"][0]["text"] = "Wrong ending"
        self.assertIn("segment text differs from transcript text", inspect_response(response, self.item)["timestamp_errors"])

    def test_german_unicode_comparison(self):
        self.assertEqual(normalize_words("Grüße, Straße!"), ["grüsse", "strasse"])

    def test_float32_44100_request_preserves_bytes_without_expected_text(self):
        samples = struct.pack("<4f", .01, -.01, .02, -.02) * 11025
        fmt = struct.pack("<HHIIHH", 3, 1, 44100, 176400, 4, 32)
        chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
        data = b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "float.wav"
            path.write_bytes(data)
            item = dict(self.item, path=str(path), duration_seconds=1, sha256=hashlib.sha256(data).hexdigest())
            request, = prepare_requests(prepare_manifest([item]))
        self.assertEqual(request["endpoint"], ENDPOINT)
        self.assertEqual(request["json"]["model"], MODEL)
        self.assertEqual(request["input_metadata"]["encoding"], 3)
        self.assertEqual(request["input_metadata"]["sample_rate"], 44100)
        self.assertEqual(base64.b64decode(request["json"]["input_audio"]["data"], validate=True), data)
        self.assertEqual(set(request["json"]), {"model", "input_audio", "language", "temperature", "response_format", "timestamp_granularities"})
        self.assertEqual(request["estimated_cost_usd"], .0000333)

    def test_all_inputs_preflight_before_any_json_is_built(self):
        second = dict(self.item, id="german", path="other.wav", language="de")
        manifest = prepare_manifest([self.item, second])
        with patch("batch_dubbing.nemotron_probe._read_input", side_effect=[(b"verified", {"duration_seconds": 2}), ValueError("changed hash")]), \
                patch("batch_dubbing.nemotron_probe.base64.b64encode") as encode:
            with self.assertRaisesRegex(ValueError, "changed hash"):
                prepare_requests(manifest)
            encode.assert_not_called()

    def test_oversized_input_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large.wav"
            with path.open("wb") as handle:
                handle.truncate(MAX_FILE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "16 MB"):
                prepare_requests(prepare_manifest([dict(self.item, path=str(path))]))

    def test_usage_cost_and_generation_header_accounting(self):
        response = self.response()
        response["usage"] = {"cost": .0000666}
        result = inspect_response(response, self.item, headers={"x-generation-id": "gen-123_example"})
        self.assertEqual(result["usage_cost_usd"], .0000666)
        self.assertEqual(result["generation_id"], "gen-123_example")
        self.assertTrue(result["accounting_complete"])
        self.assertFalse(inspect_response(self.response(), self.item)["accounting_complete"])
        for cost in (True, -1, float("nan"), float("inf"), "0.01", 10**400):
            with self.subTest(cost=cost):
                response["usage"]["cost"] = cost
                self.assertEqual(inspect_response(response, self.item)["status"], "invalid_response")
        response["usage"]["cost"] = 0
        self.assertEqual(inspect_response(response, self.item)["usage_cost_usd"], 0)
        self.assertEqual(inspect_response(response, self.item, headers={"X-Generation-Id": "bad\nheader"})["status"], "invalid_response")

    def fake_request(self, identifier):
        return dict(id=identifier, endpoint=ENDPOINT, input_metadata={"duration_seconds": 2},
                    input_sha256="a" * 64, estimated_cost_usd=.0000666,
                    json={"model": MODEL, "input_audio": {"data": "fixture", "format": "wav"}})

    def envelope(self, *, cost=.0000666, status=200):
        body = self.response()
        body["usage"] = {"cost": cost}
        return dict(status_code=status, headers={"X-Generation-Id": "gen-test"}, body=json.dumps(body))

    def execute_fake(self, folder, transport, inputs=None):
        manifest = prepare_manifest(inputs or [self.item])
        with patch("batch_dubbing.nemotron_probe.prepare_requests", return_value=[self.fake_request(i["id"]) for i in manifest["inputs"]]):
            return execute_manifest(manifest, transport=transport, api_key="secret-test-token",
                                    output_dir=Path(folder) / "results", confirmation=manifest_sha256(manifest))

    def test_execution_requires_confirmation_key_and_new_directory(self):
        manifest = prepare_manifest([self.item])
        transport = Mock()
        with tempfile.TemporaryDirectory() as folder:
            for confirmation, key in (("wrong", "secret-test-token"), (manifest_sha256(manifest), None)):
                with self.assertRaises(ValueError):
                    execute_manifest(manifest, transport=transport, api_key=key,
                                     confirmation=confirmation, output_dir=Path(folder) / "results")
            self.assertFalse((Path(folder) / "results").exists())
            with self.assertRaises(FileExistsError):
                execute_manifest(manifest, transport=transport, api_key="secret-test-token",
                                 confirmation=manifest_sha256(manifest), output_dir=folder)
        transport.assert_not_called()

    def test_preflight_failure_persists_without_transport(self):
        transport = Mock()
        manifest = prepare_manifest([self.item])
        with tempfile.TemporaryDirectory() as folder:
            report = execute_manifest(manifest, transport=transport, api_key="secret-test-token",
                                      confirmation=manifest_sha256(manifest), output_dir=Path(folder) / "results")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["attempts"], [])
            self.assertTrue((Path(folder) / "results/report.json").is_file())
        transport.assert_not_called()

    def test_two_serial_requests_cost_receipts_and_no_naturalness_claim(self):
        transport = Mock(return_value=self.envelope())
        second = dict(self.item, id="german", path="de.wav", language="de")
        with tempfile.TemporaryDirectory() as folder:
            report = self.execute_fake(folder, transport, [self.item, second])
            self.assertEqual(report["status"], "pending_review")
            self.assertEqual([a["id"] for a in report["attempts"]], ["english", "german"])
            self.assertEqual(report["known_usage_cost_usd"], .0001332)
            self.assertTrue(report["actual_cost_complete"])
            self.assertFalse(report["naturalness_verified"])
            for name in ("english", "german"):
                self.assertTrue((Path(folder) / f"results/{name}.response.txt").is_file())
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(transport.call_args.kwargs["timeout_seconds"], 60)

    def test_failure_stops_remaining_requests_preserves_partial_and_redacts_key(self):
        transport = Mock(side_effect=[self.envelope(), TimeoutError("secret-test-token timed out")])
        second = dict(self.item, id="german", path="de.wav", language="de")
        with tempfile.TemporaryDirectory() as folder:
            report = self.execute_fake(folder, transport, [self.item, second])
            stored = (Path(folder) / "results/report.json").read_text()
            self.assertNotIn("secret-test-token", stored)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["attempts"][0]["status"], "pending_review")
            self.assertEqual(report["attempts"][1]["status"], "failed")
            self.assertFalse(report["actual_cost_complete"])
        self.assertEqual(transport.call_count, 2)

    def test_invalid_response_unknown_cost_http_failure_and_budget_stop(self):
        unknown = self.envelope()
        data = json.loads(unknown["body"]); del data["usage"]; unknown["body"] = json.dumps(data)
        malformed = dict(self.envelope(), body="not JSON")
        overlap = self.envelope()
        data = json.loads(overlap["body"]); data["words"][1]["start"] = .2; overlap["body"] = json.dumps(data)
        second = dict(self.item, id="german", path="de.wav", language="de")
        for response in (unknown, malformed, overlap, self.envelope(status=429), self.envelope(cost=.2), self.envelope(cost=.09999)):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as folder:
                transport = Mock(return_value=response)
                report = self.execute_fake(folder, transport, [self.item, second])
                self.assertIn(report["status"], ("failed", "blocked"))
                transport.assert_called_once()
                self.assertTrue((Path(folder) / "results/english.response.txt").is_file())

    def test_missing_generation_receipt_keeps_known_cost_but_stops_remaining(self):
        envelope = self.envelope(); envelope["headers"] = {}
        transport = Mock(return_value=envelope)
        second = dict(self.item, id="german", path="de.wav", language="de")
        with tempfile.TemporaryDirectory() as folder:
            report = self.execute_fake(folder, transport, [self.item, second])
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["actual_cost_complete"])
        self.assertFalse(report["receipts_complete"])
        self.assertEqual(report["known_usage_cost_usd"], .0000666)
        transport.assert_called_once()

    def test_http_adapter_wall_timeout_terminates_without_retry(self):
        context = Mock()
        process = context.Process.return_value
        process.pid = 123
        process.is_alive.side_effect = [True, True, False]
        with patch("batch_dubbing.nemotron_probe.multiprocessing.get_context", return_value=context):
            with self.assertRaises(TimeoutError):
                http_transport(self.fake_request("english"), api_key="secret-test-token", timeout_seconds=60)
        context.Process.assert_called_once()
        process.start.assert_called_once()
        process.terminate.assert_called_once()

    def test_http_child_sends_only_json_no_expected_text_or_url_fallback(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.status = 200
        response.headers = {"X-Generation-Id": "gen-test"}
        response.read.return_value = self.envelope()["body"].encode()
        opener = Mock(); opener.open.return_value = response
        request = self.fake_request("english")
        with tempfile.TemporaryDirectory() as folder, patch("batch_dubbing.nemotron_probe.urllib.request.build_opener", return_value=opener):
            path = Path(folder) / "response.json"
            _http_child(path, request, "secret-test-token", 60)
            self.assertEqual(json.loads(path.read_text())["status_code"], 200)
        sent = opener.open.call_args.args[0]
        self.assertEqual(sent.full_url, ENDPOINT)
        self.assertEqual(json.loads(sent.data), request["json"])
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid"))


class TransitionManifestTests(unittest.TestCase):
    @staticmethod
    def wav(payload):
        fmt = struct.pack("<HHIIHH", 3, 1, 44100, 176400, 4, 32)
        chunks = b"fmt " + struct.pack("<I", 16) + fmt + b"fact" + struct.pack("<II", 4, len(payload)//4)
        chunks += b"data" + struct.pack("<I", len(payload)) + payload
        return b"RIFF" + struct.pack("<I", len(chunks)+4) + b"WAVE" + chunks

    def setUp(self):
        from batch_dubbing.sentence_transition_probe import build_fixture, plan_sequence
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.path = self.root / "report.json"
        fixture = build_fixture()
        self.report = dict(status="complete", stage="workload", input=fixture,
            input_sha256=hashlib.sha256(json.dumps(fixture, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            calls=5, internal_calls=7, synthesis_speed=1.0, playback_rate=1.0,
            waveform_time_stretched=False, speech_trimmed=False, steps=8, max_chunk_length=300,
            internal_chunk_silence_seconds=.3, synthetic_join_silence_seconds=.4,
            samples=[], sequences=[])
        payloads = {}
        for index, sample in enumerate(fixture["samples"]):
            frames = index + 3; payload = struct.pack("<f", (index+1)*.01) * frames
            payloads[sample["id"]] = payload
            wav = self.wav(payload); filename = sample["id"] + ".wav"
            (self.root / filename).write_bytes(wav)
            self.report["samples"].append(dict(sample, file=filename, sha256=hashlib.sha256(wav).hexdigest(),
                samples_sha256=hashlib.sha256(payload).hexdigest(), native_frames=frames, sample_rate=44100,
                raw_duration_seconds=frames/44100, reused_from_smoke=index == 0))
        for language in ("en", "de"):
            rows = [r for r in self.report["samples"] if r["language"] == language]
            plan = plan_sequence([dict(id=r["id"], native_frames=r["native_frames"]) for r in rows])
            payload = bytearray(plan["total_frames"]*4)
            for entry in plan["schedule"]:
                source = payloads[entry["id"]]
                payload[entry["start_frame"]*4:entry["end_frame"]*4] = source
                entry.update(source_samples_sha256=hashlib.sha256(source).hexdigest(),
                             sequence_slice_sha256=hashlib.sha256(source).hexdigest())
            plan["sample_content_retention_verified"] = True
            wav = self.wav(payload); filename = f"sequence_{language}.wav"
            (self.root / filename).write_bytes(wav)
            self.report["sequences"].append(dict(plan, language=language, file=filename,
                sha256=hashlib.sha256(wav).hexdigest(), samples_sha256=hashlib.sha256(payload).hexdigest()))
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.report))

    def test_derives_exact_ordered_inputs_and_full_expected_text(self):
        manifest = prepare_transition_manifest(self.path)
        self.assertEqual([i["id"] for i in manifest["inputs"]], ["sequence_en", "sequence_de"])
        self.assertEqual([i["language"] for i in manifest["inputs"]], ["en", "de"])
        for item in manifest["inputs"]:
            expected = " ".join(r["text"] for r in self.report["input"]["samples"] if r["language"] == item["language"])
            self.assertEqual(item["expected_text"], expected)
        self.assertEqual(len(prepare_requests(manifest)), 2)

    def test_rejects_scope_call_counts_and_sequence_order(self):
        original = deepcopy(self.report)
        for key, value in (("calls", 6), ("internal_calls", 8), ("stage", "smoke"), ("status", "failed"),
                           ("input_sha256", "b"*64), ("playback_rate", True)):
            self.report = deepcopy(original); self.report[key] = value; self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_transition_manifest(self.path)
        self.report = deepcopy(original); self.report["sequences"].reverse(); self.save()
        with self.assertRaises(ValueError): prepare_transition_manifest(self.path)
        self.report = deepcopy(original); self.report["input"]["samples"][0]["text"] = "Changed"; self.save()
        with self.assertRaises(ValueError): prepare_transition_manifest(self.path)

    def test_rejects_raw_file_change_frame_loss_and_schedule_edits(self):
        original = deepcopy(self.report)
        for collection, key, value in (("samples", "native_frames", 2), ("samples", "samples_sha256", "c"*64),
                                      ("samples", "file", "../else.wav"), ("sequences", "total_frames", 42)):
            self.report = deepcopy(original); self.report[collection][0][key] = value; self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): prepare_transition_manifest(self.path)
        self.report = deepcopy(original)
        self.report["sequences"][0]["schedule"][0]["sequence_slice_sha256"] = "a"*64; self.save()
        with self.assertRaises(ValueError): prepare_transition_manifest(self.path)
        self.report = original; self.save()
        (self.root / "sample_0.wav").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"): prepare_transition_manifest(self.path)

    def test_updated_sequence_hash_cannot_hide_changed_samples_or_gap(self):
        row = self.report["sequences"][0]
        original = (self.root / row["file"]).read_bytes()
        for frame in (0, self.report["samples"][0]["native_frames"]):
            payload = bytearray(original[56:]); payload[frame*4:frame*4+4] = struct.pack("<f", .5)
            wav = self.wav(payload); (self.root / row["file"]).write_bytes(wav)
            row["sha256"] = hashlib.sha256(wav).hexdigest(); row["samples_sha256"] = hashlib.sha256(payload).hexdigest()
            self.save()
            with self.subTest(frame=frame), self.assertRaisesRegex(ValueError, "native samples or fixture silence"):
                prepare_transition_manifest(self.path)


if __name__ == "__main__":
    unittest.main()
