"""Safe, reusable legacy audio fitting with bounded FFmpeg and no speech cutting.

Only already-generated mono WAVs are accepted. This module never imports a TTS
model. Short clips keep their natural tempo; excess speech fails explicitly.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import wave

from batch_dubbing.audio_edges import analyze_edges
from batch_dubbing.natural_render import _run_ffmpeg

FIT_POLICY = "natural-fit-v1"
OUTPUT_RATE = 24000  # Existing legacy stitcher produces 24 kHz silence.
_FIT_LOCK = threading.Lock()


class AudioFitError(ValueError):
    """Speech cannot fit safely; caller must not publish a partial dub."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def validate_fit_policy(max_speed=1.15, min_speed=1.0):
    maximum, minimum = _finite(max_speed, "max_speed"), _finite(min_speed, "min_speed")
    if not 1.0 <= maximum <= 1.15 or minimum != 1.0:
        raise ValueError("Natural fitting requires min_speed=1.0 and max_speed between 1.0 and 1.15; slowing or aggressive compression is disabled")
    return maximum, minimum


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mono_wav(path):
    """Decode bounded PCM16/24/32 or FLOAT32 WAV without optional libraries."""
    path = Path(path)
    file_size = path.stat().st_size
    if file_size > 64 * 1024**2:
        raise ValueError("WAV exceeds the 64 MiB input bound")
    fmt = data = None
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("Expected an uncompressed mono WAV")
        while chunk := handle.read(8):
            if len(chunk) != 8:
                raise ValueError("Truncated WAV chunk")
            name, size = struct.unpack("<4sI", chunk)
            if handle.tell() + size > file_size:
                raise ValueError("WAV chunk extends beyond file")
            if name == b"fmt ":
                blob = handle.read(size)
                if len(blob) < 16:
                    raise ValueError("Invalid WAV format")
                kind, channels, rate, _, align, bits = struct.unpack("<HHIIHH", blob[:16])
                if kind == 65534:
                    if len(blob) < 40:
                        raise ValueError("Invalid extensible WAV format")
                    kind = struct.unpack("<H", blob[24:26])[0]
                if (channels != 1 or not 1 <= rate <= 384000 or align != bits // 8
                        or not ((kind == 1 and bits in (16, 24, 32)) or (kind == 3 and bits == 32))):
                    raise ValueError("Expected mono PCM16/24/32 or FLOAT32 WAV")
                fmt = (kind, rate, bits, align)
            elif name == b"data":
                if data is not None:
                    raise ValueError("Multiple waveform data chunks are unsupported")
                data = handle.read(size)
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
    if fmt is None or data is None:
        raise ValueError("Missing waveform format or data")
    kind, rate, bits, align = fmt
    if len(data) % align or not 0 < len(data) // align <= 600 * rate:
        raise ValueError("Invalid waveform sample count or duration")
    if kind == 3:
        samples = array.array("f", data)
        if sys.byteorder != "little":
            samples.byteswap()
    elif bits in (16, 32):
        integers = array.array("h" if bits == 16 else "i", data)
        if sys.byteorder != "little":
            integers.byteswap()
        divisor = float(2 ** (bits - 1))
        samples = array.array("f", (value / divisor for value in integers))
    else:
        samples = array.array("f", (int.from_bytes(data[i:i + 3], "little", signed=True) / 8388608.0
                                   for i in range(0, len(data), 3)))
    return samples, rate


def _validate_output(path, target_duration):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels() != 1 or audio.getframerate() != OUTPUT_RATE
                or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE"):
            raise AudioFitError("Fitted artifact has an invalid format")
        frames = audio.getnframes()
        if frames <= 0 or frames > math.floor(target_duration * OUTPUT_RATE):
            raise AudioFitError("Fitted speech still exceeds its deadline; refusing to cut words")
        actual_frames = 0
        while block := audio.readframes(32768):
            actual_frames += len(block) // 2
        if actual_frames != frames:
            raise AudioFitError("Fitted waveform is truncated")
    return frames


def fit_file(audio_path, target_dur, max_speed=1.15, min_speed=1.0):
    """Return a new cached PCM16 WAV at native tempo or at most ``max_speed``.

    Conservative edge padding is removed before measuring fit. Short clips are
    neither stretched nor padded; the stitcher positions subsequent clips at
    their timestamps. Infeasible speech raises ``AudioFitError``. Every FFmpeg
    child has one-core affinity, 512 MiB memory, 60 CPU/90 wall-second bounds.
    This function serializes fitting calls within a process. Raw audio is intact.
    """
    maximum, minimum = validate_fit_policy(max_speed, min_speed)
    target = _finite(target_dur, "target_dur")
    if not 0 < target <= 600:
        raise ValueError("target_dur must be in (0, 600]")
    audio_path = Path(audio_path)
    with _FIT_LOCK:
        began = time.monotonic()
        source_sha = _digest(audio_path)
        configuration = {"policy": FIT_POLICY, "source_sha256": source_sha, "target_seconds": target,
                         "max_speed": maximum, "min_speed": minimum, "output_sample_rate": OUTPUT_RATE}
        key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
        destination = audio_path.with_name(audio_path.stem + f"_natural_fit_{key}.wav")
        receipt = destination.with_suffix(".json")
        if destination.exists() or receipt.exists():
            if not destination.is_file() or not receipt.is_file():
                raise FileExistsError("Incomplete existing fit artifact; review it instead of overwriting")
            metadata = json.loads(receipt.read_text())
            if metadata.get("configuration") != configuration or metadata.get("output_sha256") != _digest(destination):
                raise FileExistsError("Existing fit artifact does not match its provenance")
            _validate_output(destination, target)
            return destination
        samples, rate = read_mono_wav(audio_path)
        edges = analyze_edges(samples, rate)
        del samples
        retained = edges["retained_seconds"]
        tempo = 1.0
        if retained > target:
            # Reserve time for FFmpeg's finite-window tempo-filter drift.
            budget = target - min(.05, target * .05)
            tempo = retained / budget
            if tempo > maximum + 1e-12:
                raise AudioFitError(f"Speech needs {retained:.3f}s after edge cleanup but has {target:.3f}s; "
                                    f"required tempo {tempo:.3f} exceeds {maximum:.3f}. Review wording or speech timing; no words were cut.")
            tempo = min(tempo, maximum)
        gain = min(1.0, .99 / edges["peak"])
        filters = [f"atrim=start_sample={edges['start_sample']}:end_sample={edges['end_sample']}",
                   "asetpts=PTS-STARTPTS"]
        if gain < 1:
            filters.append(f"volume={gain:.12f}")
        if tempo > 1:
            filters.append(f"atempo={tempo:.12f}")
        with tempfile.TemporaryDirectory(prefix=".natural-fit-", dir=audio_path.parent) as tmp:
            tmp = Path(tmp)
            staged = tmp / "fitted.wav"
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-n",
                       "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
                       "-i", str(audio_path), "-map", "0:a:0", "-af", ",".join(filters),
                       "-threads", "1", "-ar", str(OUTPUT_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(staged)]
            remaining = 120 - (time.monotonic() - began)
            if remaining <= 0:
                raise TimeoutError("Fitting exceeded its total preparation budget")
            _run_ffmpeg(command, min(90, remaining))
            frames = _validate_output(staged, target)
            if tempo == 1 and abs(frames - edges["retained_samples"] * OUTPUT_RATE / rate) > 1:
                raise AudioFitError("Native fitting changed the retained speech duration")
            if _digest(audio_path) != source_sha:
                raise AudioFitError("Raw speech changed during fitting")
            metadata = {"configuration": configuration, "edges": edges, "tempo": tempo, "gain": gain,
                        "output_samples": frames, "output_seconds": frames / OUTPUT_RATE,
                        "output_sha256": _digest(staged), "filters": filters,
                        "no_post_tempo_trim": True}
            staged_receipt = tmp / "receipt.json"
            staged_receipt.write_text(json.dumps(metadata, indent=2) + "\n")
            # Exclusive publication never replaces a previous user artifact.
            os.link(staged_receipt, receipt)
            try:
                os.link(staged, destination)
            except BaseException:
                receipt.unlink()  # This invocation exclusively created it above.
                raise
        return destination
