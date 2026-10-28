"""Render an approved bounded schedule without dropping fitted speech samples.

No inference or video processing occurs here. All clips are validated before a
single new PCM16 timeline is published exclusively. This module has no CLI and
never runs on import. Callers must obtain any required media-processing approval.
"""
from __future__ import annotations

import array
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time
import wave


SAMPLE_RATE = 44100
TOTAL_WALL_SECONDS = 300


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_info(path):
    """Stream-check mono FLOAT32 RIFF input, including actual peak/finiteness."""
    size_on_disk = path.stat().st_size
    if size_on_disk > 128 * 1024**2:
        raise ValueError("Source WAV exceeds the 128 MiB bound")
    fmt, frames, peak = None, None, 0.0
    with path.open("rb") as handle:
        head = handle.read(12)
        if len(head) != 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            raise ValueError("Source must be RIFF WAVE")
        while chunk := handle.read(8):
            if len(chunk) != 8:
                raise ValueError("Truncated WAV chunk")
            name, size = struct.unpack("<4sI", chunk)
            if handle.tell() + size > size_on_disk:
                raise ValueError("Source WAV chunk exceeds file bounds")
            if name == b"fmt ":
                data = handle.read(size)
                if len(data) < 16:
                    raise ValueError("Invalid source WAV format")
                fmt = struct.unpack("<HHIIHH", data[:16])
                if fmt[0] != 3 or fmt[1] != 1 or fmt[2] != SAMPLE_RATE or fmt[4:] != (4, 32):
                    raise ValueError("Source must be mono 44100 Hz FLOAT32 WAV")
            elif name == b"data":
                if fmt is None or frames is not None or size % 4:
                    raise ValueError("Invalid source waveform data")
                frames = size // 4
                if not 0 < frames <= SAMPLE_RATE * 600:
                    raise ValueError("Source waveform duration exceeds bounds")
                remaining = size
                while remaining:
                    data = handle.read(min(65536, remaining))
                    if not data:
                        raise ValueError("Truncated source samples")
                    remaining -= len(data)
                    values = array.array("f", data)
                    if sys.byteorder != "little":
                        values.byteswap()
                    for sample in values:
                        if not math.isfinite(sample):
                            raise ValueError("Source contains nonfinite samples")
                        peak = max(peak, abs(sample))
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
    if frames is None or peak <= 0:
        raise ValueError("Source contains no nonzero audio")
    return {"frames": frames, "sample_rate": fmt[2], "peak": peak}


def _child_limits():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_ffmpeg(command, timeout):
    env = {**os.environ, **{name: "1" for name in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    subprocess.run(command, env=env, preexec_fn=_child_limits, timeout=timeout,
                   check=True, capture_output=True, text=True)


def render_timeline(schedule, destination, *, max_wall_seconds=300):
    """Render ``schedule`` to an exclusively created mono PCM16 WAV.

    Schema: ``status='ready'``, ``duration_seconds<=7200``, ``sample_rate=44100``,
    and ordered ``rows``. Each row needs ``id``, ``file``, ``sha256``, ``edges``
    from ``analyze_edges``, plus relative ``start``, ``end`` and ``tempo``.
    Tempo must be 1..1.15. ``end`` is the reserved hard deadline, not a request
    to trim/pad the clip. Reserve any desired fitting tolerance before calling.

    Both input and output are sample-validated. `atempo` is absent at native
    speed; when present, nothing trims its output. Any overrun fails the entire
    render. Timeline gaps are silent; only the final timeline is published.
    """
    max_wall_seconds = _number(max_wall_seconds, "max_wall_seconds")
    if not 0 < max_wall_seconds <= 7200:
        raise ValueError("Render wall budget must be in (0, 7200]")
    began = time.monotonic()

    def remaining():
        value = max_wall_seconds - (time.monotonic() - began)
        if value <= 0:
            raise TimeoutError("Renderer exceeded its total wall budget")
        return value

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.suffix.lower() != ".wav":
        raise ValueError("Destination must be a WAV file")
    if not isinstance(schedule, dict) or schedule.get("status") != "ready":
        raise ValueError("Only an entirely ready schedule can be rendered")
    if _integer(schedule.get("sample_rate"), "sample_rate") != SAMPLE_RATE:
        raise ValueError("Timeline sample rate must be 44100 Hz")
    duration = _number(schedule.get("duration_seconds"), "duration_seconds")
    if not 0 < duration <= 7200:
        raise ValueError("Timeline duration must be in (0, 7200]")
    total_frames = math.floor(duration * SAMPLE_RATE)
    rows = schedule.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Schedule must contain 1..10000 ordered rows")
    prepared, seen, previous_end = [], set(), 0.0
    for row in rows:
        remaining()
        if not isinstance(row, dict) or row.get("status", "ready") != "ready":
            raise ValueError("Blocked or invalid row")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Row IDs must be nonempty and unique")
        seen.add(identifier)
        start, end, tempo = (_number(row.get(name), name) for name in ("start", "end", "tempo"))
        if start < previous_end or end <= start or end > duration or not 1 <= tempo <= 1.15:
            raise ValueError("Invalid, overlapping or out-of-bounds row schedule")
        previous_end = end
        start_frame, end_frame = math.ceil(start * SAMPLE_RATE), math.floor(end * SAMPLE_RATE)
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError("Row has no positive sample-aligned window")
        try:
            source = Path(row["file"])
            edges = row["edges"]
            crop_start = _integer(edges["start_sample"], "crop start")
            crop_end = _integer(edges["end_sample"], "crop end")
            input_samples = _integer(edges["input_samples"], "input sample count")
            edge_rate = _integer(edges["sample_rate"], "edge sample rate")
            edge_peak = _number(edges["peak"], "edge peak")
        except (KeyError, TypeError) as exc:
            raise ValueError("Missing source or edge analysis") from exc
        expected_hash = row.get("sha256")
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)):
            raise ValueError("Expected SHA256 must be lowercase hexadecimal")
        if _digest(source) != expected_hash:
            raise ValueError(f"Source hash changed: {identifier}")
        info = _source_info(source)
        if (edge_rate != SAMPLE_RATE or input_samples != info["frames"]
                or not 0 <= crop_start < crop_end <= input_samples
                or abs(edge_peak - info["peak"]) > 1e-7):
            raise ValueError("Edge analysis does not match source waveform")
        prepared.append({"id": identifier, "file": source, "sha256": expected_hash,
                         "crop_start": crop_start, "crop_end": crop_end,
                         "start_frame": start_frame, "end_frame": end_frame,
                         "tempo": tempo, "raw_peak": info["peak"]})

    maximum_peak = max(row["raw_peak"] for row in prepared)
    gain = min(1.0, .99 / maximum_peak)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix=".natural-render-", dir=destination.parent) as temp:
        temp = Path(temp)
        rendered = []
        for index, row in enumerate(prepared):
            output = temp / f"clip_{index:04d}.wav"
            filters = [f"atrim=start_sample={row['crop_start']}:end_sample={row['crop_end']}",
                       "asetpts=PTS-STARTPTS"]
            if gain < 1:
                filters.append(f"volume={gain:.12f}")
            if row["tempo"] != 1:
                filters.append(f"atempo={row['tempo']:.12f}")
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-n",
                       "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
                       "-i", str(row["file"]), "-map", "0:a:0", "-af", ",".join(filters),
                       "-threads", "1", "-ac", "1", "-ar", str(SAMPLE_RATE),
                       "-c:a", "pcm_s16le", str(output)]
            _run_ffmpeg(command, min(90, remaining()))
            remaining()
            if _digest(row["file"]) != row["sha256"]:
                raise ValueError("Source changed during rendering")
            with wave.open(str(output), "rb") as wav:
                frames = wav.getnframes()
                if (wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE
                        or wav.getcomptype() != "NONE" or frames <= 0):
                    raise ValueError("Renderer produced invalid PCM16 audio")
                if row["tempo"] == 1 and frames != row["crop_end"] - row["crop_start"]:
                    raise ValueError("Native-speed render changed the crop's sample count")
                if row["start_frame"] + frames > row["end_frame"]:
                    raise ValueError(f"Rendered speech exceeds planned end: {row['id']}; nothing was truncated")
            rendered.append((row, output, frames))
            reports.append({"id": row["id"], "source": str(row["file"]), "source_sha256": row["sha256"],
                            "tempo": row["tempo"], "filters": filters,
                            "start_sample": row["start_frame"], "output_samples": frames,
                            "output_end_sample": row["start_frame"] + frames,
                            "planned_end_sample": row["end_frame"], "rendered_sha256": _digest(output)})

        staged = temp / "timeline.wav"
        with wave.open(str(staged), "wb") as timeline:
            timeline.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            position = 0

            def silence(frames):
                while frames:
                    remaining()
                    count = min(frames, 32768)
                    timeline.writeframesraw(b"\0" * (count * 2))
                    frames -= count

            for row, path, frames in rendered:
                if row["start_frame"] < position:
                    raise ValueError("Rendered speech overlaps; timeline not published")
                silence(row["start_frame"] - position)
                written = 0
                with wave.open(str(path), "rb") as clip:
                    while data := clip.readframes(32768):
                        remaining()
                        timeline.writeframesraw(data)
                        written += len(data) // 2
                if written != frames:
                    raise ValueError("Rendered audio is truncated; timeline not published")
                position = row["start_frame"] + frames
            silence(total_frames - position)
        remaining()
        with wave.open(str(staged), "rb") as wav:
            if wav.getnframes() != total_frames:
                raise ValueError("Timeline frame count differs from schedule")
        output_digest = _digest(staged)
        remaining()
        os.link(staged, destination)
    return {"status": "complete", "destination": str(destination), "sample_rate": SAMPLE_RATE,
            "duration_seconds": total_frames / SAMPLE_RATE, "sha256": output_digest,
            "global_gain": gain, "global_gain_applied": gain < 1, "maximum_input_peak": maximum_peak,
            "elapsed_seconds": time.monotonic() - began, "rows": reports,
            "limits": {"one_cpu_core_per_child": True, "serial": True, "child_address_space_mib": 512,
                       "child_cpu_seconds": 60, "child_wall_seconds": 90, "total_wall_seconds": max_wall_seconds},
            "processing": "Analyzed edge crop, optional bounded tempo, shared attenuation if needed, silent timeline gaps. No post-tempo trim."}
