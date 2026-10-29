"""Frozen evaluation scope validation; no model, media transform, or cloud calls."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import wave


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid {name}")
    return value


def validate_config(config):
    config = deepcopy(config)
    if config.get("version") != 1 or isinstance(config.get("version"), bool):
        raise ValueError("Evaluation scope version must be 1")
    if config.get("target_language") != "de":
        raise ValueError("This evaluation covers English source and German target only")
    count = config.get("max_units_per_case")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 12:
        raise ValueError("Evaluation permits at most 12 source units per case")
    cases = config.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 3:
        raise ValueError("Evaluation permits one to three cases")
    ids, total = set(), 0
    for case in cases:
        identifier = case.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", identifier) or identifier in ids:
            raise ValueError("Case IDs must be unique safe names")
        ids.add(identifier)
        filename = case.get("input_file")
        if not isinstance(filename, str) or "\\" in filename:
            raise ValueError("Input must be a relative WAV path")
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".wav":
            raise ValueError("Input must stay within the repository and be a WAV")
        if not re.fullmatch(r"[0-9a-f]{64}", case.get("source_sha256", "")):
            raise ValueError("Input SHA256 must be pinned")
        offset = _number(case.get("offset_seconds"), "offset")
        duration = _number(case.get("duration_seconds"), "duration", positive=True)
        source_duration = _number(case.get("source_duration_seconds"), "source duration", positive=True)
        if duration > 30 or offset + duration > source_duration:
            raise ValueError("Evaluation excerpt exceeds source or 30-second bound")
        total += duration
    if total > 90 or config.get("smoke_case_id") not in ids:
        raise ValueError("Invalid smoke selection or total evaluation duration")
    return config


def verify_input(case, path):
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("Evaluation requires existing mono 16 kHz PCM16 input")
        duration = wav.getnframes() / wav.getframerate()
    if abs(duration - case["source_duration_seconds"]) > 1 / 16000:
        raise ValueError("Source duration changed")
    if case["offset_seconds"] + case["duration_seconds"] > duration:
        raise ValueError("Requested excerpt extends past source")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != case["source_sha256"]:
        raise ValueError("Source hash changed")
    return {"path": str(path), "duration_seconds": duration, "sha256": digest.hexdigest()}


def load_config(path, *, root):
    config = validate_config(json.loads(Path(path).read_text()))
    root = Path(root).resolve()
    for case in config["cases"]:
        source = (root / case["input_file"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Input symlink escapes repository")
        verify_input(case, source)
    return config


def selected_cases(config, smoke=False):
    config = validate_config(config)
    return [case for case in config["cases"] if not smoke or case["id"] == config["smoke_case_id"]]
