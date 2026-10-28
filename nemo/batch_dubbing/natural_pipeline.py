"""Provider-independent dubbing from source-linked utterances to bounded audio.

Callbacks own model execution. This module loads no models, starts no services,
and contains no video names, timestamp corrections, or language-specific phrases.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import array
import sys

from batch_dubbing.audio_edges import analyze_edges
from batch_dubbing.speech_schedule import schedule_turn

POLICY_VERSION = "source-speech-v1"


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid {name}")
    return number


def validate_units(units, duration):
    """Require source linkage; never upgrade old display timestamps to evidence."""
    duration = _number(duration, "duration")
    if not 0 < duration <= 7200 or not units or len(units) > 10000:
        raise ValueError("Invalid duration or utterance count")
    result, ids, previous_end, word_indices, previous_index = [], set(), 0.0, set(), None
    for source in units:
        row = deepcopy(source)
        if row.get("timing_basis") != "asr_words":
            raise ValueError("Source-linked speech units required; subtitle display timing is insufficient")
        for key in ("id", "speaker", "source_text", "translated_text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key}")
        if row["id"] in ids:
            raise ValueError("Duplicate speech-unit ID")
        ids.add(row["id"])
        for key in ("source_start", "source_end", "corridor_start", "corridor_end"):
            row[key] = _number(row.get(key), key)
        if (row["source_start"] < previous_end or row["source_end"] <= row["source_start"]
                or row["source_end"] > duration or row["corridor_end"] > duration
                or row["corridor_end"] <= row["corridor_start"]
                or row["source_end"] <= row["corridor_start"]
                or row["source_start"] >= row["corridor_end"]):
            raise ValueError("Invalid source speech timeline")
        previous_end = row["source_end"]
        indices, words = row.get("source_word_indices"), row.get("words")
        if not isinstance(indices, list) or not isinstance(words, list) or not indices or len(indices) != len(words):
            raise ValueError("Word provenance is required")
        word_end = row["source_start"]
        for index, word in zip(indices, words):
            if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                    or index in word_indices or word.get("index") != index
                    or word.get("speaker") != row["speaker"]
                    or not isinstance(word.get("word"), str) or not word["word"].strip()):
                raise ValueError("Invalid or duplicated word provenance")
            if previous_index is not None and index != previous_index + 1:
                raise ValueError("Source word indices must be continuous; no interior words may be dropped")
            previous_index = index
            start, end = _number(word.get("start"), "word.start"), _number(word.get("end"), "word.end")
            if start < word_end or end <= start or end > row["source_end"]:
                raise ValueError("Words do not match source timeline")
            word_end = end
            word_indices.add(index)
        if words[0]["start"] != row["source_start"] or word_end != row["source_end"]:
            raise ValueError("Source bounds must equal the first and last word bounds")
        if indices != sorted(indices):
            raise ValueError("Word indices must be chronological")
        terms = row.get("protected_terms", [])
        if not isinstance(terms, list) or any(not isinstance(term, str) or not term for term in terms):
            raise ValueError("Protected terms must be nonempty strings")
        result.append(row)
    return result


def _save_float(path, samples, sample_rate):
    data = array.array("f", samples)
    if sys.byteorder != "little":
        data.byteswap()
    payload = data.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    content = b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    with path.open("xb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", len(content)) + content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pipeline(units, *, duration_seconds, output_dir, synthesize,
                 revise=None, max_synthesis_calls=None, max_revisions_per_unit=0,
                 max_rate=1.15, max_shift=.25):
    """Generate serially, measure, schedule turns, and optionally revise once.

    ``synthesize(unit, text)`` returns samples, sample_rate and optional voice /
    predicted_duration. ``revise(request)`` returns a string, within the caller's
    separately released inference budget. Revision never changes timing or ID.
    No waveform is rendered here. A blocked result has an empty render schedule.
    All raw attempts and measurements survive, including rejected revisions.
    """
    units = validate_units(units, duration_seconds)
    if max_revisions_per_unit not in (0, 1) or isinstance(max_revisions_per_unit, bool):
        raise ValueError("At most one explicitly budgeted revision per unit")
    if max_revisions_per_unit and revise is None:
        raise ValueError("A released revision provider is required")
    if (isinstance(max_synthesis_calls, bool) or not isinstance(max_synthesis_calls, int)
            or max_synthesis_calls < len(units) or max_synthesis_calls > len(units) * 2):
        raise ValueError("Declare a synthesis call cap between one and two per unit")
    # Validate policy before the first model callback.
    schedule_turn([], corridor_start=0, corridor_end=duration_seconds,
                  max_rate=max_rate, max_shift=max_shift)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    attempts, selected, calls, revision_calls = [], {}, 0, 0
    unit_positions = {unit["id"]: index for index, unit in enumerate(units)}
    report = {"policy": POLICY_VERSION, "status": "running", "units": units,
              "attempts": attempts, "revision_requests": [], "revision_failures": [],
              "max_synthesis_calls": max_synthesis_calls,
              "max_revisions_per_unit": max_revisions_per_unit,
              "quality_limit": "Timing and protected-term checks are not proof of meaning preservation or naturalness."}

    def persist():
        report.update(synthesis_calls=calls, revision_calls=revision_calls)
        target = output / "report.json"
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)

    def generate(unit, text, attempt):
        nonlocal calls
        if calls >= max_synthesis_calls:
            raise ValueError("Synthesis call budget exhausted")
        calls += 1
        value = synthesize(deepcopy(unit), text)
        samples, sr = value["samples"], value["sample_rate"]
        if sr != 44100 or len(samples) > sr * 120:
            raise ValueError("Expected bounded mono 44100 Hz native synthesis")
        edges = analyze_edges(samples, sr)
        predicted = value.get("predicted_duration")
        if predicted is not None:
            predicted = _number(predicted, "predicted_duration")
        filename = output / f"raw_{unit_positions[unit['id']]:05d}_{attempt}.wav"
        digest = _save_float(filename, samples, sr)
        measured = {"id": unit["id"], "text": text, "attempt": attempt,
                    "file": str(filename.resolve()), "sha256": digest, "edges": edges,
                    "voice": value.get("voice"), "predicted_duration": predicted,
                    "retained_duration": edges["retained_seconds"] + .05,
                    "head_guard": (edges["first_active_sample"] - edges["start_sample"]) / sr,
                    "tail_guard": (edges["end_sample"] - edges["last_active_end_sample"]) / sr + .05}
        attempts.append(measured)
        persist()
        return measured

    def schedule():
        turns = defaultdict(list)
        for unit in units:
            take = selected[unit["id"]]
            turns[unit["corridor_start"], unit["corridor_end"], unit["speaker"]].append(
                {**unit, **take, "alignment_verified": True})
        plans = [schedule_turn(rows, corridor_start=start, corridor_end=end,
                               max_rate=max_rate, max_shift=max_shift)
                 for (start, end, speaker), rows in turns.items()]
        return plans

    try:
        for unit in units:
            selected[unit["id"]] = generate(unit, unit["translated_text"], 0)
        plans = schedule()
        conflicts = {ident for plan in plans if plan["status"] == "blocked"
                     for conflict in plan["diagnostics"]["conflicts"] for ident in conflict["clip_ids"]}
        for unit in units:
            if unit["id"] not in conflicts:
                continue
            take = selected[unit["id"]]
            request = {"id": unit["id"], "source_text": unit["source_text"],
                "current_translation": take["text"], "source_language": unit.get("source_language"),
                "target_language": unit.get("target_language"),
                "measured_seconds": take["edges"]["retained_seconds"],
                "target_seconds": unit["source_end"] - unit["source_start"],
                "max_tempo": max_rate, "protected_terms": unit.get("protected_terms", []),
                "instruction": "Rewrite the translation in natural spoken language to fit the target time. Preserve all facts, names, quantities, negation, speaker intent and temporal meaning. Do not summarize away information, add claims, change timestamps, or return commentary. If faithful shortening is impossible, return no candidate."}
            report["revision_requests"].append(request)
            if not max_revisions_per_unit or calls >= max_synthesis_calls:
                continue
            revision_calls += 1
            text = revise(deepcopy(request))
            if not isinstance(text, str) or not text.strip() or text.strip() == take["text"]:
                report["revision_failures"].append({"id": unit["id"], "reason": "No distinct revision"})
                continue
            text = text.strip()
            if any(term not in text for term in request["protected_terms"]):
                report["revision_failures"].append({"id": unit["id"], "reason": "Protected term lost"})
                continue
            revised = generate(unit, text, 1)
            if revised["edges"]["retained_seconds"] < take["edges"]["retained_seconds"]:
                selected[unit["id"]] = revised
        plans = schedule()
        ready = all(plan["status"] == "ready" for plan in plans)
        placements = sorted([row for p in plans for row in p["schedule"]], key=lambda row: row["start"])
        if any(a["end"] > b["start"] + 1e-9 for a, b in zip(placements, placements[1:])):
            ready = False
            report["cross_turn_conflict"] = True
        rows = [{"id": row["id"], "file": selected[row["id"]]["file"],
                 "sha256": selected[row["id"]]["sha256"], "edges": selected[row["id"]]["edges"],
                 "start": row["start"], "end": row["end"], "tempo": row["rate"]} for row in placements]
        report.update(status="ready" if ready else "needs_revision", turn_plans=plans,
            selected_attempts={key: value["attempt"] for key, value in selected.items()},
            render={"status": "ready" if ready else "blocked", "sample_rate": 44100,
                    "duration_seconds": duration_seconds, "rows": rows if ready else []})
        persist()
        return report
    except Exception as exc:
        report.update(status="failed", error=str(exc), render={"status": "blocked", "rows": []})
        persist()
        raise
