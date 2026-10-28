"""Immutable speech units retaining source word identity through translation.

No models, translation providers, or media tools are invoked. Sentence detection
uses punctuation heuristics; timing always comes from original ASR word bounds.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re


def _number(value, name, minimum=0.0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _speaker(value):
    if (not isinstance(value, str) or not value.strip()
            or value.strip().casefold() in {"unknown", "unknown_speaker", "unattributed", "none", "null", "?"}):
        raise ValueError("An explicit known speaker is required")
    return value


def _digest(prefix, value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return prefix + hashlib.sha256(data).hexdigest()


def _terminal(text):
    return bool(re.search(r'[.!?…。！？][\"\'»›”’“\)\]]*$', text.rstrip()))


def reconcile_speaker_corridors(words: Sequence[Mapping], turns: Sequence[Mapping]):
    """Reconcile diarization boundaries using positive original-word overlap.

    Original word and turn records are never changed. Attribution must equal the
    unique strongest positive-overlap speaker. Simultaneous different-speaker
    turns, ties, uncovered words and overlapping words block the entire result.
    Consecutive words with the same evidenced speaker share a corridor. Original
    turn extents are retained where possible and expanded to contain full words;
    adjacent different-speaker runs divide their *silent word gap* at its midpoint.
    This last boundary only allocates scheduling space, never changes word times.
    """
    for records, name in ((words, "words"), (turns, "turns")):
        if (not isinstance(records, Sequence) or isinstance(records, (str, bytes))
                or len(records) > 200_000):
            raise ValueError(f"{name} must be a bounded sequence")
    report = {"status": "blocked", "policy": "positive-overlap-corridors-v1",
              "turns": [], "word_evidence": [], "adjustments": [], "conflicts": [],
              "source_word_count": len(words), "raw_turn_count": len(turns),
              "word_timestamps_changed": False}

    def blocked(reason, **details):
        report["conflicts"].append({"reason": reason, **details})
        return report

    if not words or not turns:
        return blocked("Missing source words or diarization turns")
    merged, previous_start = [], -1.0
    for index, raw in enumerate(turns):
        if not isinstance(raw, Mapping):
            return blocked("Invalid diarization record", turn_index=index)
        try:
            start, end = _number(raw.get("start"), "turn.start"), _number(raw.get("end"), "turn.end")
            speaker = _speaker(raw.get("speaker"))
        except ValueError as exc:
            return blocked(str(exc), turn_index=index)
        if end <= start or start < previous_start:
            return blocked("Invalid or unordered diarization bounds", turn_index=index)
        previous_start = start
        if merged and start < merged[-1]["end"] and speaker != merged[-1]["speaker"]:
            return blocked("Simultaneous different-speaker turns", turn_index=index,
                           previous_turn_indices=merged[-1]["raw_indices"])
        if merged and start <= merged[-1]["end"] and speaker == merged[-1]["speaker"]:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["raw_indices"].append(index)
        else:
            merged.append({"start": start, "end": end, "speaker": speaker, "raw_indices": [index]})
    ends = [row["end"] for row in merged]
    previous_end, runs = 0.0, []
    for index, raw in enumerate(words):
        if not isinstance(raw, Mapping):
            return blocked("Invalid word record", word_index=index)
        try:
            start, end = _number(raw.get("start"), "word.start"), _number(raw.get("end"), "word.end")
            speaker = _speaker(raw.get("speaker"))
        except ValueError as exc:
            return blocked(str(exc), word_index=index)
        if not isinstance(raw.get("word"), str) or not raw["word"].strip():
            return blocked("Empty source word", word_index=index)
        if end <= start or start < previous_end:
            return blocked("Invalid, unordered or overlapping source words", word_index=index)
        previous_end = end
        evidence, supports = {}, {}
        for candidate_index in range(bisect_left(ends, start), len(merged)):
            candidate = merged[candidate_index]
            if candidate["start"] >= end:
                break
            overlap = min(end, candidate["end"]) - max(start, candidate["start"])
            if overlap > 0:
                name = candidate["speaker"]
                evidence[name] = evidence.get(name, 0.0) + overlap
                supports.setdefault(name, []).append(candidate)
        if not evidence:
            return blocked("No positive diarization overlap", word_index=index)
        best = max(evidence.values())
        winners = [name for name, overlap in evidence.items() if abs(overlap - best) <= 1e-9]
        if len(winners) != 1 or winners[0] != speaker:
            return blocked("Ambiguous or conflicting speaker attribution", word_index=index,
                           attributed_speaker=speaker, overlap_seconds=evidence)
        support = supports[speaker]
        raw_indices = sorted({i for row in support for i in row["raw_indices"]})
        report["word_evidence"].append({"word_index": index, "speaker": speaker,
                                       "overlap_seconds": evidence, "raw_turn_indices": raw_indices})
        left, right = min(row["start"] for row in support), max(row["end"] for row in support)
        if runs and runs[-1]["speaker"] == speaker:
            run = runs[-1]
            run["word_end"] = end
            run["last_word_index"] = index
            run["raw_start"] = min(run["raw_start"], left)
            run["raw_end"] = max(run["raw_end"], right)
            run["raw_indices"].update(raw_indices)
        else:
            runs.append({"speaker": speaker, "word_start": start, "word_end": end,
                         "first_word_index": index, "last_word_index": index,
                         "raw_start": left, "raw_end": right, "raw_indices": set(raw_indices)})
    for index, run in enumerate(runs):
        desired_start = min(run["raw_start"], run["word_start"])
        desired_end = max(run["raw_end"], run["word_end"])
        left = (runs[index - 1]["word_end"] + run["word_start"]) / 2 if index else 0.0
        right = (run["word_end"] + runs[index + 1]["word_start"]) / 2 if index + 1 < len(runs) else desired_end
        start, end = max(desired_start, left), min(desired_end, right)
        report["turns"].append({"start": start, "end": end, "speaker": run["speaker"]})
        report["adjustments"].append({"corridor_index": index, "speaker": run["speaker"],
            "raw_turn_indices": sorted(run["raw_indices"]),
            "first_word_index": run["first_word_index"], "last_word_index": run["last_word_index"],
            "raw_extent": [run["raw_start"], run["raw_end"]], "corridor_extent": [start, end],
            "leading_expansion_seconds": max(0.0, run["raw_start"] - start),
            "trailing_expansion_seconds": max(0.0, end - run["raw_end"]),
            "gap_boundary_restricted": start != desired_start or end != desired_end})
    report["status"] = "ready"
    return report


@dataclass(frozen=True)
class SourceWord:
    index: int
    word: str
    start: float
    end: float
    speaker: str

    def to_dict(self):
        return {"index": self.index, "word": self.word, "start": self.start,
                "end": self.end, "speaker": self.speaker}


@dataclass(frozen=True)
class SpeechUnit:
    id: str
    speaker: str
    source_start: float
    source_end: float
    corridor_start: float
    corridor_end: float
    source_text: str
    words: tuple[SourceWord, ...]
    flags: tuple[str, ...]
    source_id: str
    turn_id: str
    translated_text: str | None = None
    timing_basis: str = "asr_words"

    @property
    def source_word_indices(self):
        return tuple(word.index for word in self.words)

    def to_dict(self):
        return {"id": self.id, "speaker": self.speaker,
                "source_start": self.source_start, "source_end": self.source_end,
                "corridor_start": self.corridor_start, "corridor_end": self.corridor_end,
                "source_text": self.source_text, "translated_text": self.translated_text,
                "words": [word.to_dict() for word in self.words],
                "source_word_indices": list(self.source_word_indices),
                "flags": list(self.flags), "source_id": self.source_id,
                "turn_id": self.turn_id, "timing_basis": self.timing_basis}


def build_speech_units(
    words: Sequence[Mapping], turns: Sequence[Mapping], *, source_id="",
    soft_max_seconds=20.0, hard_max_seconds=30.0, max_characters=500,
    pause_seconds=0.5, rounding_tolerance=0.02,
):
    """Group attributed ASR words without deriving speech times from captions.

    A sentence ending, meaningful pause, or speaker turn ends a unit. After the
    soft duration target, clause punctuation may also end it. The hard duration
    or character ceiling splits only between words and flags unfinished units.
    A single oversized word is retained intact and explicitly flagged.

    Every word must fit exactly one same-speaker turn, allowing at most 20 ms
    source/turn boundary rounding. Original word timestamps are never clamped;
    any such tolerance use is flagged. Overlapping words/turns and ambiguous
    assignments fail. ``source_id`` namespaces IDs across recordings; pass an
    asset identifier/hash when assembling units from multiple recordings.
    """
    soft_max_seconds = _number(soft_max_seconds, "soft_max_seconds")
    hard_max_seconds = _number(hard_max_seconds, "hard_max_seconds")
    pause_seconds = _number(pause_seconds, "pause_seconds")
    rounding_tolerance = _number(rounding_tolerance, "rounding_tolerance")
    if (not 0 < soft_max_seconds <= hard_max_seconds or pause_seconds == 0
            or rounding_tolerance > .02 or not isinstance(source_id, str)
            or isinstance(max_characters, bool) or not isinstance(max_characters, int)
            or max_characters <= 0):
        raise ValueError("Invalid speech-unit boundary policy")
    for value, name in ((words, "words"), (turns, "turns")):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a bounded sequence")
    if len(words) > 200_000 or len(turns) > 200_000:
        raise ValueError("Too many source records")
    valid_turns = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise ValueError("Turns must be mappings")
        start, end = _number(turn.get("start"), "turn.start"), _number(turn.get("end"), "turn.end")
        speaker = _speaker(turn.get("speaker"))
        if end <= start or (valid_turns and start < valid_turns[-1]["end"]):
            raise ValueError("Turns must have positive durations, ordered without overlap")
        record = {"index": index, "start": start, "end": end, "speaker": speaker}
        record["id"] = _digest("turn_", {**record, "source_id": source_id})
        valid_turns.append(record)
    ends = [turn["end"] for turn in valid_turns]
    validated = []
    previous_end = None
    for index, word in enumerate(words):
        if not isinstance(word, Mapping):
            raise ValueError("Words must be mappings")
        text = word.get("word")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every source word must have nonempty text; no words are silently dropped")
        start, end = _number(word.get("start"), "word.start"), _number(word.get("end"), "word.end")
        speaker = _speaker(word.get("speaker"))
        if end <= start or (previous_end is not None and start < previous_end):
            raise ValueError("Words must have positive durations, ordered without overlap")
        previous_end = end
        candidates = []
        first = bisect_left(ends, start - rounding_tolerance)
        for turn_index in range(first, len(valid_turns)):
            turn = valid_turns[turn_index]
            if turn["start"] > start + rounding_tolerance:
                break
            if (speaker == turn["speaker"] and start >= turn["start"] - rounding_tolerance
                    and end <= turn["end"] + rounding_tolerance):
                candidates.append(turn)
        if len(candidates) != 1:
            raise ValueError(f"Source word {index} has no unique same-speaker turn assignment")
        turn = candidates[0]
        if validated and turn["index"] < validated[-1][1]["index"]:
            raise ValueError("Source turn assignments reverse in time")
        validated.append((SourceWord(index, text, start, end, speaker), turn))

    result, current, current_turn, current_characters = [], [], None, 0

    def flush(reason):
        nonlocal current, current_characters
        if not current:
            return
        source_text = " ".join(word.word.strip() for word in current)
        flags = []
        if not _terminal(source_text):
            flags.append("unfinished_sentence")
            if reason in {"hard_duration_limit", "character_limit", "soft_duration_boundary"}:
                flags.append("unfinished_at_size_limit")
        if reason in {"hard_duration_limit", "character_limit", "soft_duration_boundary"}:
            flags.append(reason)
        if any(word.end - word.start > hard_max_seconds or len(word.word.strip()) > max_characters for word in current):
            flags.append("oversized_source_word")
        if current[0].start < current_turn["start"] or current[-1].end > current_turn["end"]:
            flags.append("turn_boundary_rounding_tolerance_used")
        identity = {"source_id": source_id, "turn_id": current_turn["id"],
                    "words": [word.to_dict() for word in current]}
        result.append(SpeechUnit(
            id=_digest("speech_", identity), speaker=current[0].speaker,
            source_start=current[0].start, source_end=current[-1].end,
            corridor_start=current_turn["start"], corridor_end=current_turn["end"],
            source_text=source_text, words=tuple(current), flags=tuple(flags),
            source_id=source_id, turn_id=current_turn["id"],
        ))
        current = []
        current_characters = 0

    for word, turn in validated:
        if current:
            if current_turn["id"] != turn["id"]:
                flush("turn_boundary")
            elif word.start - current[-1].end >= pause_seconds:
                flush("source_pause")
            elif word.end - current[0].start > hard_max_seconds:
                flush("hard_duration_limit")
            elif current_characters + 1 + len(word.word.strip()) > max_characters:
                flush("character_limit")
        current_turn = turn
        current_characters += len(word.word.strip()) + bool(current)
        current.append(word)
        text = word.word.rstrip()
        if _terminal(text):
            flush("sentence_end")
        elif word.end - current[0].start > hard_max_seconds:
            flush("hard_duration_limit")
        elif len(word.word.strip()) > max_characters:
            flush("character_limit")
        elif (word.end - current[0].start >= soft_max_seconds
              and re.search(r'[,;:，；：][\"\'»›”’“\)\]]*$', text)):
            flush("soft_duration_boundary")
    flush("input_end")
    assert tuple(index for unit in result for index in unit.source_word_indices) == tuple(range(len(words)))
    return tuple(result)


def bind_translations(units: Sequence[SpeechUnit], translations: Mapping[str, str]):
    """Bind one translation per stable source ID; preserve all timing/provenance."""
    if (not isinstance(units, Sequence) or isinstance(units, (str, bytes))
            or not isinstance(translations, Mapping)
            or any(not isinstance(unit, SpeechUnit) for unit in units)):
        raise ValueError("Expected source speech units and an ID-to-text mapping")
    ids = [unit.id for unit in units]
    if len(set(ids)) != len(ids) or set(translations) != set(ids):
        raise ValueError("Translations must cover every unique source ID exactly; no missing or extra IDs")
    bound = []
    for unit in units:
        text = translations[unit.id]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each translation must be nonempty text")
        bound.append(replace(unit, translated_text=text.strip()))
    return tuple(bound)


def bind_caption_translations(units: Sequence[SpeechUnit], captions_by_id: Mapping):
    """Join each separately translated utterance's display captions by source ID.

    The caller must have translated each unit's own source audio. Display cue
    timestamps remain estimates; only the immutable unit's ASR word bounds are
    speech timing. Returned captions retain a source ID and its known speaker.
    """
    if (not isinstance(units, Sequence) or not isinstance(captions_by_id, Mapping)
            or any(not isinstance(unit, SpeechUnit) for unit in units)):
        raise ValueError("Expected source units and per-unit caption results")
    if len({unit.id for unit in units}) != len(units) or set(captions_by_id) != {unit.id for unit in units}:
        raise ValueError("Caption translations must cover every source unit exactly")
    translations, captions = {}, []
    for unit in units:
        rows = captions_by_id[unit.id]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ValueError(f"No translation returned for source unit {unit.id}")
        texts = []
        previous_end = None
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Translated captions need nonempty text")
            start, end = _number(row.get("start"), "caption.start"), _number(row.get("end"), "caption.end")
            # infer_chunk rounds display times to milliseconds. Do not change
            # the source unit or reject a <=1ms formatting roundoff.
            if (end <= start or start < unit.source_start - .001
                    or end > unit.source_end + .001
                    or (previous_end is not None and start < previous_end)):
                raise ValueError("Translated display captions leave their source unit or overlap")
            previous_end = end
            text = row["text"].strip()
            texts.append(text)
            captions.append({**row, "start": start, "end": end, "text": text,
                             "speaker": unit.speaker, "source_id": unit.id,
                             "timing_basis": "display_estimate"})
        translations[unit.id] = " ".join(texts)
    return bind_translations(units, translations), captions


def build_shorter_revision_request(
    unit: SpeechUnit, *, retained_duration, target_available_seconds, max_rate=1.15,
):
    """Describe a faithful shorter revision for any backend; never call a model.

    Actual returned text still needs semantic review and fresh measured synthesis.
    Character/word ratios are not duration guarantees. The source identity and
    timing are immutable; a backend cannot extend the available window silently.
    """
    if not isinstance(unit, SpeechUnit) or not unit.translated_text:
        raise ValueError("A source unit with a bound translation is required")
    retained_duration = _number(retained_duration, "retained_duration")
    target_available_seconds = _number(target_available_seconds, "target_available_seconds")
    max_rate = _number(max_rate, "max_rate", 1.0)
    maximum_native = target_available_seconds * max_rate
    if not retained_duration or not target_available_seconds or not math.isfinite(maximum_native):
        raise ValueError("Measured duration and available window must be positive and finite")
    return {
        "schema": 1, "task": "faithful_shorter_translation",
        "source_unit_id": unit.id, "source_text": unit.source_text,
        "current_translation": unit.translated_text,
        "source_start": unit.source_start, "source_end": unit.source_end,
        "corridor_start": unit.corridor_start, "corridor_end": unit.corridor_end,
        "source_words": [word.to_dict() for word in unit.words],
        "measured_retained_duration_seconds": retained_duration,
        "target_available_seconds": target_available_seconds,
        "max_rate": max_rate, "maximum_native_duration_seconds": maximum_native,
        "instructions": [
            "Use concise natural phrasing in the same target language as the current translation.",
            "Preserve every substantive meaning, named entity, number, unit, negation, modality, and causal or temporal relationship in the source.",
            "Do not delete information, invent facts, change speaker identity, or alter the source ID or timing to make speech fit.",
            "Shorten phrasing rather than content. If a faithful shorter version is not possible, return cannot_shorten_faithfully instead of omitting meaning.",
            "Duration is a synthesis target, not a word-count guarantee; the result requires semantic review and measured resynthesis.",
        ],
        "response_schema": {"source_unit_id": unit.id,
                            "status": "candidate or cannot_shorten_faithfully",
                            "translated_text": "candidate text, or unchanged text when unable",
                            "explanation": "what wording changed or why faithful shortening was not possible"},
    }
