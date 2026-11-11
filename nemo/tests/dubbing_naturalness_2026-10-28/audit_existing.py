"""Read-only timing audit of existing artifacts; stdlib, no model/media execution.

Run: python3 dubbing_naturalness_2026-10-28/audit_existing.py
Only audit JSON files are written. WAVs are opened for headers, never decoded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import statistics
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).with_name("metrics.json")
REVIEW_DEST = Path(__file__).with_name("review_data.json")
PROVENANCE = {}


def relative(path):
    return str(path.relative_to(ROOT))


def record(path):
    data = path.read_bytes()
    PROVENANCE[relative(path)] = {
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)
    }
    return data


def read_json(path):
    return json.loads(record(path))


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def ends_sentence(text):
    return bool(re.search(r'[.!?…][”"\u2019\']?$', text.rstrip()))


def summarize_rates(rates):
    return {
        "count": len(rates), "min": min(rates),
        "median": statistics.median(rates),
        "p90": percentile(rates, .90), "p95": percentile(rates, .95),
        "max": max(rates),
        "equal_1": sum(r == 1 for r in rates),
        "above_1_25": sum(r > 1.25 for r in rates),
        "above_1_5": sum(r > 1.5 for r in rates),
        "above_2": sum(r > 2 for r in rates),
        "adjacent_absolute_rate_changes_above_0_35": sum(
            abs(a - b) > .35 for a, b in zip(rates, rates[1:])
        ),
    }


def load_merger():
    path = ROOT / "modal_control/dub_segments.py"
    record(path)
    spec = importlib.util.spec_from_file_location("audited_dub_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_speech_cues


def reconstruct(data, report, index, merge):
    cues = [dict(c) for c in data["translations"]["de"]
            if c["end"] <= report["seconds"]]
    voices = report.get("voice_assignments", data["voices"])
    attributed = {c["speaker"] for c in cues if c.get("speaker")}
    # Mirrors continuity worker's single-attributed-speaker rule for videos
    # 02-04; videos 00/01 use their recorded duo/single-speaker assignments.
    if index in (2, 3, 4) and len(attributed) == 1:
        voices = {next(iter(attributed)): None}
    missing = sum(c.get("speaker") is None for c in cues)
    for cue in cues:
        if cue.get("speaker") is not None:
            continue
        if len(voices) == 1:
            cue["speaker"] = next(iter(voices))
            continue
        scores = {}
        for original in data["english"]:
            overlap = max(0, min(cue["end"], original["end"]) -
                          max(cue["start"], original["start"]))
            speaker = original.get("speaker")
            if speaker in voices and overlap:
                scores[speaker] = scores.get(speaker, 0) + overlap
        if not scores:
            raise ValueError("Cannot reconstruct speaker assignment")
        cue["speaker"] = max(scores, key=scores.get)
    segments = merge(cues)
    matches = len(segments) == len(report["segments"])
    speakers_match = matches and all(
        segment["speaker"] == actual["speaker"]
        for segment, actual in zip(segments, report["segments"])
    )
    if not speakers_match:
        raise ValueError("Reconstructed segment count or speaker order differs")
    return cues, segments, missing


def audit_video(index, merge):
    folder = ROOT / "modal_control/batch_inputs"
    stem = f"video_{index:02d}"
    meta = read_json(folder / (stem + ".json"))
    data = read_json(folder / (stem + ".gpu_full.json"))
    report = read_json(folder / (stem + ".cpu_v2_full.json"))
    rates = [s["rate"] for s in report["segments"]]
    result = {
        "video_index": index, "source_path_at_run": meta["source"],
        "duration_seconds": report["seconds"],
        "report_path": relative(folder / (stem + ".cpu_v2_full.json")),
        "rate_summary": summarize_rates(rates),
    }
    try:
        cues, segments, missing = reconstruct(data, report, index, merge)
        result["reconstruction"] = {
            "status": "matched_segment_count_and_speaker_order",
            "input_cues": len(cues), "missing_speakers_inferred": missing,
            "segment_count": len(segments),
            "segments_under_3_seconds": sum(s["end"] - s["start"] < 3 for s in segments),
            "segments_without_sentence_terminal": sum(not ends_sentence(s["text"]) for s in segments),
            "limitation": "Reconstructed from current pure merge code and worker speaker rules; matching counts and speaker order do not prove historical text/interval identity. Latest reports omit raw audio durations and boundaries. No subjective quality conclusion.",
        }
        examples = []
        for n in sorted(range(len(rates)), key=lambda n: rates[n], reverse=True)[:5]:
            segment = segments[n]
            fit_end = segments[n + 1]["start"] if n + 1 < len(segments) else report["seconds"]
            window = fit_end - segment["start"]
            examples.append({
                "index": n, **segment, "fit_window_end": fit_end,
                "fit_window_seconds": window, "reported_rate": rates[n],
                "raw_seconds_inferred_from_rate": window * rates[n] if rates[n] > 1 else None,
                "duration_basis": "inferred; rates equal to 1 hide the raw duration",
            })
        result["highest_rate_examples"] = examples
    except ValueError as error:
        result["reconstruction"] = {"status": "unavailable", "error": str(error)}
    return result, rates


def wav_header(path):
    try:
        with wave.open(str(path), "rb") as handle:
            return {"duration_seconds": handle.getnframes() / handle.getframerate(),
                    "frames": handle.getnframes(), "sample_rate": handle.getframerate(),
                    "channels": handle.getnchannels(), "sample_width_bytes": handle.getsampwidth()}
    except (OSError, wave.Error) as error:
        return {"error": str(error)}


def audit_benchmark(folder):
    rows = read_json(folder / "segments.json")
    measured = []
    for row in rows:
        header = wav_header(folder / row["path"])
        window = row["source_end"] - row["source_start"]
        duration = header.get("duration_seconds")
        measured.append({
            "index": row["index"], "path": relative(folder / row["path"]),
            "source_start": row["source_start"], "source_end": row["source_end"],
            "source_window_seconds": window, "metadata_audio_seconds": row["audio_seconds"],
            "header": header,
            "metadata_duration_error_seconds": duration - row["audio_seconds"] if duration is not None else None,
            "required_speedup_to_window": max(1, duration / window) if duration is not None else None,
            "padding_seconds_at_original_speed": max(0, window - duration) if duration is not None else None,
        })
    result = {
        "model_directory": folder.name, "segments": measured,
        "total_padding_seconds": sum(r["padding_seconds_at_original_speed"] or 0 for r in measured),
        "nonterminal_text_boundaries": sum(not ends_sentence(r["text"]) for r in rows),
        "boundary_examples": [{"left_index": a["index"], "boundary_seconds": a["source_end"],
                               "left_text": a["text"], "right_text": b["text"]}
                              for a, b in zip(rows, rows[1:]) if not ends_sentence(a["text"])][:3],
        "interpretation_limit": "Padding follows the existing fixed-window packager. This is a timing measure, not a test of whether synthesized words are complete or speech sounds natural.",
    }
    proxy_path = folder / "quality_proxy.json"
    if proxy_path.exists():
        proxy = read_json(proxy_path)
        result["existing_asr_proxy"] = {k: proxy.get(k) for k in ("metric", "evaluator", "wer", "cer", "deletions")}
    return result


def distribution(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "p10": percentile(values, .1), "p90": percentile(values, .9),
            "p95": percentile(values, .95), "max": max(values)}


def word_count(text):
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text, re.UNICODE))


def parse_srt(path):
    text = record(path).decode("utf-8-sig").replace("\r\n", "\n")
    pattern = re.compile(r"(?m)^(\d+)\n(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\n(.*?)(?=\n\s*\n|\Z)", re.S)
    def stamp(value):
        h, m, s, ms = map(int, re.split(r"[:,]", value))
        return h * 3600 + m * 60 + s + ms / 1000
    return [{"index": int(number) - 1, "start": stamp(start), "end": stamp(end),
             "text": body.strip().replace("\n", " ")}
            for number, start, end, body in pattern.findall(text)]


def review_cues(actual, metadata, end=600):
    parity = len(actual) == len(metadata) and all(
        abs(a["start"] - b["start"]) <= .0011 and abs(a["end"] - b["end"]) <= .0011
        and a["text"] == b["text"].strip().replace("\n", " ")
        for a, b in zip(actual, metadata)
    )
    rows = []
    for row in actual:
        if row["start"] >= end or row["end"] <= 0:
            continue
        duration = row["end"] - row["start"]
        words = word_count(row["text"])
        rows.append({**row, "visible_start": max(0, row["start"]),
                     "visible_end": min(end, row["end"]),
                     "crosses_review_end": row["end"] > end,
                     "speaker": metadata[row["index"]].get("speaker") if parity else None,
                     "duration_seconds": duration, "words": words,
                     "text_words_per_cue_second": words / duration,
                     "ends_sentence": ends_sentence(row["text"])})
    return rows, parity


def cue_summary(rows, end=600):
    intervals = sorted((r["visible_start"], r["visible_end"]) for r in rows)
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    gaps = [max(0, b["start"] - a["end"]) for a, b in zip(rows, rows[1:])]
    return {
        "cue_count": len(rows),
        "complete_cues": sum(not r["crosses_review_end"] for r in rows),
        "duration_seconds": distribution([r["duration_seconds"] for r in rows]),
        "words_per_cue": distribution([r["words"] for r in rows]),
        "text_words_per_cue_second": distribution([r["text_words_per_cue_second"] for r in rows]),
        "cues_under_1_second": sum(r["duration_seconds"] < 1 for r in rows),
        "cues_under_2_seconds": sum(r["duration_seconds"] < 2 for r in rows),
        "boundaries_without_sentence_terminal": sum(not a["ends_sentence"] for a in rows[:-1]),
        "adjacent_speaker_changes": sum(a["speaker"] is not None and b["speaker"] is not None and
                                        a["speaker"] != b["speaker"] for a, b in zip(rows, rows[1:])),
        "adjacent_speaker_unknown": sum(a["speaker"] is None or b["speaker"] is None
                                        for a, b in zip(rows, rows[1:])),
        "gap_seconds_between_cues": distribution(gaps),
        "positive_gap_count": sum(g > .001 for g in gaps),
        "total_positive_gap_seconds_between_cues": sum(gaps),
        "gaps_over_0_2_seconds": sum(g > .2 for g in gaps),
        "gaps_over_0_4_seconds": sum(g > .4 for g in gaps),
        "gaps_over_1_second": sum(g > 1 for g in gaps),
        "subtitle_visible_seconds_union": sum(stop - start for start, stop in merged),
        "subtitle_uncovered_seconds": end - sum(stop - start for start, stop in merged),
        "overlapping_cue_boundaries": sum(b["start"] < a["end"] - .001 for a, b in zip(rows, rows[1:])),
    }


def build_review_video(index, key, merge):
    folder = ROOT / "modal_control/batch_inputs"
    stem = f"video_{index:02d}"
    meta = read_json(folder / (stem + ".json"))
    data = read_json(folder / (stem + ".gpu_full.json"))
    report = read_json(folder / (stem + ".cpu_v2_full.json"))
    _, all_segments, missing = reconstruct(data, report, index, merge)
    original = ROOT / "new_guys" / key / Path(meta["source"]).name
    dub = original.with_name(original.stem + ".german-dub.mp4")
    if not original.is_file() or not dub.is_file():
        raise FileNotFoundError("Expected original and existing dub for review")
    english_path = original.with_suffix(".en.srt")
    german_path = original.with_suffix(".de.srt")
    english, english_parity = review_cues(parse_srt(english_path), data["english"])
    german, german_parity = review_cues(parse_srt(german_path), data["translations"]["de"])
    segments = []
    for number, row in enumerate(all_segments):
        if row["start"] >= 600 or row["end"] <= 0:
            continue
        fit_end = all_segments[number + 1]["start"] if number + 1 < len(all_segments) else report["seconds"]
        rate = report["segments"][number]["rate"]
        segments.append({**row, "index": number, "rate": rate,
                         "voice": report["segments"][number]["voice"],
                         "fit_window_end": fit_end, "fit_window_seconds": fit_end - row["start"],
                         "visible_start": max(0, row["start"]), "visible_end": min(600, fit_end),
                         "crosses_review_end": fit_end > 600,
                         "raw_seconds_inferred_from_rate": (fit_end - row["start"]) * rate if rate > 1 else None,
                         "ends_sentence": ends_sentence(row["text"]), "words": word_count(row["text"])})
    bins = []
    for start in range(0, 600, 30):
        selected_segments = [r for r in segments if start <= r["start"] < start + 30]
        row = {"start": start, "end": start + 30,
               "merged_segment_onsets": len(selected_segments),
               "tempo_rates_by_segment_onset": summarize_rates([r["rate"] for r in selected_segments]) if selected_segments else None}
        for name, cues in (("english", english), ("german", german)):
            selected = [r for r in cues if start <= r["start"] < start + 30]
            row[name] = {"cue_onsets": len(selected),
                         "words_in_cues_starting_in_bin": sum(r["words"] for r in selected),
                         "cue_duration_seconds": distribution([r["duration_seconds"] for r in selected]),
                         "text_words_per_cue_second": distribution([r["text_words_per_cue_second"] for r in selected])}
        bins.append(row)
    return {
        "id": key, "video_index": index, "review_start": 0, "review_end": 600,
        "title": original.stem, "original_path": relative(original), "dub_path": relative(dub),
        "english_subtitle_path": relative(english_path), "german_subtitle_path": relative(german_path),
        "actual_srt_matches_metadata": {"english": english_parity, "german": german_parity},
        "english_cues": english, "german_cues": german, "merged_segments": segments,
        "english_between_cue_gap_proxy": [
            {"start": a["end"], "end": b["start"], "seconds": b["start"] - a["end"],
             "previous_cue_index": a["index"], "next_cue_index": b["index"]}
            for a, b in zip(english, english[1:]) if b["start"] - a["end"] > .001
        ],
        "summary": {"english": cue_summary(english), "german": cue_summary(german),
                    "merged_tempo": summarize_rates([r["rate"] for r in segments]),
                    "merged_boundaries_without_sentence_terminal": sum(not r["ends_sentence"] for r in segments[:-1]),
                    "merged_speaker_changes": sum(a["speaker"] != b["speaker"] for a, b in zip(segments, segments[1:])),
                    "full_video_speaker_gap_inferences": missing},
        "bins_30s": bins,
        "reconstruction_limit": "Merged boundaries are inferred from current merge/speaker rules and match full report segment count and speaker order. Raw durations are inferred only for reported rates >1. Subtitle end is not a measured speech offset.",
    }


def write_review(merge):
    for source in ("batch_dubbing/gpu_worker.py", "modal_control/smoke_nemo_helpers.py",
                   "local_subtitles.py", "nemo_audio.py"):
        record(ROOT / source)
    videos = [build_review_video(index, key, merge) for index, key in
              ((4, "mind_control"), (0, "bill_ackman"))]
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "default_video_id": "mind_control", "path_base": "repository root",
        "scope": "Existing original English video, existing German dub, and actual English/German SRTs for first 600 seconds of each of two videos. Read-only source inspection.",
        "word_rate_definition": "Unicode word tokens divided by full subtitle cue duration. Text density is not measured articulation rate; English and German token counts are not phonetically equivalent. Cues crossing 600 seconds retain whole text and whole-cue density; visible boundaries are separately clipped.",
        "bins_definition": "30-second bins group cues/segments by start time. Words in those cues are assigned as whole text to their onset bin, not presented as words acoustically spoken in that bin.",
        "timing_provenance": {
            "english": "gpu_worker.py obtains Parakeet word timestamps, then smoke_nemo_helpers._words_to_segs groups words into cues. Cue start/end retain first/last word boundaries. Returned JSON stores only cues, not the underlying individual word timestamps; those cannot be recovered from SRT.",
            "german": "local_subtitles.infer_chunk takes Canary segment timings and calls nemo_audio._split_coarse_segs, distributing each segment duration across display lines in proportion to word counts. German cue times therefore are not forced-aligned spoken German word boundaries.",
            "english_gap_proxy": "Gaps between English ASR-derived cue boundaries are approximate inter-cue speech gaps; not acoustic VAD or ground truth. They omit pauses internal to cues and cannot distinguish ASR omission from actual silence. Do not equate subtitle gaps or amplitude-threshold silence with absence of human speech.",
        },
        "limitations": ["No listening assessment or audio decoding in this audit.",
                        "Subtitle timing, translation accuracy and speaker labels are not human-verified.",
                        "Media paths exist but media byte content is not hashed by this script.",
                        "No source content beyond the review interval is inferred for partial last cues."],
        "videos": videos, "input_provenance": PROVENANCE,
    }
    REVIEW_DEST.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


def main():
    merge = load_merger()
    for source in ("batch_dubbing/cpu_worker.py", "batch_dubbing/cpu_duo_worker.py",
                   "batch_dubbing/cpu_override_worker.py", "batch_dubbing/cpu_continuity_worker.py",
                   "speech_model_benchmark_2026-10-27/code/organize_results.py"):
        record(ROOT / source)
    videos, rates = [], []
    for index in range(5):
        video, values = audit_video(index, merge)
        videos.append(video)
        rates.extend(values)
    benchmark_root = ROOT / "speech_model_benchmark_2026-10-27/04_dubbing"
    benchmarks = [audit_benchmark(benchmark_root / model) for model in
                  ("supertonic_3", "qwen_tts_06b", "ming_05b", "voxcpm2")]
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Five existing new_guys German dub metadata reports; four existing 24-segment benchmark raw WAV sets. JSON reads and WAV headers only, no inference or audio decoding.",
        "limitations": [
            "No listening assessment; measured timing cannot establish perceived naturalness.",
            "Rates are post-synthesis tempo multipliers, not synthesis real-time factors.",
            "Missing latest raw synthesis WAVs prevent a faithful no-synthesis rerender of latest videos.",
            "JSON/source SHA256 hashes identify audit inputs; WAV headers verify duration but WAV content is not hashed.",
            "Sentence-terminal detection is a punctuation heuristic, not a linguistic annotation.",
        ],
        "all_latest_segments": summarize_rates(rates), "videos": videos,
        "benchmark": benchmarks, "input_provenance": PROVENANCE,
    }
    DEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_review(merge)
    print(json.dumps({"output": str(DEST), "all_latest_segments": result["all_latest_segments"],
                      "benchmark_wav_header_count": sum(len(b["segments"]) for b in benchmarks)}, indent=2))


if __name__ == "__main__":
    main()
