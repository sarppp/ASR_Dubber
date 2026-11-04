#!/usr/bin/env python3
"""Subtitle-first dubbing pipeline (default).

Subtitles come from the local subtitle pipeline — Parakeet Ultra ASR, cue
packing, whole-sentence Tower translation, deterministic split back onto the cue
boundaries, plus the combined multi-language SRT/ASS — and the dubbing step then
reads the target-language SRT exactly as the older pipeline did.

    video ──► Ultra ──► cues ──► sentences ──► Tower ──► <lang>.srt
                                                     └─► dub.py (target lang)

The old NeMo + Canary/Gemma flow still exists as ``--pipeline legacy-dubbing``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_utils import (
    END_PRODUCT_DIR,
    NEMO_CODE_DIR,
    NEMO_DIR,
    QWEN_DIR,
    QWEN_PY,
    WHISPER_PY,
    _banner,
    _python,
    _run,
)
from pipeline_paths import (
    _derive_run_label,
    _detect_source_language,
    _finalize_outputs,
    _find_video,
    _validate_translated_srt,
)

if str(NEMO_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEMO_CODE_DIR))
from local_mt import (  # noqa: E402
    DUB_MAX_SPEED,
    TTS_CHARS_PER_SECOND,
    OllamaTranslator,
    group_sentences,
)

SUBTITLE_PY = NEMO_CODE_DIR / ".venv-local-asr" / "bin" / "python"
SUBTITLE_ENTRY = NEMO_CODE_DIR / "batch_subtitles.py"
KNOWN = ("en", "de", "fr")


def _interpreter() -> Path:
    if SUBTITLE_PY.exists():
        return SUBTITLE_PY
    fallback = NEMO_CODE_DIR / ".venv" / "bin" / "python"
    if fallback.exists():
        print(f"⚠️  {SUBTITLE_PY} missing — falling back to {fallback}")
        return fallback
    return Path(sys.executable)


def _video_base(video: Path, trim: int) -> str:
    base = re.split(r"[._]nemo|__", video.stem)[0]
    return f"{base}_t{trim}" if trim else base


def _subtitle_languages(args, source: str, target: str) -> list[str]:
    if args.subtitle_languages:
        ordered = [lang for lang in args.subtitle_languages if lang in KNOWN]
        if not ordered:
            sys.exit(f"❌  --subtitle-languages must use {', '.join(KNOWN)}")
        if target in KNOWN and target not in ordered:
            ordered.append(target)  # dubbing needs the target subtitle
        return ordered
    ordered = []
    for lang in (source, target):
        if lang in KNOWN and lang not in ordered:
            ordered.append(lang)
    if not ordered:
        sys.exit(f"❌  Neither source ('{source}') nor target ('{target}') is one of "
                 f"{', '.join(KNOWN)}; pass --subtitle-languages explicitly")
    return ordered


def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    cues = []
    stamp = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    for block in re.split(r"\r?\n\r?\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        match = stamp.match(lines[1])
        if not match:
            continue
        g = [int(v) for v in match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append((start, end, " ".join(lines[2:]).strip()))
    return cues


def _block_cues(cues, merge_gap: float, merge_max_dur: float):
    """Group cues the way dub.py merges them into TTS segments."""
    groups: list[list[tuple[float, float, str]]] = []
    for start, end, text in cues:
        if (groups and start - groups[-1][-1][1] <= merge_gap
                and end - groups[-1][0][0] <= merge_max_dur):
            groups[-1].append((start, end, text))
        else:
            groups.append([(start, end, text)])
    return groups


NEGATIONS = {
    "de": ("nicht", "kein", "keine", "keinen", "keinem", "keiner", "nie", "niemals",
           "nichts", "ohne"),
    "fr": ("ne", "n'", "pas", "jamais", "aucun", "aucune", "rien", "sans"),
    "en": ("not", "no", "never", "nothing", "without", "n't"),
}


def _preservation_problem(before: str, after: str, language: str) -> str | None:
    """Cheap deterministic check that the adaptation kept the content.

    It cannot judge style, only that the obvious carriers of meaning survived:
    numbers, negation and a sane amount of substance. Anything suspicious keeps
    the original line instead of silently losing it.
    """
    if len(after) < 0.55 * len(before):
        return "too short (possible dropped content)"
    words = after.split()
    if len(words) >= 9:
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        if len(set(trigrams)) < len(trigrams) / 3:
            return "repetition loop"
    if len(after) > 2.0 * len(before):
        return "longer than the original"
    for number in re.findall(r"\d+", before):
        if number not in after:
            return f"number {number} missing"
    lowered_before, lowered_after = before.lower(), after.lower()
    for marker in NEGATIONS.get(language, ()):
        if marker in lowered_before.split() and marker not in lowered_after:
            return f"negation '{marker}' missing"
    return None


def _rate(language: str) -> float:
    return float(os.getenv("SUBTITLE_TTS_CPS")
                 or TTS_CHARS_PER_SECOND.get(language, 12.0))


def _sentence_spans(cues: list[dict]) -> list[tuple[int, int]]:
    """Sentence groups as cue index ranges, same rules as local_mt."""
    from local_mt import MAX_SENTENCE_CUES, MAX_SENTENCE_SECONDS

    spans: list[tuple[int, int]] = []
    begin = 0
    for index, cue in enumerate(cues):
        text = (cue.get("text") or "").strip()
        span = float(cues[index]["end"]) - float(cues[begin]["start"])
        done = (bool(re.search(r"[.!?]$", text))
                or index - begin + 1 >= MAX_SENTENCE_CUES
                or span >= MAX_SENTENCE_SECONDS)
        if done:
            spans.append((begin, index))
            begin = index + 1
    if begin < len(cues):
        spans.append((begin, len(cues) - 1))
    return spans


def _adapt(srt: Path, source_srt: Path, source_language: str, language: str,
           translator: OllamaTranslator, *,
           target: float = 1.0, max_sentences: int | None = None
           ) -> tuple[Path, list[dict], Path]:
    """Steps 3-4: gate whole sentences at `target` tempo, adapt the failures.

    Target 1.0 means the voice speaks at its natural pace: the wording gets
    shorter, the audio is never sped up. The unit is a complete sentence, which
    is exactly what the dub synthesises (one cue per sentence).
    """
    cps = _rate(language)
    target_cues = [{"start": a, "end": b, "text": t} for a, b, t in _parse_srt(srt)]
    source_cues = [{"start": a, "end": b, "text": t} for a, b, t in _parse_srt(source_srt)]
    groups = _sentence_spans(target_cues)
    if max_sentences:
        groups = groups[:max_sentences]

    aim = target * 0.92  # the real TTS runs a few percent longer than the estimate
    _banner(f"Step 2/4 — Gate + adaptation ({language}, {cps:.1f} chars/s, "
            f"target {target:.2f}x, aim {aim:.2f}x)")

    rows: list[dict] = []
    spoken: list[tuple[float, float, str]] = []
    over: list[int] = []

    for number, (first, last) in enumerate(groups, 1):
        start, end = target_cues[first]["start"], target_cues[last]["end"]
        span = max(0.01, end - start)
        text = " ".join((c["text"] or "").strip() for c in target_cues[first:last + 1]).strip()
        source_text = " ".join((c["text"] or "").strip()
                               for c in source_cues[first:last + 1]).strip()
        before = len(text) / cps / span
        row = {"index": number, "start": round(start, 3), "end": round(end, 3),
               "span": round(span, 2), "chars_before": len(text),
               "ratio_before": round(before, 2), "text_before": text}

        adapted, ratio = text, before
        rejected_reason = None
        if before > aim:
            # Re-translate from the source with a budget: asking a translation
            # model to "rewrite this German" makes it switch language and loop.
            for _attempt in range(2):
                budget = max(12, int(len(adapted) * (aim / ratio) * 0.95))
                candidate = translator.translate_sentence(
                    source_text, source_language, language, max_chars=budget)
                candidate = re.sub(r"\s+", " ", candidate).strip().strip('"')
                if not candidate:
                    break
                candidate_ratio = len(candidate) / cps / span
                problem = _preservation_problem(text, candidate, language)
                if problem:
                    rejected_reason = rejected_reason or problem
                elif candidate_ratio < ratio:
                    adapted, ratio = candidate, candidate_ratio
                if ratio <= aim:
                    break
            problem = (None if adapted != text
                       else rejected_reason or _preservation_problem(text, adapted, language))
            if problem:
                row["kept_original"] = problem
                print(f"   [{number:02d}] {before:.2f}x  adaptation rejected "
                      f"({problem}); keeping the full meaning")
                adapted, ratio = text, before
            else:
                print(f"   [{number:02d}] {before:.2f}x -> {len(adapted)} chars "
                      f"({ratio:.2f}x)")

        if ratio > target:
            over.append(number)
        row.update(chars_after=len(adapted), text_after=adapted,
                   ratio_after=round(ratio, 2))
        rows.append(row)
        spoken.append((start, end, adapted))

    if over:
        print(f"   ⚠️  {len(over)} sentence(s) still above {target:.2f}x: {over}")

    adapted_path = srt.with_name(srt.stem + ".adapted.srt")
    adapted_path.write_text(_render_srt(spoken), encoding="utf-8-sig")
    dub_path = srt.with_name(srt.stem.replace(".adapted", "") + ".dub.srt")
    dub_path.write_text(_render_srt(spoken), encoding="utf-8-sig")
    print(f"   dub script: {len(spoken)} complete sentences -> {dub_path.name}")
    return adapted_path, rows, dub_path


def _stage_video(video: Path, workdir: Path) -> Path:
    """batch_subtitles works on directories, so give it a folder with this one video."""
    workdir.mkdir(parents=True, exist_ok=True)
    link = workdir / video.name
    if not link.exists():
        try:
            link.symlink_to(video.resolve())
        except OSError:
            shutil.copy2(video, link)
    return link


def _produce_subtitles(args, video: Path, workdir: Path, source: str,
                       target: str, base: str, nemo_dir: Path) -> dict[str, Path]:
    languages = _subtitle_languages(args, source, target)
    _stage_video(video, workdir)

    produced: dict[str, Path] = {}
    missing = [lang for lang in languages
               if not (workdir / f"{video.stem}.nemo.{lang}.srt").exists()]
    multi_missing = not (workdir / f"{video.stem}.multi.srt").exists()

    if missing or multi_missing:
        cmd = _python(_interpreter(), NEMO_CODE_DIR) + [
            str(SUBTITLE_ENTRY),
            "--backend", "local",
            "--input-dir", str(workdir),
            "--source-language", source,
            "--languages", *languages,
        ]
        if args.asr_model:
            cmd += ["--asr-model", args.asr_model]
        if args.translator:
            cmd += ["--translator", args.translator]
        _banner(f"Step 1/4 — Subtitles ({' + '.join(languages)}) via Ultra + Tower")
        _run(cmd, cwd=str(NEMO_CODE_DIR), label="Subtitle generation")
    else:
        print(f"⏭️  Subtitles already generated for {video.name} ({', '.join(languages)})")

    # Move out of the staging dir so clean_subs/dub find them next to the video.
    for lang in languages:
        src = workdir / f"{video.stem}.nemo.{lang}.srt"
        if not src.exists():
            sys.exit(f"❌  Subtitle step produced no {lang} SRT for {video.name}")
        dest = nemo_dir / f"{base}.nemo.{lang}.srt"
        shutil.move(str(src), str(dest))
        produced[lang] = dest

    for suffix in (".multi.srt", ".multi.ass"):
        src = workdir / f"{video.stem}{suffix}"
        if src.exists():
            shutil.move(str(src), str(nemo_dir / f"{base}{suffix}"))
    return produced


def _write_adaptation_report(json_path: Path, markdown_path: Path,
                             rows: list[dict], target: float) -> None:
    payload = {"target_tempo": target, "blocks": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    lines = [f"# Adaptation report (target {target:.2f}x, natural pace)", ""]
    lines.append("| # | time | span | before | after | status | text before | text after |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        note = row.get("kept_original", "adapted")
        lines.append(
            f"| {row['index']} | {row['start']:.2f}-{row['end']:.2f} | {row['span']:.2f}s "
            f"| {row['ratio_before']:.2f}x | {row['ratio_after']:.2f}x | {note} "
            f"| {row['text_before']} | {row['text_after']} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_srt(cues: list[tuple[float, float, str]]) -> str:
    def stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for index, (start, end, text) in enumerate(cues, 1):
        lines.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{text}\n")
    return "\n".join(lines)


def _adapt_from_report(dub_srt: Path, source_srt: Path, source_language: str,
                       language: str, rows: list[dict], report_path: Path,
                       translator: OllamaTranslator, target: float) -> list[float]:
    """Step 4b: re-adapt only the sentences the real TTS measured as too long."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    failing = [seg for seg in payload.get("segments", [])
               if (seg.get("tempo") or 0) > target]
    if not failing:
        return []

    source_cues = [{"start": a, "end": b, "text": t} for a, b, t in _parse_srt(source_srt)]
    spans = _sentence_spans(source_cues)
    cps = _rate(language)
    changed_starts: list[float] = []
    _banner(f"Step 3b/4 — Re-adapt {len(failing)} sentence(s) with measured durations")
    for seg in failing:
        index = seg.get("index")
        if not index or index < 1 or index > len(rows):
            continue
        row = rows[index - 1]
        current = row["text_after"]
        tempo = seg.get("tempo") or target
        budget = max(12, int(len(current) * (target / tempo) * 0.95))
        first, last = spans[index - 1]
        source_text = " ".join((c["text"] or "").strip()
                               for c in source_cues[first:last + 1]).strip()
        candidate = translator.translate_sentence(
            source_text, source_language, language, max_chars=budget)
        candidate = re.sub(r"\s+", " ", candidate).strip().strip('"')
        if not candidate:
            continue
        problem = _preservation_problem(current, candidate, language)
        if problem:
            print(f"   [{index:02d}] measured {tempo:.2f}x — candidate rejected ({problem})")
            continue
        row["text_after"] = candidate
        row["ratio_measured"] = round(tempo, 2)
        row["chars_after"] = len(candidate)
        changed_starts.append(float(row["start"]))
        print(f"   [{index:02d}] measured {tempo:.2f}x -> {len(candidate)} chars "
              f"(budget {budget})")
    if not changed_starts:
        return []

    spoken = [(row["start"], row["end"], row["text_after"]) for row in rows]
    dub_srt.write_text(_render_srt(spoken), encoding="utf-8-sig")
    adapted = dub_srt.with_name(dub_srt.stem.replace(".dub", ".adapted") + ".srt")
    adapted.write_text(_render_srt(spoken), encoding="utf-8-sig")
    return changed_starts


def _invalidate_fit_cache(dub_workdir: Path, starts: list[float], tol: float = 0.05) -> None:
    """Drop cached fitted clips for edited sentences.

    dub.py's checkpoint validates only that the clip file exists, not that the
    text is unchanged, so a re-adapted line would otherwise reuse the old take.
    """
    for checkpoint in dub_workdir.rglob("checkpoint.json"):
        try:
            entries = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            continue
        keep = []
        for entry in entries:
            if any(abs(float(entry.get("start", -99)) - start) < tol for start in starts):
                Path(entry.get("clip", "")).unlink(missing_ok=True)
            else:
                keep.append(entry)
        checkpoint.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")


def _dub(args, video: Path, srt: Path, base: str, target: str) -> tuple[Path, int]:
    dub_workdir = QWEN_DIR / "output" / "dub" / base
    if args.skip_dub:
        print("⏭️  Skipping dub (--skip-dub)")
        return dub_workdir, 0

    print(f"📄 Using SRT : {srt}")
    _validate_translated_srt(srt, target)
    dub_workdir.mkdir(parents=True, exist_ok=True)

    cmd = _python(QWEN_PY, QWEN_DIR) + [
        str(QWEN_DIR / "dub.py"),
        str(video),
        str(srt),
        "--language", target,
        "--tts-engine", args.tts_engine,
        "--supertonic-voice", args.supertonic_voice,
        "--qwen-mode", args.qwen_mode,
        "--workdir", str(dub_workdir),
    ]
    if args.no_demucs:
        cmd.append("--no-demucs")
    if args.max_speed is not None:
        cmd += ["--max-speed", str(args.max_speed)]
    if args.min_speed is not None:
        cmd += ["--min-speed", str(args.min_speed)]
    if args.merge_gap is not None:
        cmd += ["--merge-gap", str(args.merge_gap)]
    if args.merge_max_dur is not None:
        cmd += ["--merge-max-dur", str(args.merge_max_dur)]
    cmd += ["--tts-workers", str(args.tts_workers), "--tts-devices", args.tts_devices]

    _banner(f"Step 3/4 — Dubbing into {target} with {args.tts_engine}")
    rc = _run(cmd, cwd=str(QWEN_DIR), label=f"Dubbing ({args.tts_engine})", check=False)
    return dub_workdir, rc


def run(args) -> int:
    nemo_dir = Path(args.input_dir).resolve() if args.input_dir else NEMO_DIR
    end_product_dir = Path(args.output_dir).resolve() if args.output_dir else END_PRODUCT_DIR
    if args.input_file:
        input_file = Path(args.input_file).resolve()
        if not input_file.exists():
            sys.exit(f"❌  Input file not found: {input_file}")
        nemo_dir = input_file.parent
    end_product_dir.mkdir(parents=True, exist_ok=True)

    video = Path(args.input_file).resolve() if args.input_file else _find_video(
        target_lang=args.target_lang, nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    if video is None:
        sys.exit(f"❌  No unprocessed video found in {nemo_dir} for target '{args.target_lang}'")

    base = _video_base(video, args.trim)
    source = args.language or _detect_source_language(video, whisper_model=args.whisper_model)
    if not source:
        sys.exit("❌  Language detection failed. Pass --language explicitly.")
    target = source if args.target_lang == "same" else args.target_lang

    _banner("Step 0/4 — Pick video + source language")
    print(f"🎬 {video.name}  (base '{base}')")
    print(f"🌐 Source: {source}  →  Target: {target}")
    print("🧩 Pipeline: subtitle-dubbing (Ultra + Tower)")

    workdir = nemo_dir / ".subtitle_dub" / base
    subtitles = _produce_subtitles(args, video, workdir, source, target, base, nemo_dir)
    target_srt = subtitles.get(target)
    if target_srt is None:
        sys.exit(f"❌  No {target} subtitle was produced for {video.name}")

    adapted_srt, rows, dub_srt = _adapt(
        target_srt, subtitles[source], source, target, OllamaTranslator(),
        target=args.fit_target, max_sentences=args.max_sentences)
    _write_adaptation_report(nemo_dir / f"{base}.adaptation.json",
                             nemo_dir / f"{base}.adaptation.md", rows, args.fit_target)

    dub_workdir, rc = _dub(args, video, dub_srt, base, target)
    if rc != 0:
        report = dub_workdir / "fit_report.json"
        if report.exists():
            changed = _adapt_from_report(
                dub_srt, subtitles[source], source, target, rows, report,
                OllamaTranslator(), args.fit_target)
            if changed:
                _invalidate_fit_cache(dub_workdir, changed)
                _write_adaptation_report(nemo_dir / f"{base}.adaptation.json",
                                         nemo_dir / f"{base}.adaptation.md", rows,
                                         args.fit_target)
                dub_workdir, rc = _dub(args, video, dub_srt, base, target)
    if rc != 0:
        print("❌  Dubbing still does not fit at natural pace; see the fit report.")
        sys.exit(rc)

    _banner("Step 4/4 — Finalize + gather outputs")
    run_label = _derive_run_label(source, target, video=video,
                                  nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    _finalize_outputs(run_label, dub_workdir=None if args.skip_dub else dub_workdir,
                      nemo_dir=nemo_dir)
    print(f"\n✅ Subtitle dubbing complete → {end_product_dir / run_label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point; `run_pipeline.py` calls run() with its own args."""
    from run_pipeline import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_lang is None and args.run_mode != "transcribe":
        parser.error("--target-lang is required")
    if args.run_mode == "transcribe":
        args.target_lang = args.target_lang or "same"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
