"""Sentence-first local pipeline: ASR once, translate whole sentences, cut to cues.

This is the flow validated against Ling 3.0 and used by the trial runs:

  1. one ASR pass per video, in the source language, with word timestamps;
  2. words are packed into subtitle cues (max 10 words / 5 s / 80 chars);
  3. cues are grouped into complete sentences;
  4. each sentence is translated whole by a local Ollama model;
  5. the translation is cut back onto the cue boundaries;
  6. per-language SRTs plus one combined multi-language SRT+ASS are always written.

ASR is pluggable (parakeet-ultra by default, canary/parakeet-v2/v3/qwen3-asr
optional). Translation is pluggable (tower-plus-2b by default,
translategemma:4b optional).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
import wave
from pathlib import Path

from local_mt import DEFAULT_MT_MODEL, OllamaTranslator, translate_cues
from local_subtitles import chunk_boundaries, render_srt, save_json
from subtitle_combine import write_files

log = logging.getLogger("local_asr")

DEFAULT_ASR_MODEL = os.getenv("ASR_MODEL", "parakeet-ultra")
MT_MODELS = {
    "tower": "tower-plus-2b:latest",
    "gemma": "translategemma:4b",
    "tower-7b": "hymt2-7b",
}
LEGACY = "canary"  # in-pass translation, the old behaviour


def resolve_mt_model(translator: str | None) -> str:
    if not translator:
        return DEFAULT_MT_MODEL
    return MT_MODELS.get(translator, translator)


def resolve_asr_model(model_name: str | None) -> str:
    """Map a short name (``parakeet-v3``) to the HF id the loaders expect."""
    from nemo_audio import ASR_MODELS

    if not model_name:
        return DEFAULT_ASR_MODEL
    return ASR_MODELS.get(model_name.lower(), model_name)


def load_asr(model_name: str):
    """Load any supported ASR model."""
    from nemo_model import _load_model

    return _load_model(resolve_asr_model(model_name), precision="fp32", device="cuda")


def transcribe_chunk(model, model_name: str, audio: Path, source_language: str) -> list[dict]:
    """Transcribe one chunk into cue-level segments with word timestamps."""
    name = (model_name or "").lower()

    from parakeet_ultra import _is_parakeet_ultra, _transcribe_parakeet_ultra
    from qwen3_asr import _is_qwen3_asr, _transcribe_qwen3_asr

    if _is_parakeet_ultra(model_name):
        _, segments = _transcribe_parakeet_ultra(model, str(audio), 0.0)
        return segments
    if _is_qwen3_asr(model_name):
        _, segments = _transcribe_qwen3_asr(model, str(audio), 0.0, source_language)
        return segments
    if "canary" in name:
        from local_subtitles import infer_chunk

        return infer_chunk(model, audio, source_language, source_language, 0.0, 1e9)
    from nemo_model import _transcribe_parakeet

    _, segments = _transcribe_parakeet(model, str(audio), 0.0)
    return segments


def _prepare_audio(video: Path, cache: Path, fingerprint: dict) -> Path:
    from batch_subtitles import extract_audio

    audio = cache / f"{video.stem}.wav"
    metadata = cache / f"{video.stem}.audio.json"
    if not audio.exists() or not metadata.exists() or json.loads(metadata.read_text()) != fingerprint:
        print(f"Preparing full audio: {video.name}", flush=True)
        temporary = audio.with_suffix(".tmp.wav")
        temporary.unlink(missing_ok=True)
        extract_audio(video, temporary)
        temporary.replace(audio)
        save_json(metadata, fingerprint)
    return audio


def _slice_chunk(source: wave.Wave_read, cache: Path, start: float, end: float) -> Path:
    chunk_path = cache / "current_chunk.wav"
    source.setpos(round(start * source.getframerate()))
    with wave.open(str(chunk_path), "wb") as chunk:
        chunk.setparams(source.getparams())
        chunk.writeframes(source.readframes(round((end - start) * source.getframerate())))
    return chunk_path


def run(plan: list[dict], source_language: str, report: Path,
        asr_model: str | None = None, translator: str | None = None,
        fit_budget: bool = True, tts_engine: str | None = None) -> None:
    import torch
    from batch_subtitles import probe_duration, srt_stats

    asr_model = resolve_asr_model(asr_model)
    mt_model = resolve_mt_model(translator)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    cache = report.parent / ".local_asr"
    cache.mkdir(exist_ok=True)

    def report_progress():
        save_json(report, {"backend": "local", "source_language": source_language,
                           "asr_model": asr_model, "mt_model": mt_model,
                           "precision": "fp32", "jobs": plan})

    report_progress()
    print(f"ASR: {asr_model} | MT: {mt_model}", flush=True)
    model = load_asr(asr_model)
    translator_client = OllamaTranslator(model=mt_model)

    for filename in dict.fromkeys(job["video"] for job in plan):
        jobs = [j for j in plan if j["video"] == filename]
        pending = [j for j in jobs if j["status"] == "pending"]
        if not pending:
            continue
        video = Path(filename)
        fingerprint = {"bytes": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns}
        audio = _prepare_audio(video, cache, fingerprint)
        duration = probe_duration(video)
        boundaries = chunk_boundaries(audio)

        # One ASR pass per video, shared by every target language.
        asr_path = cache / f"{video.stem}.asr.json"
        identity = {**fingerprint, "model": asr_model, "source": source_language,
                    "boundaries": [list(b) for b in boundaries]}
        checkpoint = {"identity": identity, "chunks": {}}
        if asr_path.exists():
            previous = json.loads(asr_path.read_text())
            if previous.get("identity") == identity:
                checkpoint = previous

        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with wave.open(str(audio)) as source:
            for index, (start, end) in enumerate(boundaries):
                if str(index) in checkpoint["chunks"]:
                    continue
                chunk_path = _slice_chunk(source, cache, start, end)
                segments = transcribe_chunk(model, asr_model, chunk_path, source_language)
                for segment in segments:
                    segment["start"] = round(float(segment["start"]) + start, 3)
                    segment["end"] = round(min(end, float(segment["end"]) + start), 3)
                checkpoint["chunks"][str(index)] = segments
                save_json(asr_path, checkpoint)
                print(f"{video.name} asr {index + 1}/{len(boundaries)} "
                      f"({end:.0f}/{duration:.0f}s audio)", flush=True)
                torch.cuda.empty_cache()

        cues = [s for i in range(len(boundaries)) for s in checkpoint["chunks"].get(str(i), [])]
        cues = [{**c, "text": (c.get("text") or "").strip()} for c in cues if (c.get("text") or "").strip()]
        if not cues:
            raise RuntimeError(f"ASR produced no text for {video.name}")
        peak_asr = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        print(f"{video.name}: {len(cues)} cues "
              f"(max {max(len(c['text']) for c in cues)} chars)", flush=True)

        rows_by_language: dict[str, list[dict]] = {}
        for job in pending:
            language = job["language"]
            job.update(status="running", chunks_total=len(boundaries),
                       chunks_done=len(checkpoint["chunks"]))
            report_progress()
            job_started = time.monotonic()
            if language == source_language:
                rows = cues
            else:
                rows = translate_cues(
                    cues, source_language, language, translator_client,
                    fit_budget=fit_budget,
                    engine=tts_engine,
                    progress=lambda n, total, text: print(f"  {video.name} [{language}] "
                                                          f"{n}/{total} {text!r}", flush=True))
            rows_by_language[language] = rows
            output = render_srt(rows)
            stats = srt_stats(output, duration)
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            with Path(job["output"]).open("xb") as destination:
                destination.write(output)
            job.update(stats, status="complete",
                       elapsed_seconds=round(time.monotonic() - job_started, 1),
                       peak_vram_gib=peak_asr)
            report_progress()
            print(f"Saved {Path(job['output']).name}: {stats['cues']} cues", flush=True)

        # Always: one combined file with every produced language.
        stem = Path(pending[0]["output"]).with_suffix("")
        stem = stem.parent / video.stem
        written = write_files(rows_by_language, stem, order=list(rows_by_language))
        print(f"Combined: {', '.join(p.name for p in written)}", flush=True)

    del model
    translator_client.unload()
    gc.collect()
    torch.cuda.empty_cache()
