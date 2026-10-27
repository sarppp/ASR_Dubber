"""Sequential Canary inference for small local GPUs, with chunk checkpoints."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import subprocess
import textwrap
import time
import wave
from pathlib import Path

MODEL = "nvidia/canary-1b-v2"


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def chunk_boundaries(audio: Path, maximum: float = 25.0) -> list[tuple[float, float]]:
    """Prefer quiet boundaries so short inference chunks don't split speech."""
    with wave.open(str(audio)) as wav:
        duration = wav.getnframes() / wav.getframerate()
    scan = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(audio),
         "-af", "silencedetect=noise=-35dB:d=0.25", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    pauses = [float(end) - float(length) / 2 for end, length in re.findall(
        r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", scan.stderr
    )]
    chunks, start = [], 0.0
    while start < duration:
        end = min(start + maximum, duration)
        if end < duration:
            candidates = [p for p in pauses if start + maximum * 0.5 <= p <= end]
            if candidates:
                end = candidates[-1]
        chunks.append((start, end))
        start = end
    return chunks


def load_canary():
    import torch
    from nemo_model import _import_nemo_asr
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU is unavailable; run outside the restricted sandbox")
    asr = _import_nemo_asr()
    from nemo import __version__ as nemo_version
    if tuple(map(int, nemo_version.split(".")[:2])) < (2, 5):
        raise RuntimeError("Use .venv-local-asr/bin/python: translated timestamps require NeMo >=2.5")
    checkpoint = hf_hub_download(MODEL, filename="canary-1b-v2.nemo", local_files_only=True)
    print(f"Loading cached {MODEL} on CPU, then transferring FP16 weights to {torch.cuda.get_device_name(0)}", flush=True)
    model = asr.models.ASRModel.restore_from(checkpoint, map_location="cpu")
    model = model.half().cuda().eval()
    # NeMo deliberately does not register this auxiliary model as a child module.
    # Convert it explicitly, otherwise translation alignment uses FP32 VRAM.
    aligner = getattr(model, "timestamps_asr_model", None)
    if aligner is None:
        raise RuntimeError("Cached Canary checkpoint does not contain its timestamp alignment model")
    aligner.half().cuda().eval()
    torch.cuda.empty_cache()
    print(f"Model and aligner: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated", flush=True)
    return model


def infer_chunk(model, audio: Path, source: str, target: str, offset: float, end: float) -> list[dict]:
    import torch
    from nemo_audio import _words_to_segs, _split_coarse_segs

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        hypotheses = model.transcribe(
            [str(audio)], batch_size=1, num_workers=0, verbose=False,
            source_lang=source, target_lang=target, timestamps=True, return_hypotheses=True,
        )
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    hyp = hypotheses[0]
    used_fallback = False
    # Music/silence can produce punctuation without any alignable speech.
    if not any(character.isalnum() for character in hyp.text):
        return []
    stamps = getattr(hyp, "timestamp", None) or {}
    if source == target and not (stamps.get("word") or stamps.get("segment")):
        # Long vocalizations can make Canary's generated text impossible to
        # force-align. For source-language captions, use the already-loaded
        # CTC model's own transcription and timestamps instead of inventing times.
        print(f"Automatic CTC transcription fallback at {offset:.2f}s", flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            fallback = model.timestamps_asr_model.transcribe(
                [str(audio)], batch_size=1, num_workers=0, verbose=False,
                timestamps=True, return_hypotheses=True,
            )
        if isinstance(fallback, tuple):
            fallback = fallback[0]
        hyp = fallback[0]
        if not any(character.isalnum() for character in hyp.text):
            return []
        stamps = getattr(hyp, "timestamp", None) or {}
        used_fallback = True
    if source == target and stamps.get("word"):
        segments = _words_to_segs(stamps["word"], max_w=12, max_ch=80)
    else:
        segments = [{"text": item["segment"], "start": item["start"], "end": item["end"]}
                    for item in stamps.get("segment", [])]
        segments = _split_coarse_segs(segments, max_w=12, max_ch=80)
    if not segments:
        raise RuntimeError(f"Canary returned speech without timestamps at {offset:.2f}s: {hyp.text[:120]!r}")
    result = []
    for segment in segments:
        start = max(offset, offset + float(segment["start"]))
        stop = min(end, offset + float(segment["end"]))
        if stop <= start:
            continue
        result.append({"start": round(start, 3), "end": round(stop, 3), "text": segment["text"].strip(),
                       **({"transcription_fallback": "ctc"} if used_fallback else {})})
    return result


def render_srt(segments: list[dict]) -> bytes:
    from nemo_audio import _fmt_ts
    blocks = []
    for i, segment in enumerate(sorted(segments, key=lambda s: (s["start"], s["end"])), 1):
        text = "\n".join(textwrap.wrap(segment["text"], width=42, break_long_words=False, break_on_hyphens=False))
        blocks.append(f"{i}\n{_fmt_ts(segment['start'])} --> {_fmt_ts(segment['end'])}\n{text}\n")
    return "\n".join(blocks).encode("utf-8")


def run_local_jobs(plan: list[dict], source_language: str, report: Path) -> None:
    from batch_subtitles import extract_audio, probe_duration, srt_stats
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Cached NVIDIA .nemo archives use trusted full checkpoints.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    cache = report.parent / ".local_asr"
    cache.mkdir(exist_ok=True)

    def report_progress():
        save_json(report, {"backend": "local", "source_language": source_language,
                           "model": MODEL, "precision": "fp16", "jobs": plan})

    report_progress()
    model = load_canary()
    for filename in dict.fromkeys(job["video"] for job in plan):
        jobs = [j for j in plan if j["video"] == filename and j["status"] == "pending"]
        if not jobs:
            continue
        video = Path(filename)
        fingerprint = {"bytes": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns}
        audio = cache / f"{video.stem}.wav"
        metadata = cache / f"{video.stem}.audio.json"
        if not audio.exists() or not metadata.exists() or json.loads(metadata.read_text()) != fingerprint:
            print(f"Preparing full audio: {video.name}", flush=True)
            temporary = audio.with_suffix(".tmp.wav")
            temporary.unlink(missing_ok=True)
            extract_audio(video, temporary)
            temporary.replace(audio)
            save_json(metadata, fingerprint)
        duration = probe_duration(video)
        boundaries = chunk_boundaries(audio)
        for job in jobs:
            language = job["language"]
            checkpoint_path = cache / f"{video.stem}.{language}.json"
            identity = {**fingerprint, "model": MODEL, "source": source_language, "target": language,
                        "boundaries": [list(b) for b in boundaries]}
            checkpoint = {"identity": identity, "chunks": {}}
            if checkpoint_path.exists():
                previous = json.loads(checkpoint_path.read_text())
                if previous.get("identity") == identity:
                    checkpoint = previous
            job.update(status="running", chunks_total=len(boundaries), chunks_done=len(checkpoint["chunks"]))
            report_progress()
            print(f"Starting {video.name} [{language}], {len(boundaries)} chunks", flush=True)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            with wave.open(str(audio)) as source:
                for index, (start, end) in enumerate(boundaries):
                    if str(index) in checkpoint["chunks"]:
                        continue
                    chunk_path = cache / "current_chunk.wav"
                    source.setpos(round(start * source.getframerate()))
                    with wave.open(str(chunk_path), "wb") as chunk:
                        chunk.setparams(source.getparams())
                        chunk.writeframes(source.readframes(round((end - start) * source.getframerate())))
                    checkpoint["chunks"][str(index)] = infer_chunk(model, chunk_path, source_language, language, start, end)
                    save_json(checkpoint_path, checkpoint)
                    job["chunks_done"] = len(checkpoint["chunks"])
                    report_progress()
                    torch.cuda.empty_cache()
                    print(f"{video.name} [{language}] {index + 1}/{len(boundaries)} ({end:.0f}/{duration:.0f}s audio)", flush=True)
            segments = [s for i in range(len(boundaries)) for s in checkpoint["chunks"][str(i)]]
            output = render_srt(segments)
            stats = srt_stats(output, duration)
            with Path(job["output"]).open("xb") as destination:
                destination.write(output)
            job.update(stats, status="complete", elapsed_seconds=round(time.monotonic() - started, 1),
                       peak_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 2))
            report_progress()
            print(f"Saved {Path(job['output']).name}: {stats['cues']} cues", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
