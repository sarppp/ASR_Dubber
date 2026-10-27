#!/usr/bin/env python3
"""Generate EN/FR/DE subtitles through NeMo, on Modal or a local NVIDIA GPU.

From nemo/:
  uv run --no-sync --env-file .env python batch_subtitles.py --input-dir new_ones
  python3 batch_subtitles.py --input-dir new_ones --dry-run
  .venv-local-asr/bin/python batch_subtitles.py --backend local --input-dir new_ones

Each video is transcribed once in the source language, and Canary translates
the audio directly into each other language. Valid existing SRTs are skipped.
Only 16 kHz mono audio is uploaded; the videos remain untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


MEDIA_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wav", ".mp3", ".flac", ".m4a", ".ogg"}
TIMESTAMP = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d),(\d{3}) --> (\d{2,}):([0-5]\d):([0-5]\d),(\d{3})$")


def srt_stats(data: bytes, duration: float | None = None) -> dict:
    """Reject empty, malformed, unordered, or out-of-bounds subtitle results."""
    text = data.decode("utf-8-sig").replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("Empty subtitles")
    previous = -1
    first = None
    last = 0
    blocks = re.split(r"\n\s*\n", text)
    for index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(index):
            raise ValueError(f"Malformed cue {index}")
        match = TIMESTAMP.fullmatch(lines[1].strip())
        if not match or not " ".join(lines[2:]).strip():
            raise ValueError(f"Missing timestamp or text in cue {index}")
        h, m, s, ms, eh, em, es, ems = map(int, match.groups())
        start = ((h * 60 + m) * 60 + s) * 1000 + ms
        end = ((eh * 60 + em) * 60 + es) * 1000 + ems
        if start < previous or end <= start:
            raise ValueError(f"Invalid timing in cue {index}")
        if duration is not None and end > (duration + 1) * 1000:
            raise ValueError(f"Cue {index} extends beyond the media duration")
        previous = start
        first = start if first is None else first
        last = max(last, end)
    return {"cues": len(blocks), "first_seconds": first / 1000, "last_seconds": last / 1000}


def make_plan(directory: Path, languages: list[str]) -> list[dict]:
    videos = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )
    if not videos:
        raise ValueError(f"No media files found in {directory}")
    stems = [p.stem for p in videos]
    if len(set(stems)) != len(stems):
        raise ValueError("Media with the same stem would overwrite the same SRT; use distinct names")
    plan = []
    for video in videos:
        for language in dict.fromkeys(languages):
            output = video.with_name(f"{video.stem}.nemo.{language}.srt")
            job = {"video": str(video), "language": language, "output": str(output), "status": "pending"}
            if output.exists():
                # Refuse to overwrite damaged files silently; preserve them for inspection.
                job.update(srt_stats(output.read_bytes()), status="existing")
            plan.append(job)
    return plan


def probe_duration(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_audio(video: Path, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:a:0",
         "-vn", "-ac", "1", "-ar", "16000",
         "-af", "aresample=async=1:first_pts=0", "-c:a", "pcm_s16le", str(output)],
        check=True,
    )


async def run_jobs(plan: list[dict], source_language: str, workers: int, report: Path) -> None:
    from nemo_modal_app import app, transcribe_nemo_remote, MODEL_CANARY, select_nemo_model

    semaphore = asyncio.Semaphore(workers)

    def save_report():
        temporary = report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"source_language": source_language, "jobs": plan}, indent=2) + "\n")
        temporary.replace(report)

    save_report()
    async with app.run.aio():
        for filename in dict.fromkeys(job["video"] for job in plan):
            jobs = [job for job in plan if job["video"] == filename and job["status"] == "pending"]
            if not jobs:
                continue
            video = Path(filename)
            print(f"Preparing audio: {video.name}", flush=True)
            try:
                duration = probe_duration(video)
                with tempfile.TemporaryDirectory(prefix="nemo-subtitles-") as work:
                    wav = Path(work) / f"{video.stem}.wav"
                    extract_audio(video, wav)
                    audio = wav.read_bytes()
                    print(f"Audio: {duration / 60:.1f} minutes, {len(audio) / 1024**2:.1f} MiB", flush=True)

                    async def generate(job):
                        async with semaphore:
                            language = job["language"]
                            job["status"] = "running"
                            save_report()
                            print(f"Starting {video.name}: {language}", flush=True)
                            try:
                                result = await transcribe_nemo_remote.remote.aio(
                                    video_filename=wav.name, video_data=audio,
                                    language=source_language,
                                    nemo_model=select_nemo_model(source_language) if language == source_language else MODEL_CANARY,
                                    target_lang=None if language == source_language else language,
                                    precision="fp16" if os.environ.get("NEMO_GPU") == "T4" else "bf16", diarize=False,
                                    # Bound memory for feature-length movies and avoid repeated calibration.
                                    chunk_override_sec=120,
                                )
                                destination = Path(job["output"])
                                try:
                                    stats = srt_stats(result, duration)
                                except ValueError:
                                    # Keep costly GPU output available for diagnosis or repair.
                                    rejected = destination.with_suffix(".rejected.srt")
                                    with rejected.open("xb") as output:
                                        output.write(result)
                                    raise
                                # Exclusive creation also protects results from another local run.
                                with destination.open("xb") as output:
                                    output.write(result)
                                job.update(stats, status="complete", duration_seconds=duration)
                                print(f"Saved {destination.name}: {stats['cues']} cues", flush=True)
                            except Exception as exc:
                                job.update(status="failed", error=str(exc))
                                print(f"FAILED {video.name} ({language}): {exc}", flush=True)
                            save_report()

                    await asyncio.gather(*(generate(job) for job in jobs))
            except Exception as exc:
                for job in jobs:
                    if job["status"] == "pending":
                        job.update(status="failed", error=str(exc))
                save_report()
                print(f"FAILED preparing {video.name}: {exc}", flush=True)
    failures = [job for job in plan if job["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} subtitle jobs failed; see {report}. Rerun to resume.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("new_ones"))
    parser.add_argument("--backend", choices=["modal", "local"], default="modal")
    parser.add_argument("--source-language", choices=["en", "fr", "de", "es"], default="en")
    parser.add_argument("--languages", nargs="+", choices=["en", "fr", "de"], default=["en", "fr", "de"])
    parser.add_argument("--jobs", type=int, default=2, help="Maximum simultaneous Modal GPU calls (default: 2)")
    parser.add_argument("--gpu", choices=["T4", "A10G", "A100", "L4"], default="T4", help="Modal GPU type (default: T4)")
    parser.add_argument("--dry-run", action="store_true", help="List planned outputs without contacting Modal")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    # Canary translation supports English <-> other languages, not arbitrary pairs.
    if args.source_language != "en" and any(lang not in ("en", args.source_language) for lang in args.languages):
        parser.error("Direct translation requires English as source or target")
    directory = args.input_dir.resolve()
    try:
        plan = make_plan(directory, args.languages)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for job in plan:
        print(f"{job['status']:8s} {Path(job['output']).name}", flush=True)
    for marker in sorted(directory.glob("*:Zone.Identifier")):
        video = marker.with_name(marker.name.removesuffix(":Zone.Identifier"))
        if not video.exists():
            print(f"Missing video (metadata only): {video.name}", flush=True)
    if args.dry_run or all(job["status"] == "existing" for job in plan):
        return
    report = directory / ("subtitle_local_report.json" if args.backend == "local" else "subtitle_batch_report.json")
    try:
        if args.backend == "local":
            from local_subtitles import run_local_jobs
            run_local_jobs(plan, args.source_language, report)
        else:
            import modal
            os.environ["NEMO_GPU"] = args.gpu
            with modal.enable_output():
                asyncio.run(run_jobs(plan, args.source_language, args.jobs, report))
    except Exception as exc:
        for job in plan:
            if job["status"] == "running":
                job.update(status="failed", error=str(exc))
        report.write_text(json.dumps({
            "source_language": args.source_language, "backend": args.backend,
            "status": "failed", "error": str(exc), "jobs": plan,
        }, indent=2) + "\n")
        parser.exit(1, f"Batch stopped: {exc}\nReport: {report}\n")


if __name__ == "__main__":
    main()
