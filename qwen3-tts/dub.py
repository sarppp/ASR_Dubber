#!/usr/bin/env python3
"""
dub.py — Video dubbing pipeline (NeMo SRT + Qwen TTS)
======================================================

Full workflow:
  1. nemo.py --diarize      → video.nemo.de.diarize.srt
  2. translate.py           → video.nemo.de.diarize_fr.srt   (Gemma via Ollama)
  3. dub.py (this script)   → final_dub.mp4

Usage:
  # With background music preservation (demucs):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt

  # Without demucs (faster, replaces full audio track):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --no-demucs

  # Voice cloning mode:
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --qwen-mode clone

  # Clone but skip demucs (use original video audio for voice refs):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --qwen-mode clone --no-demucs
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Voice tables
# ---------------------------------------------------------------------------
QWEN_FEMALE_VOICES = ["vivian", "ono_anna", "Chelsie"]
QWEN_MALE_VOICES   = ["ryan", "ethan"]

# Qwen TTS requires full lowercase language names, not ISO codes
LANG_CODE_TO_QWEN = {
    "fr": "french",  "en": "english", "de": "german",  "es": "spanish",
    "it": "italian", "ja": "japanese","ko": "korean",  "pt": "portuguese",
    "ru": "russian", "zh": "chinese", "auto": "auto",
}

def _qwen_lang(code: str) -> str:
    """Convert ISO code to Qwen language name, e.g. 'fr' → 'french'."""
    code = code.strip().lower()
    name = LANG_CODE_TO_QWEN.get(code, code)
    if name not in LANG_CODE_TO_QWEN.values():
        log.warning(f"Unknown language code '{code}' — passing as-is. "
                    f"Supported: {sorted(LANG_CODE_TO_QWEN.values())}")
    return name


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def _srt_ts(t: str) -> float:
    """HH:MM:SS,mmm → seconds."""
    t = t.strip().replace(",", ".")
    h, m, s = t.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def parse_srt(path: Path) -> List[Dict]:
    """
    Parse a diarized (and already-translated) SRT.

    Handles both formats produced by the pipeline:
      [Speaker 2] Bonjour le monde          ← nemo.py --diarize style
      [Speaker 2] Bonjour le monde          ← translate.py preserves the tag

    Returns list of:
      {"index": int, "start": float, "end": float, "speaker": str, "text": str}
    """
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []

    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1],
        )
        if not ts_match:
            continue

        start = _srt_ts(ts_match.group(1))
        end   = _srt_ts(ts_match.group(2))

        # Join continuation lines (translate.py sometimes wraps long lines)
        raw_text = " ".join(lines[2:]).strip()

        # Extract [Speaker N] label
        spk_match = re.match(r"\[([^\]]+)\]\s*(.*)", raw_text, re.DOTALL)
        if spk_match:
            speaker = spk_match.group(1).strip()
            text    = spk_match.group(2).strip()
        else:
            speaker = "Speaker 1"
            text    = raw_text

        # Restore any pipe-encoded newlines that translate.py may have left
        text = text.replace(" | ", " ").replace("|", " ").strip()

        if not text:
            continue

        segments.append({
            "index":   idx,
            "start":   start,
            "end":     end,
            "speaker": speaker,
            "text":    text,
        })

    log.info(f"Parsed {len(segments)} segments from SRT")
    speakers = sorted({s["speaker"] for s in segments})
    log.info(f"Speakers found: {speakers}")
    return segments


# ---------------------------------------------------------------------------
# Audio extraction (raw, no separation)
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path) -> None:
    """Extract full mono 16 kHz WAV from video (used for clone refs when --no-demucs)."""
    if out_wav.exists():
        log.info(f"✓ Reusing {out_wav.name}")
        return
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(out_wav), "-y", "-loglevel", "error"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Demucs vocal separation (optional)
# ---------------------------------------------------------------------------

def separate_audio(video_path: Path, temp_dir: Path) -> Tuple[Path, Optional[Path]]:
    """
    Run demucs htdemucs to split vocals from background.
    Returns (vocals_path, background_path).
    """
    demucs_out = temp_dir / "demucs_out"
    raw_wav    = temp_dir / "input_raw.wav"
    # demucs names the output folder after the input file stem
    base   = demucs_out / "htdemucs" / raw_wav.stem
    vocals = base / "vocals.wav"
    bg     = base / "no_vocals.wav"

    if vocals.exists() and bg.exists():
        log.info("✓ Reusing existing demucs separation")
        return vocals, bg

    log.info("🎶 Separating vocals with demucs…")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         str(raw_wav), "-y", "-loglevel", "error"],
        check=True,
    )
    subprocess.run(
        ["demucs", "-n", "htdemucs", "--two-stems=vocals",
         str(raw_wav), "-o", str(demucs_out)],
        check=True,
    )
    if not vocals.exists():
        raise FileNotFoundError(
            f"demucs did not produce vocals at {vocals}\n"
            f"Check what folder demucs actually created under {demucs_out / 'htdemucs'}"
        )
    return vocals, bg


# ---------------------------------------------------------------------------
# Clone reference extraction
# ---------------------------------------------------------------------------

def extract_clone_refs(
    segments: List[Dict],
    audio_source: Path,   # vocals (demucs) or raw video audio (no-demucs)
    cast_dir: Path,
) -> Dict[str, Path]:
    """
    For each speaker, extract their longest segment from audio_source
    as a reference WAV for Qwen clone mode.
    Returns {speaker: wav_path}
    """
    cast_dir.mkdir(parents=True, exist_ok=True)

    # Longest segment per speaker
    best: Dict[str, Tuple[float, float, float]] = {}
    for seg in segments:
        spk = seg["speaker"]
        dur = max(0.0, seg["end"] - seg["start"])
        if dur > 0 and (spk not in best or dur > best[spk][0]):
            best[spk] = (dur, seg["start"], seg["end"])

    refs: Dict[str, Path] = {}
    log.info("🎙️  Extracting clone reference WAVs…")
    for spk, (dur, start, end) in best.items():
        safe_name = re.sub(r"[^\w\-]", "_", spk)
        out_wav   = cast_dir / f"{safe_name}.wav"

        if out_wav.exists() and out_wav.stat().st_size > 1000:
            log.info(f"   ✓ {spk}: reusing {out_wav.name}")
            refs[spk] = out_wav
            continue

        if dur < 1.0:
            log.warning(f"   ⚠️  {spk}: longest segment only {dur:.2f}s — too short for clone")
            continue

        log.info(f"   → {spk}: {dur:.2f}s @ {start:.2f}–{end:.2f}s")
        subprocess.run(
            ["ffmpeg", "-ss", str(start), "-t", str(dur),
             "-i", str(audio_source),
             "-ac", "1", "-ar", "16000", "-y", str(out_wav), "-loglevel", "error"],
            check=True,
        )
        if out_wav.exists() and out_wav.stat().st_size > 1000:
            refs[spk] = out_wav

    return refs


# ---------------------------------------------------------------------------
# Voice assignment (custom mode)
# ---------------------------------------------------------------------------

def build_voice_map(segments: List[Dict]) -> Dict[str, str]:
    """Assign a Qwen voice to each speaker, alternating female/male pools."""
    seen: List[str] = []
    for seg in segments:
        if seg["speaker"] not in seen:
            seen.append(seg["speaker"])

    voice_map: Dict[str, str] = {}
    fi = mi = 0
    for i, spk in enumerate(seen):
        if i % 2 == 0:
            voice_map[spk] = QWEN_FEMALE_VOICES[fi % len(QWEN_FEMALE_VOICES)]
            fi += 1
        else:
            voice_map[spk] = QWEN_MALE_VOICES[mi % len(QWEN_MALE_VOICES)]
            mi += 1

    log.info("🎤 Voice assignments:")
    for spk, voice in voice_map.items():
        log.info(f"   {spk} → {voice}")
    return voice_map


# ---------------------------------------------------------------------------
# Qwen TTS (subprocess into qwen3-tts uv env)
# ---------------------------------------------------------------------------

def _qwen_python(qwen_project_dir: Path) -> str:
    venv_python = qwen_project_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    log.warning(f"qwen3-tts .venv not found at {venv_python}, falling back to 'python'")
    return "python"


def _qwen_worker(script_dir: Path) -> str:
    worker = script_dir / "qwen_tts_worker.py"
    if worker.exists():
        return str(worker)
    raise FileNotFoundError(
        f"qwen_tts_worker.py not found at {script_dir}. "
        "It should sit next to dub.py in the qwen3-tts folder."
    )


def generate_tts_custom(
    text: str, voice: str, language: str,
    output: Path, qwen_python: str, qwen_worker: str,
) -> bool:
    result = subprocess.run(
        [qwen_python, qwen_worker,
         "--text", text, "--output", str(output),
         "--mode", "custom", "--voice", voice, "--language", language],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"Qwen custom TTS failed:\n{result.stderr[-600:]}")
        return False
    return output.exists() and output.stat().st_size > 500


def generate_tts_clone(
    text: str, ref_audio: Path, language: str,
    output: Path, qwen_python: str, qwen_worker: str,
) -> bool:
    result = subprocess.run(
        [qwen_python, qwen_worker,
         "--text", text, "--output", str(output),
         "--mode", "clone", "--ref-audio", str(ref_audio), "--language", language],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"Qwen clone TTS failed:\n{result.stderr[-600:]}")
        return False
    return output.exists() and output.stat().st_size > 500


# ---------------------------------------------------------------------------
# Speed-fit audio clip to a target duration
# ---------------------------------------------------------------------------

def _audio_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def speed_fit(audio_path: Path, target_dur: float, max_speed: float = 1.35) -> Path:
    """
    Fit audio_path into target_dur seconds.
    - Too short → pad tail with silence  (keeps natural cadence)
    - Too long  → speed up, capped at max_speed  (avoids chipmunk)
    """
    curr = _audio_duration(audio_path)
    if curr <= 0:
        return audio_path

    out   = audio_path.with_name(audio_path.stem + "_fit.wav")
    ratio = curr / target_dur

    if ratio < 0.95:
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", f"apad,atrim=0:{target_dur:.6f}",
             "-y", str(out), "-loglevel", "error"],
            check=True,
        )
    else:
        speed = min(ratio, max_speed)
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-filter:a", f"atempo={speed:.4f}",
             "-vn", "-y", str(out), "-loglevel", "error"],
            check=True,
        )

    return out if out.exists() else audio_path


# ---------------------------------------------------------------------------
# Stitch + mix
# ---------------------------------------------------------------------------

def stitch_and_mix(
    final_files: List[Tuple[Path, float, float]],
    video_path: Path,
    output_dir: Path,
    temp_dir: Path,
    background: Optional[Path] = None,   # None when --no-demucs
    trim_to: Optional[float] = None,     # trim video to this many seconds (from SRT end)
) -> Path:
    """
    Concatenate dubbed clips with silence gaps → dub track.
    Then mix over video:
      - With demucs:    dub (loud) + background music (quiet) + original video
      - Without demucs: dub track replaces audio entirely
    If trim_to is set, the output video is trimmed to that duration.
    """
    concat_list = temp_dir / "concat.txt"
    cur = 0.0

    with open(concat_list, "w") as f:
        for clip_path, start, end in final_files:
            if not clip_path.exists():
                log.warning(f"Missing clip, skipping: {clip_path}")
                continue
            gap = start - cur
            if gap > 0.05:
                sil = temp_dir / f"sil_{cur:.3f}.wav"
                subprocess.run(
                    f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap:.6f}'
                    f' "{sil}" -y -loglevel error',
                    shell=True, check=True,
                )
                f.write(f"file '{sil.resolve()}'\n")
            f.write(f"file '{clip_path.resolve()}'\n")
            cur = end

    dub_track = output_dir / "dub_track.wav"
    subprocess.run(
        f'ffmpeg -f concat -safe 0 -i "{concat_list}" -c copy "{dub_track}" -y -loglevel error',
        shell=True, check=True,
    )

    final = output_dir / "final_dub.mp4"

    # Build trim flag if needed (re-encodes video to allow cutting)
    if trim_to:
        log.info(f"✂️  Trimming output video to {trim_to:.2f}s (matches SRT duration)")
        trim_flags  = ["-t", str(trim_to)]
        video_codec = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
    else:
        trim_flags  = []
        video_codec = ["-c:v", "copy"]

    if background and background.exists():
        subprocess.run(
            ["ffmpeg",
             "-i", str(video_path),
             "-i", str(dub_track),
             "-i", str(background),
             *trim_flags,
             "-filter_complex",
             "[1:a]volume=1.5[v];[2:a]volume=0.4[b];[v][b]amix=inputs=2:duration=first[out]",
             "-map", "0:v", "-map", "[out]",
             *video_codec, str(final), "-y", "-loglevel", "error"],
            check=True,
        )
    else:
        subprocess.run(
            ["ffmpeg",
             "-i", str(video_path),
             "-i", str(dub_track),
             *trim_flags,
             "-map", "0:v", "-map", "1:a",
             *video_codec, str(final), "-y", "-loglevel", "error"],
            check=True,
        )

    return final


# ---------------------------------------------------------------------------
# Checkpoint (save progress so a crash doesn't lose completed segments)
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, final_files: list) -> None:
    data = [{"clip": str(clip), "start": start, "end": end}
            for clip, start, end in final_files]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = []
        for entry in data:
            clip = Path(entry["clip"])
            if clip.exists() and clip.stat().st_size > 500:
                result.append((clip, float(entry["start"]), float(entry["end"])))
        if result:
            log.info(f"✓ Loaded checkpoint: {len(result)} segments already done")
        return result
    except Exception as e:
        log.warning(f"Could not load checkpoint: {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dub a video from a pre-translated diarized SRT using Qwen TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Full pipeline:
  1. uv run --project ../nemo python nemo.py video.mp4 --language de --diarize
  2. python translate.py   (Gemma via Ollama → translated SRT)
  3. uv run python dub.py video.mp4 translated.srt --language fr

Examples:
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --qwen-mode clone
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --no-demucs
        """,
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Input video file (auto-discovered in --search-dir if omitted)")
    parser.add_argument("srt",   nargs="?", default=None,
                        help="Pre-translated diarized SRT (auto-discovered if omitted)")
    parser.add_argument("--language",   default="fr",
                        help="Target language to pass to Qwen TTS (default: fr)")
    parser.add_argument("--qwen-mode",  default="clone", choices=["custom", "clone"],
                        help="'custom' = fixed voice | 'clone' = voice cloned from speaker (default: clone)")
    parser.add_argument("--no-demucs",  action="store_true",
                        help="Skip vocal separation — faster but loses background music")
    parser.add_argument("--search-dir", default="../nemo",
                        help="Folder to auto-discover video + SRT from (default: ../nemo)")
    parser.add_argument("--workdir",    default="output/dub",
                        help="Working directory for intermediate files (default: output/dub)")
    parser.add_argument("--qwen-dir",   default=".",
                        help="Path to the qwen3-tts uv project (default: current folder)")
    parser.add_argument("--max-speed",  type=float, default=1.35,
                        help="Max TTS speed-up before capping (default: 1.35)")
    args = parser.parse_args()

    search_dir = Path(args.search_dir).resolve()
    VIDEO_EXT  = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    # ── Auto-discover video + SRT ─────────────────────────────────────────────
    if args.video is None or args.srt is None:
        if not search_dir.exists():
            log.error(f"Search dir not found: {search_dir}"); return 1

        # Find translated SRTs: *.diarize_??.srt
        srt_candidates = sorted(search_dir.glob("*.diarize_??.srt"))
        if not srt_candidates and args.srt is None:
            log.error(
                f"No translated SRTs found in {search_dir} — "
                "expected pattern: *.nemo.LANG.diarize_TARGETLANG.srt"
            )
            return 1
            return 1

        chosen_srt = Path(args.srt).resolve() if args.srt else srt_candidates[0]
        if len(srt_candidates) > 1 and args.srt is None:
            log.info(f"Multiple SRTs found, using: {chosen_srt.name}")

        # Derive video stem: impost_trimmed_2min.nemo.de.diarize_fr.srt → impost_trimmed_2min
        stem_match = re.match(r"^(.+?)\.nemo\.", chosen_srt.name)
        video_stem = stem_match.group(1) if stem_match else None

        if args.video:
            chosen_video = Path(args.video).resolve()
        elif video_stem:
            matches = [f for f in search_dir.iterdir()
                       if f.stem == video_stem and f.suffix.lower() in VIDEO_EXT]
            if not matches:
                log.error(f"No video found for stem '{video_stem}' in {search_dir}"); return 1
            chosen_video = matches[0]
        else:
            videos = [f for f in search_dir.iterdir() if f.suffix.lower() in VIDEO_EXT]
            if not videos:
                log.error(f"No video files found in {search_dir}"); return 1
            chosen_video = videos[0]

        video_path = chosen_video.resolve()
        srt_path   = chosen_srt.resolve()
    else:
        video_path = Path(args.video).resolve()
        srt_path   = Path(args.srt).resolve()

    if not video_path.exists():
        log.error(f"Video not found: {video_path}"); return 1
    if not srt_path.exists():
        log.error(f"SRT not found: {srt_path}"); return 1

    qwen_dir   = Path(args.qwen_dir).resolve()
    work_dir   = Path(args.workdir).resolve()
    temp_dir   = work_dir / "temp"
    output_dir = work_dir / "output"
    cast_dir   = work_dir / "cast_samples"

    for d in (work_dir, temp_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    script_dir  = Path(__file__).resolve().parent
    qwen_python = _qwen_python(qwen_dir)
    qwen_worker = _qwen_worker(script_dir)

    log.info("=" * 60)
    log.info(f"Video        : {video_path.name}")
    log.info(f"SRT          : {srt_path.name}")
    log.info(f"Language     : {args.language}")
    log.info(f"Qwen mode    : {args.qwen_mode}")
    log.info(f"Demucs       : {'disabled' if args.no_demucs else 'enabled'}")
    log.info(f"Qwen python  : {qwen_python}")
    log.info("=" * 60)

    # ── 1. Parse SRT ─────────────────────────────────────────────────────────
    segments = parse_srt(srt_path)
    if not segments:
        log.error("No segments parsed — make sure this is a diarized+translated SRT")
        return 1

    # ── 2. Audio separation or raw extract ───────────────────────────────────
    background: Optional[Path] = None
    if args.no_demucs:
        # Just extract raw audio for clone refs (if needed)
        audio_for_refs = temp_dir / "input_raw.wav"
        if args.qwen_mode == "clone":
            extract_audio(video_path, audio_for_refs)
    else:
        vocals, background = separate_audio(video_path, temp_dir)
        audio_for_refs = vocals

    # ── 3. Clone refs ─────────────────────────────────────────────────────────
    clone_refs: Dict[str, Path] = {}
    if args.qwen_mode == "clone":
        clone_refs = extract_clone_refs(segments, audio_for_refs, cast_dir)
        if not clone_refs:
            log.warning("⚠️  No clone refs extracted — falling back to custom mode")
            args.qwen_mode = "custom"

    # ── 4. Voice map ─────────────────────────────────────────────────────────
    voice_map = build_voice_map(segments)

    # ── 5. TTS loop ──────────────────────────────────────────────────────────
    qwen_language = _qwen_lang(args.language)
    log.info(f"Qwen language: '{args.language}' → '{qwen_language}'")
    checkpoint_path = work_dir / "checkpoint.json"
    final_files: List[Tuple[Path, float, float]] = _load_checkpoint(checkpoint_path)
    done_indices = {int(Path(c).stem.split("_")[1]) for c, _, _ in final_files}
    clone_broken  = False
    custom_broken = False

    log.info(f"🗣️  Synthesising {len(segments)} segments…")
    for seg in segments:
        i          = seg["index"]
        spk        = seg["speaker"]
        text       = seg["text"]
        start, end = seg["start"], seg["end"]
        target_dur = max(0.1, end - start)

        raw_out = temp_dir / f"seg_{i:04d}.wav"

        if i in done_indices:
            log.info(f"   [{i:04d}] ✓ in checkpoint")
            continue
        if raw_out.exists() and raw_out.stat().st_size > 500:
            log.info(f"   [{i:04d}] ✓ cached")
        else:
            log.info(f"   [{i:04d}] {spk} | {text[:60]!r}")

            ok = False

            if args.qwen_mode == "clone" and not clone_broken:
                ref = clone_refs.get(spk)
                if ref and ref.exists():
                    ok = generate_tts_clone(
                        text, ref, qwen_language, raw_out, qwen_python, qwen_worker
                    )
                    if not ok:
                        log.warning(f"   [{i:04d}] Clone failed — switching to custom")
                        clone_broken = True
                else:
                    log.warning(f"   [{i:04d}] No clone ref for '{spk}' — using custom")

            if not ok and not custom_broken:
                voice = voice_map.get(spk, QWEN_FEMALE_VOICES[0])
                ok = generate_tts_custom(
                    text, voice, qwen_language, raw_out, qwen_python, qwen_worker
                )
                if not ok:
                    log.error(f"   [{i:04d}] Custom TTS also failed — skipping segment")
                    custom_broken = True
                    continue

            if not ok:
                continue

        if not raw_out.exists():
            continue

        fitted = speed_fit(raw_out, target_dur, max_speed=args.max_speed)
        final_files.append((fitted, start, end))
        _save_checkpoint(checkpoint_path, final_files)
        log.info(f"   [{i:04d}] ✓ saved to checkpoint ({len(final_files)} total)")

    if not final_files:
        log.error("No audio was generated. Check Qwen TTS errors above.")
        return 1

    # ── 6. Stitch + mix ──────────────────────────────────────────────────────
    # Trim video to SRT duration if SRT is shorter than the full video
    srt_end = max(s["end"] for s in segments)
    log.info("🎬 Stitching and mixing…")
    final = stitch_and_mix(
        final_files, video_path, output_dir, temp_dir,
        background=background,   # None when --no-demucs
        trim_to=srt_end,
    )

    log.info("=" * 60)
    log.info(f"✅ Done!  →  {final}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())