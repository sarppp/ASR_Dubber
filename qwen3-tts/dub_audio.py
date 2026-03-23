"""
dub_audio.py — Audio extraction, TTS, speed fitting, stitching,
and checkpoint management for the dub pipeline.
"""

import json
import logging
import os
import re
import subprocess
import sys

from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio extraction (raw, no separation)
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path, trim_sec: float = 0) -> None:
    """Extract mono 16 kHz WAV from video (used for clone refs when --no-demucs).
    If trim_sec > 0, only extracts the first trim_sec seconds."""
    if out_wav.exists():
        log.info(f"✓ Reusing {out_wav.name}")
        return
    trim_flag = ["-t", f"{trim_sec:.3f}"] if trim_sec > 0 else []
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", *trim_flag, "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(out_wav), "-y", "-loglevel", "error"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Demucs vocal separation (optional)
# ---------------------------------------------------------------------------

def separate_audio(video_path: Path, temp_dir: Path, trim_sec: float = 0) -> Tuple[Path, Optional[Path]]:
    """
    Run demucs htdemucs to split vocals from background.
    Returns (vocals_path, background_path).
    If trim_sec > 0, only the first trim_sec seconds are extracted before demucs
    (much faster when --trim was used upstream).
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

    trim_flag = ["-t", f"{trim_sec:.3f}"] if trim_sec > 0 else []
    if trim_sec > 0:
        log.info(f"🎶 Separating vocals with demucs (first {trim_sec:.1f}s only)…")
    else:
        log.info("🎶 Separating vocals with demucs…")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", *trim_flag, "-acodec", "pcm_s16le",
         str(raw_wav), "-y", "-loglevel", "error"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals",
         "--device", "cuda", str(raw_wav), "-o", str(demucs_out)],
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
# Qwen TTS — persistent worker (model loaded once, all segments served via IPC)
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


class PersistentTTSWorker:
    """Keeps a qwen_tts_worker.py subprocess alive for the lifetime of the TTS loop.

    The model is loaded once on first use.  All segments are served through the
    same process via JSON-line IPC on stdin/stdout.

    VRAM note: each worker holds one 1.7B bfloat16 model (~3.4 GB).
      - 6 GB GPU  → one worker at a time is safe (clone OR custom, not both)
      - 16 GB+    → clone + custom workers can run concurrently
    dub.py shuts the clone worker before starting the custom worker when
    clone_broken is set, so at most one model is ever resident on tight GPUs.
    """

    MODEL_LOAD_TIMEOUT = 900  # seconds — model download + CUDA init can take >5 min in Docker
    REQUEST_TIMEOUT    = 600  # seconds per synthesis request

    def __init__(self, mode: str, qwen_python: str, qwen_worker_path: str,
                 device_id: Optional[int] = None) -> None:
        self.mode = mode
        self._qwen_python = qwen_python
        self._qwen_worker_path = qwen_worker_path
        self._device_id = device_id
        self._proc: Optional[subprocess.Popen] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _start(self) -> None:
        dev = f"cuda:{self._device_id}" if self._device_id is not None else "cuda"
        log.info(f"Starting persistent TTS worker (mode={self.mode}, device={dev})…")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if self._device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._device_id)
        self._proc = subprocess.Popen(
            [self._qwen_python, "-u", self._qwen_worker_path, "--mode", self.mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,   # inherit → prints directly to terminal (no pipe buffering)
            text=True,
            bufsize=1,
            env=env,
        )
        import select
        import time
        deadline = time.monotonic() + self.MODEL_LOAD_TIMEOUT
        ready = False
        last_line = ""
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"TTS worker (mode={self.mode}) exited before sending READY "
                    f"(rc={self._proc.returncode}) — check logs above for details"
                )
            rlist, _, _ = select.select([self._proc.stdout], [], [], 1.0)
            if rlist:
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line == "READY":
                    ready = True
                    break
                elif line.startswith("LOAD_ERROR:"):
                    self._proc.wait(timeout=10)
                    raise RuntimeError(f"TTS worker model load failed: {line}")
                elif line:
                    last_line = line
                    log.debug(f"Worker stdout: {line}")

        if not ready:
            self._proc.kill()
            raise RuntimeError(
                f"TTS worker did not send READY within {self.MODEL_LOAD_TIMEOUT}s "
                f"(last line got {last_line!r}) — check logs above for details"
            )
        log.info(f"TTS worker ready (mode={self.mode})")

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start()

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write('{"quit": true}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── synthesis ────────────────────────────────────────────────────────────

    def generate_custom(
        self, text: str, voice: str, language: str, output: Path,
    ) -> bool:
        return self._send({
            "text": text, "voice": voice,
            "language": language, "output": str(output),
        })

    def generate_clone(
        self, text: str, ref_audio: Path, language: str, output: Path,
        ref_text: str = "",
    ) -> bool:
        return self._send({
            "text": text, "ref_audio": str(ref_audio), "ref_text": ref_text,
            "language": language, "output": str(output),
        })

    def _send(self, request: dict) -> bool:
        self._ensure_alive()
        import select
        import time
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            deadline = time.monotonic() + self.REQUEST_TIMEOUT
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    log.error("TTS worker died during synthesis")
                    self._proc = None
                    return False
                rlist, _, _ = select.select([self._proc.stdout], [], [], 1.0)
                if rlist:
                    raw_line = self._proc.stdout.readline()
                    if not raw_line:
                        # EOF means worker closed stdout (probably crashed)
                        log.error("TTS worker stdout closed unexpectedly")
                        self._proc = None
                        return False
                        
                    line = raw_line.strip()
                    if not line:
                        continue
                        
                    try:
                        resp = json.loads(line)
                    except json.JSONDecodeError:
                        log.debug(f"Worker output (non-JSON): {line}")
                        continue
                        
                    if not resp.get("ok"):
                        log.error(f"TTS error: {resp.get('error', '?')}")
                    return resp.get("ok", False)
            log.error(f"TTS worker timed out after {self.REQUEST_TIMEOUT}s")
            self._proc.kill()
            self._proc = None
            return False
        except Exception as exc:
            log.error(f"TTS worker IPC error: {exc}")
            self._proc = None
            return False


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
    Fit audio_path into EXACTLY target_dur seconds so the dub timeline
    stays locked to the video.

    Every branch guarantees the output is exactly target_dur via
    ``apad,atrim=0:{target_dur}`` — pad if short, trim if long.
    This eliminates cumulative drift in the stitched dub track.

    - Very short (ratio < 0.85):  pad tail with silence.
    - Slightly short (0.85 ≤ ratio < 1.0):  slow down gently + pad/trim.
    - Long (1.0 < ratio ≤ max_speed):  speed up + pad/trim.
    - Very long (ratio > max_speed):  speed up to max_speed + hard-trim.
    """
    curr = _audio_duration(audio_path)
    if curr <= 0:
        return audio_path

    out   = audio_path.with_name(audio_path.stem + "_fit.wav")
    ratio = curr / target_dur

    if ratio < 0.85:
        # Too short to slow down without sounding weird → pad silence at end
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", f"apad,atrim=0:{target_dur:.6f}",
             "-y", str(out), "-loglevel", "error"],
            check=True,
        )
    elif ratio <= max_speed:
        # 0.85–1.0: gentle slow-down;  1.0–max_speed: speed up
        # apad + atrim guarantee exact target duration (atempo alone drifts by
        # a few ms due to sample-rate rounding in ffmpeg)
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-filter:a", f"atempo={ratio:.4f},apad,atrim=0:{target_dur:.6f}",
             "-vn", "-y", str(out), "-loglevel", "error"],
            check=True,
        )
    else:
        # Severely over — speed up to max_speed + hard-trim to target_dur
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-filter:a", f"atempo={max_speed:.4f},apad,atrim=0:{target_dur:.6f}",
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
) -> Tuple[Path, List[Tuple[float, float, float, float]]]:
    """
    Concatenate dubbed clips with silence gaps → dub track.
    Then mix over video:
      - With demucs:    dub (loud) + background music (quiet) + original video
      - Without demucs: dub track replaces audio entirely
    If trim_to is set, the output video is trimmed to that duration.

    Returns (final_video_path, actual_positions) where actual_positions is
    a list of (actual_start, actual_end, original_start, original_end) for
    each clip — representing where each segment actually lands in the dubbed
    audio timeline.
    """
    concat_list = temp_dir / "concat.txt"
    actual_cur = 0.0    # tracks actual position in the dubbed audio timeline
    actual_positions: List[Tuple[float, float, float, float]] = []

    with open(concat_list, "w") as f:
        for clip_path, start, end in final_files:
            if not clip_path.exists():
                log.warning(f"Missing clip, skipping: {clip_path}")
                continue
            # Absolute positioning: place each clip at its original start time.
            # The gap is computed from actual_cur (real position) so cumulative
            # drift from previous clips is corrected each iteration.
            gap = start - actual_cur
            if gap > 0.001:
                sil = temp_dir / f"sil_{actual_cur:.3f}.wav"
                subprocess.run(
                    f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap:.6f}'
                    f' "{sil}" -y -loglevel error',
                    shell=True, check=True,
                )
                f.write(f"file '{sil.resolve()}'\n")
                actual_cur += gap
            elif gap < -0.001:
                # Previous clip overran into this slot (shouldn't happen with
                # exact speed_fit, but handle gracefully).  The clip will start
                # slightly late — better than overlapping or crashing.
                log.warning(
                    f"Clip at {start:.3f}s: previous segment overran by "
                    f"{-gap:.3f}s — slight timing shift"
                )
            actual_start = actual_cur
            clip_dur = _audio_duration(clip_path)
            actual_cur += clip_dur
            actual_positions.append((actual_start, actual_cur, start, end))
            f.write(f"file '{clip_path.resolve()}'\n")

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

    return final, actual_positions


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
