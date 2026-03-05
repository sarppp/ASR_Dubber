"""
dub_audio.py — Audio extraction, TTS, speed fitting, stitching,
and checkpoint management for the dub pipeline.
"""

import json
import logging
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

    MODEL_LOAD_TIMEOUT = 300  # seconds to wait for "READY" (model download included)
    REQUEST_TIMEOUT    = 120  # seconds per synthesis request

    def __init__(self, mode: str, qwen_python: str, qwen_worker_path: str) -> None:
        self.mode = mode
        self._qwen_python = qwen_python
        self._qwen_worker_path = qwen_worker_path
        self._proc: Optional[subprocess.Popen] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _start(self) -> None:
        log.info(f"Starting persistent TTS worker (mode={self.mode})…")
        self._proc = subprocess.Popen(
            [self._qwen_python, self._qwen_worker_path, "--mode", self.mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,   # inherit → visible in container logs
            text=True,
            bufsize=1,     # line-buffered
        )
        # Block until model is loaded
        import select
        import time
        deadline = time.monotonic() + self.MODEL_LOAD_TIMEOUT
        ready = False
        last_line = ""
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"TTS worker (mode={self.mode}) exited before sending READY "
                    f"(rc={self._proc.returncode})"
                )
            rlist, _, _ = select.select([self._proc.stdout], [], [], 1.0)
            if rlist:
                line = self._proc.stdout.readline()
                if not line:
                    # EOF
                    break
                line = line.strip()
                if line == "READY":
                    ready = True
                    break
                elif line:
                    last_line = line
                    log.debug(f"Worker stdout: {line}")
                    
        if not ready:
            self._proc.kill()
            raise RuntimeError(
                f"TTS worker did not send READY within {self.MODEL_LOAD_TIMEOUT}s "
                f"(last line got {last_line!r})"
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
