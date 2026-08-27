"""
dub_audio.py — Audio extraction, TTS, speed fitting, stitching,
and checkpoint management for the dub pipeline.
"""

import json
import logging
import os
import queue as _queue
import re
import subprocess
import sys
import threading

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
    min_ref_dur: float = 3.0,  # target minimum duration for clone ref
) -> Dict[str, Path]:
    """
    For each speaker, extract audio from audio_source as a reference WAV
    for Qwen clone mode. Uses the longest segment, or concatenates multiple
    segments if the longest is too short.
    Returns {speaker: wav_path}
    """
    cast_dir.mkdir(parents=True, exist_ok=True)

    # Collect all segments per speaker, sorted by duration (longest first)
    speaker_segs: Dict[str, List[Tuple[float, float, float]]] = {}
    for seg in segments:
        spk = seg["speaker"]
        dur = max(0.0, seg["end"] - seg["start"])
        if dur > 0:
            if spk not in speaker_segs:
                speaker_segs[spk] = []
            speaker_segs[spk].append((dur, seg["start"], seg["end"]))

    # Sort each speaker's segments by duration descending
    for spk in speaker_segs:
        speaker_segs[spk].sort(reverse=True)

    refs: Dict[str, Path] = {}
    log.info("🎙️  Extracting clone reference WAVs…")
    for spk, segs in speaker_segs.items():
        safe_name = re.sub(r"[^\w\-]", "_", spk)
        out_wav   = cast_dir / f"{safe_name}.wav"

        if out_wav.exists() and out_wav.stat().st_size > 1000:
            log.info(f"   ✓ {spk}: reusing {out_wav.name}")
            refs[spk] = out_wav
            continue

        best_dur, best_start, best_end = segs[0]

        # If longest segment is long enough, use it directly
        if best_dur >= min_ref_dur:
            log.info(f"   → {spk}: {best_dur:.2f}s @ {best_start:.2f}–{best_end:.2f}s")
            subprocess.run(
                ["ffmpeg", "-ss", str(best_start), "-t", str(best_dur),
                 "-i", str(audio_source),
                 "-ac", "1", "-ar", "16000", "-y", str(out_wav), "-loglevel", "error"],
                check=True,
            )
        else:
            # Concatenate multiple segments to reach minimum duration
            total_dur = 0.0
            selected = []
            for dur, start, end in segs:
                selected.append((dur, start, end))
                total_dur += dur
                if total_dur >= min_ref_dur:
                    break

            if total_dur < 1.0:
                log.warning(f"   ⚠️  {spk}: total segments only {total_dur:.2f}s — too short for clone")
                continue

            # Create concat file for ffmpeg
            concat_file = cast_dir / f"{safe_name}_concat.txt"
            with open(concat_file, "w") as f:
                for dur, start, end in selected:
                    f.write(f"file '{audio_source}'\n")
                    f.write(f"inpoint {start}\n")
                    f.write(f"outpoint {end}\n")

            log.info(f"   → {spk}: concatenating {len(selected)} segments ({total_dur:.2f}s total)")
            subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-ac", "1", "-ar", "16000", "-y", str(out_wav), "-loglevel", "error"],
                check=True,
            )
            concat_file.unlink(missing_ok=True)

        if out_wav.exists() and out_wav.stat().st_size > 1000:
            refs[spk] = out_wav

    return refs


def detect_speaker_genders(clone_refs: Dict[str, Path]) -> Dict[str, str]:
    """
    Estimate gender for each speaker from their clone-ref WAV using pitch (F0).
    Returns {speaker: "female"/"male"}.  Speakers with no ref are omitted.
    Requires librosa; silently returns {} if not available.
    """
    try:
        import numpy as np
        import librosa
    except Exception:
        return {}

    out: Dict[str, str] = {}
    for spk, wav in clone_refs.items():
        try:
            y, sr = librosa.load(str(wav), sr=16000, mono=True)
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0 = f0[np.isfinite(f0)]
            if f0.size == 0:
                continue
            med = float(np.median(f0))
            out[spk] = "female" if med >= 165 else "male"
            log.info(f"   gender [{spk}]: pitch≈{med:.0f}Hz → {out[spk]}")
        except Exception:
            continue
    return out


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


# Each worker speaks the same stdin/stdout JSON protocol, so the TTS engine is
# swappable by pointing SharedTTSManager at a different worker script + python.
# Layout: <root>/qwen3-tts/{dub.py,qwen_tts_worker.py}
#         <root>/cosyvoice-tts/cosyvoice_tts_worker.py  (+ its own .venv)
TTS_ENGINES = {
    "qwen":      {"project": "qwen3-tts",      "worker": "qwen_tts_worker.py"},
    "cosyvoice": {"project": "cosyvoice-tts",  "worker": "cosyvoice_tts_worker.py"},
}


def resolve_tts_engine(
    engine: str,
    script_dir: Path,
    python_override: Optional[str] = None,
    worker_override: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (python_executable, worker_script_path) for a TTS engine.

    ``script_dir`` is the folder holding dub.py (``<root>/qwen3-tts``).
    Sibling engine projects live at ``<root>/<project>`` with their own
    ``.venv``.  Explicit overrides win (used by the Modal images).
    """
    spec = TTS_ENGINES.get(engine)
    if spec is None:
        raise ValueError(
            f"Unknown --tts-engine {engine!r}. Choose one of: {sorted(TTS_ENGINES)}"
        )

    root = script_dir.parent
    project_dir = script_dir if spec["project"] == script_dir.name else root / spec["project"]

    worker = Path(worker_override) if worker_override else project_dir / spec["worker"]
    if not worker.exists():
        raise FileNotFoundError(
            f"{spec['worker']} not found at {worker}. "
            + ("Run: git submodule update --init --recursive && "
               "uv sync --project cosyvoice-tts" if engine == "cosyvoice" else "")
        )

    if python_override:
        python = python_override
    else:
        venv_py = project_dir / ".venv" / "bin" / "python"
        python = str(venv_py) if venv_py.exists() else "python"
        if python == "python":
            log.warning(f"{spec['project']} .venv not found at {venv_py}; using 'python'")

    return python, str(worker)


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

    # Serialize model loading across all workers: prevents GPU memory bandwidth
    # contention when N workers start simultaneously.
    _startup_lock: threading.Lock = threading.Lock()

    MODEL_LOAD_TIMEOUT = 500  # seconds — model download + CUDA init (remote GPU has fast internet)
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
        # Hold the class-level lock so that when N workers start simultaneously
        # each model finishes loading before the next begins (avoids VRAM spike).
        with PersistentTTSWorker._startup_lock:
            self._start_inner()

    def _start_inner(self) -> None:
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
        import select
        import time
        try:
            self._ensure_alive()
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
# Shared TTS Manager — single model instance, multiple request threads
# ---------------------------------------------------------------------------

class SharedTTSManager:
    """Shares ONE model instance across N worker threads via request queue.
    
    Problem: N workers each load the model → N× VRAM usage → OOM.
    Solution: 1 model + request queue → 1× VRAM + parallel CPU/GPU overlap.
    
    Architecture:
      - 1 synthesis thread: pops requests from queue, runs GPU inference
      - N worker threads: submit requests, wait for result, then do CPU speed_fit
    
    This overlaps CPU (speed_fit) with GPU (TTS) while keeping VRAM constant.
    """
    
    def __init__(self, mode: str, qwen_python: str, qwen_worker_path: str,
                 device_id: Optional[int] = None):
        self.mode = mode
        self._qwen_python = qwen_python
        self._qwen_worker_path = qwen_worker_path
        self._device_id = device_id
        
        self._request_queue: _queue.Queue = _queue.Queue()
        self._result_events: Dict[int, threading.Event] = {}
        self._results: Dict[int, bool] = {}
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        
        self._worker: Optional[PersistentTTSWorker] = None
        self._synth_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start the single model instance and synthesis thread."""
        # Load model once
        self._worker = PersistentTTSWorker(
            self.mode, self._qwen_python, self._qwen_worker_path, self._device_id
        )
        self._worker._ensure_alive()
        
        # Start synthesis thread that processes queue
        self._synth_thread = threading.Thread(
            target=self._synthesis_loop, daemon=True, name="tts-synth"
        )
        self._synth_thread.start()
        log.info(f"SharedTTSManager started (mode={self.mode}, device={self._device_id})")
    
    def _synthesis_loop(self) -> None:
        """Process TTS requests sequentially from queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for request with timeout to check stop_event periodically
                try:
                    req_id, request = self._request_queue.get(timeout=0.5)
                except _queue.Empty:
                    continue
                
                # Run synthesis
                ok = self._worker._send(request) if self._worker else False
                
                # Store result and signal waiting thread
                self._results[req_id] = ok
                if req_id in self._result_events:
                    self._result_events[req_id].set()
                    
            except Exception as e:
                log.error(f"Synthesis loop error: {e}")
    
    def submit(self, request: dict, timeout: float = 600) -> bool:
        """Submit a TTS request and wait for result.
        
        Returns True on success, False on failure.
        """
        # Get unique request ID
        with self._counter_lock:
            req_id = self._request_counter
            self._request_counter += 1
        
        # Create event for this request
        event = threading.Event()
        self._result_events[req_id] = event
        self._results[req_id] = False
        
        # Submit to queue
        self._request_queue.put((req_id, request))
        
        # Wait for result
        if event.wait(timeout=timeout):
            result = self._results.get(req_id, False)
        else:
            log.error(f"TTS request {req_id} timed out after {timeout}s")
            result = False
        
        # Cleanup
        self._result_events.pop(req_id, None)
        self._results.pop(req_id, None)
        
        return result
    
    def generate_custom(self, text: str, voice: str, language: str, output: Path) -> bool:
        return self.submit({
            "text": text, "voice": voice,
            "language": language, "output": str(output),
        })
    
    def generate_clone(self, text: str, ref_audio: Path, language: str, 
                       output: Path, ref_text: str = "") -> bool:
        return self.submit({
            "text": text, "ref_audio": str(ref_audio), "ref_text": ref_text,
            "language": language, "output": str(output),
        })
    
    def close(self) -> None:
        """Stop synthesis thread and free model."""
        self._stop_event.set()
        if self._synth_thread and self._synth_thread.is_alive():
            self._synth_thread.join(timeout=5)
        if self._worker:
            self._worker.close()
            self._worker = None
        log.info("SharedTTSManager closed")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Rhythm plan — derive real speech/pause structure from the ORIGINAL audio
# ---------------------------------------------------------------------------

def _parse_srt_min(path: Path) -> List[Dict]:
    """Minimal SRT parse → [{start,end,text}] with the [Speaker N] tag stripped."""
    out: List[Dict] = []
    txt = Path(path).read_text(encoding="utf-8")
    for block in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        rest = lines[1:] if re.match(r"^\d+$", lines[0]) else lines
        m = re.match(r"([\d:,.]+)\s*-->\s*([\d:,.]+)", rest[0])
        if not m:
            continue

        def _t(s: str) -> float:
            s = s.strip().replace(",", ".")
            h, mm, ss = s.split(":")
            return int(h) * 3600 + int(mm) * 60 + float(ss)

        body = re.sub(r"\[[^\]]+\]", "", " ".join(rest[1:])).strip()
        if body:
            out.append(dict(index=len(out) + 1, start=_t(m.group(1)),
                            end=_t(m.group(2)), text=body))
    return out


def build_dub_plan(
    fr_segments: List[Dict],
    orig_media: Path,
    orig_srt: Path,
    device: str = "auto",
) -> Dict[int, Dict]:
    """
    Forced-align the ORIGINAL audio to its transcript, then for each (merged)
    translated segment report where that content actually sits in the original:

        orig_start / orig_end : first/last original word for this content
        orig_dur              : voiced + micro-pause seconds the speaker spent
        pause_before          : real pause between this segment and the previous

    This is the rhythm reference the placement/fit stages use instead of the
    loose NeMo subtitle grid.  Returns {} (→ caller keeps SRT timing) if the
    original SRT is missing or alignment is unavailable.
    """
    try:
        from forced_align import align_segments
    except Exception as exc:
        log.warning(f"forced alignment unavailable ({exc}); using SRT timing")
        return {}

    orig_srt = Path(orig_srt)
    if not orig_srt.exists():
        log.warning(f"original SRT not found at {orig_srt}; using SRT timing")
        return {}
    en_segs = _parse_srt_min(orig_srt)
    if not en_segs:
        return {}

    log.info(f"🎯 Forced-aligning original audio → transcript ({orig_srt.name})…")
    fa = align_segments(Path(orig_media), en_segs, device=device)
    if not fa.ok:
        log.warning(f"forced alignment failed ({fa.note}); using SRT timing")
        return {}
    log.info(f"   {fa.note}")

    words = sorted((w.start, w.end) for w in fa.words if w.end > w.start)
    if not words:
        return {}

    # Assign every original word to exactly one translated segment (nearest by
    # midpoint) so segment spans never overlap and the pauses between them fall
    # out cleanly.
    bounds = [(seg["start"], seg["end"], si) for si, seg in enumerate(fr_segments)]
    buckets: List[List[Tuple[float, float]]] = [[] for _ in fr_segments]
    for a, b in words:
        mid = (a + b) / 2
        inside = [si for s, e, si in bounds if s <= mid <= e]
        if inside:
            si = inside[0]
        else:
            si = min(range(len(bounds)),
                     key=lambda k: min(abs(mid - bounds[k][0]), abs(mid - bounds[k][1])))
        buckets[si].append((a, b))

    plan: Dict[int, Dict] = {}
    prev_end = 0.0
    for si, seg in enumerate(fr_segments):
        win = buckets[si]
        if not win:
            s, e = seg["start"], seg["end"]
            plan[seg["index"]] = dict(orig_start=s, orig_end=e, orig_dur=e - s,
                                      pause_before=round(max(0.0, s - prev_end), 3),
                                      aligned=False)
            prev_end = e
            continue
        dur, pe = 0.0, win[0][0]
        for a, b in win:
            gap = a - pe
            if 0 < gap < 0.6:          # keep the speaker's own micro-pauses
                dur += gap
            dur += max(0.0, b - a)
            pe = b
        plan[seg["index"]] = dict(orig_start=round(win[0][0], 3),
                                  orig_end=round(win[-1][1], 3),
                                  orig_dur=round(dur, 3),
                                  pause_before=round(max(0.0, win[0][0] - prev_end), 3),
                                  aligned=True)
        prev_end = win[-1][1]

    n_al = sum(1 for p in plan.values() if p["aligned"])
    log.info(f"   rhythm plan: {n_al}/{len(plan)} segments aligned to the original")
    return plan


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


def strip_leading_silence(audio_path: Path) -> Path:
    """
    Remove leading silence from a TTS clip.

    Qwen TTS (and most neural TTS) prepends ~0.1–0.5s of silence before the
    first word.  Without stripping, proportional splits assign that silence to
    the first sub-segment, making the clip start ~0.3s late.

    Uses ffmpeg silenceremove: only the leading silence is removed;
    trailing silence and inter-word pauses are kept intact.
    Returns a new path (original unchanged); falls back to original on error.
    """
    out = audio_path.with_name(audio_path.stem + "_ns.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-40dB",
             "-y", str(out), "-loglevel", "error"],
            check=True,
        )
        if out.exists() and out.stat().st_size > 500:
            stripped_dur = _audio_duration(out)
            orig_dur     = _audio_duration(audio_path)
            # Sanity: stripping shouldn't remove more than 1 s or 30 % of audio
            if stripped_dur > 0 and (orig_dur - stripped_dur) < min(1.0, orig_dur * 0.3):
                return out
    except Exception:
        pass
    return audio_path


def _last_word_boundary(audio_path: Path, before_sec: float) -> float:
    """
    Return the timestamp of the last inter-word silence that *starts* before
    ``before_sec``.  If none is found, returns ``before_sec`` (fallback to
    hard trim).

    Uses ffmpeg silencedetect at -35 dB / 20 ms — short enough to catch the
    brief pauses between TTS words without catching intra-phoneme dips.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", "silencedetect=noise=-35dB:duration=0.02",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        best = None
        for line in r.stderr.splitlines():
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                t = float(m.group(1))
                if t < before_sec:
                    best = t
        if best is not None:
            return best
    except Exception:
        pass
    return before_sec


def _trim_edge_silence(audio_path: Path, out: Path) -> bool:
    """Write ``out`` = ``audio_path`` with leading AND trailing silence removed.

    Neural TTS pads ~0.1–0.6s of silence before the first word and after the
    last; left in, the lead pushes the clip late and the tail becomes dead air
    the subtitle track spreads text across.  ``silenceremove`` strips the head,
    then ``areverse | silenceremove | areverse`` strips the tail; inter-word
    pauses are untouched.  Returns True on success.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-af",
             "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,"
             "areverse,"
             "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,"
             "areverse",
             "-vn", "-y", str(out), "-loglevel", "error"],
            check=True,
        )
        return out.exists() and out.stat().st_size > 500
    except Exception:
        return False


def speed_fit(audio_path: Path, target_dur: float,
              max_speed: float = 1.35, min_speed: float = 1.0) -> Path:
    """
    Fit audio_path so it lands within target_dur seconds without drawling.

    The dub timeline is kept locked to the video by *absolute* placement in
    ``stitch_and_mix`` (each clip starts at its SRT time; the gap to the next
    clip is recomputed from the real cursor every iteration), so speed_fit no
    longer has to emit exactly target_dur.  It therefore never slows speech or
    pads silence to fill an over-long slot — that was the source of the
    "sometimes slow / draggy, subtitle lingers over dead air" problem.

    - clip ≤ target (fits):           trim edge silence, keep natural pace.
                                      The unused time becomes a real pause.
    - target < clip ≤ target·max_speed: gentle speed-up to fit exactly.
    - clip > target·max_speed:        speed up to max_speed, then trim at the
                                      last word boundary before target_dur.

    min_speed (default 1.0 = never slow): set below 1.0 to re-enable filling
    short clips by slowing them down to that ratio (old behaviour: 0.65).
    """
    if _audio_duration(audio_path) <= 0:
        return audio_path

    out = audio_path.with_name(audio_path.stem + "_fit.wav")

    # Legacy opt-in: slow a very short clip down to fill the slot + pad the rest.
    if min_speed < 1.0:
        curr = _audio_duration(audio_path)
        ratio = curr / target_dur
        if ratio < min_speed:
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path),
                 "-filter:a", f"atempo={min_speed:.4f},apad,atrim=0:{target_dur:.6f}",
                 "-vn", "-y", str(out), "-loglevel", "error"],
                check=True,
            )
            return out if out.exists() else audio_path

    # Trim the TTS lead/tail silence FIRST, then judge fit on the real speech.
    trimmed = audio_path.with_name(audio_path.stem + "_trim.wav")
    speech = audio_path if not _trim_edge_silence(audio_path, trimmed) else trimmed
    curr = _audio_duration(speech)
    ratio = curr / target_dur if target_dur > 0 else 1.0

    if ratio <= 1.0:
        # Real speech fits — keep natural pace; leftover time becomes a pause.
        result = speech
    elif ratio <= max_speed:
        # Mild speed-up to fit exactly (apad+atrim guard atempo's ms rounding).
        subprocess.run(
            ["ffmpeg", "-i", str(speech),
             "-filter:a", f"atempo={ratio:.4f},apad,atrim=0:{target_dur:.6f}",
             "-vn", "-y", str(out), "-loglevel", "error"],
            check=True,
        )
        result = out
    else:
        # Severely over — speed up to max_speed, then trim at a word boundary
        # to avoid cutting mid-syllable, then pad to exactly target_dur.
        compressed = audio_path.with_name(audio_path.stem + "_cmp.wav")
        subprocess.run(
            ["ffmpeg", "-i", str(speech),
             "-filter:a", f"atempo={max_speed:.4f}",
             "-vn", "-y", str(compressed), "-loglevel", "error"],
            check=True,
        )
        cut_at = _last_word_boundary(compressed, target_dur)
        subprocess.run(
            ["ffmpeg", "-i", str(compressed),
             "-filter:a", f"atrim=0:{cut_at:.6f},apad,atrim=0:{target_dur:.6f}",
             "-vn", "-y", str(out), "-loglevel", "error"],
            check=True,
        )
        compressed.unlink(missing_ok=True)
        result = out

    if result is trimmed and trimmed.exists():
        trimmed.replace(out)
        return out
    if result is audio_path:
        return audio_path
    if trimmed.exists() and trimmed != out:
        trimmed.unlink(missing_ok=True)
    return out if out.exists() else audio_path


def _speech_chunks(audio_path: Path, noise_db: float = -35.0,
                   min_sil: float = 0.16) -> List[Tuple[float, float]]:
    """Return [(start,end)] of the voiced runs in a clip (splits on inter-phrase
    silences)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-af",
             f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
            capture_output=True, text=True,
        )
    except Exception:
        return []
    total = _audio_duration(audio_path)
    sil: List[Tuple[float, float]] = []
    cur = None
    for line in r.stderr.splitlines():
        a = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        b = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if a:
            cur = float(a.group(1))
        elif b and cur is not None:
            sil.append((max(0.0, cur), float(b.group(1))))
            cur = None
    chunks, pos = [], 0.0
    for s0, s1 in sil:
        if s0 > pos + 0.05:
            chunks.append((pos, s0))
        pos = max(pos, s1)
    if total - pos > 0.05:
        chunks.append((pos, total))
    return chunks or [(0.0, total)]


def redistribute_to_duration(audio_path: Path, target_dur: float,
                             max_pause: float = 2.2) -> Path:
    """
    Stretch a clip to ~target_dur by inserting silence at its inter-phrase
    boundaries — NOT by slowing the speech.  Mimics how a speaker fills a longer
    slot: same words, longer pauses between phrases.

    Falls back to the original clip if it has no internal boundaries to space
    (single phrase) or already fills the target.
    """
    cur = _audio_duration(audio_path)
    if cur <= 0 or target_dur <= cur + 0.3:
        return audio_path
    chunks = _speech_chunks(audio_path)
    if len(chunks) < 2:
        return audio_path

    speech = sum(e - s for s, e in chunks)
    n_gaps = len(chunks) - 1
    budget = min(target_dur, speech + n_gaps * max_pause) - speech
    if budget <= 0.3:
        return audio_path
    per_gap = budget / n_gaps

    out = audio_path.with_name(audio_path.stem + "_rd.wav")
    parts_dir = audio_path.parent
    concat = parts_dir / f"{audio_path.stem}_rd.txt"
    sr_probe = 24000
    try:
        with open(concat, "w") as f:
            for k, (s, e) in enumerate(chunks):
                seg = parts_dir / f"{audio_path.stem}_rd{k:02d}.wav"
                subprocess.run(
                    ["ffmpeg", "-ss", f"{s:.4f}", "-to", f"{e:.4f}", "-i", str(audio_path),
                     "-c", "copy", "-y", str(seg), "-loglevel", "error"],
                    check=True,
                )
                f.write(f"file '{seg.resolve()}'\n")
                if k < n_gaps:
                    sil = parts_dir / f"{audio_path.stem}_rdsil{k:02d}.wav"
                    subprocess.run(
                        f'ffmpeg -f lavfi -i anullsrc=r={sr_probe}:cl=mono '
                        f'-t {per_gap:.4f} "{sil}" -y -loglevel error',
                        shell=True, check=True,
                    )
                    f.write(f"file '{sil.resolve()}'\n")
        subprocess.run(
            f'ffmpeg -f concat -safe 0 -i "{concat}" -c copy "{out}" -y -loglevel error',
            shell=True, check=True,
        )
    except Exception as exc:
        log.warning(f"redistribute_to_duration failed ({exc}); keeping natural clip")
        return audio_path
    finally:
        for p in parts_dir.glob(f"{audio_path.stem}_rd*"):
            if p != out:
                p.unlink(missing_ok=True)
    return out if out.exists() and out.stat().st_size > 500 else audio_path


# ---------------------------------------------------------------------------
# Proportional TTS split across original sub-segment timings
# ---------------------------------------------------------------------------

def split_tts_proportional(
    raw_path: Path,
    subsegments: List[Dict],   # list of {"start": float, "end": float}
    temp_dir: Path,
    base_stem: str,
    strip_silence: bool = True,
) -> List[Tuple[Path, float, float]]:
    """
    Split a merged TTS clip proportionally back across original sub-segment timings.

    The TTS audio is sliced in proportion to each sub's duration relative to the
    total sub duration.  Each slice is returned as (wav_path, sub_start, sub_end)
    ready to be passed to speed_fit individually.

    This ensures speech is placed at the right position in the video instead of
    a single block at the start followed by a long silence.
    """
    if len(subsegments) <= 1:
        # Nothing to split — caller should use raw_path directly
        s = subsegments[0]
        return [(raw_path, s["start"], s["end"])]

    raw_dur = _audio_duration(raw_path)
    if raw_dur <= 0:
        return []

    total_sub_dur = sum(s["end"] - s["start"] for s in subsegments)
    if total_sub_dur <= 0:
        return []

    results: List[Tuple[Path, float, float]] = []
    offset = 0.0
    for i, sub in enumerate(subsegments):
        sub_dur = sub["end"] - sub["start"]
        proportion = sub_dur / total_sub_dur
        tts_slice = raw_dur * proportion

        out = temp_dir / f"{base_stem}_sub{i:02d}.wav"
        subprocess.run(
            ["ffmpeg",
             "-ss", f"{offset:.6f}",
             "-t",  f"{tts_slice:.6f}",
             "-i",  str(raw_path),
             "-y",  str(out),
             "-loglevel", "error"],
            check=True,
        )
        results.append((out, sub["start"], sub["end"]))
        offset += tts_slice

    return results


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
    placements: Optional[Dict[int, float]] = None,  # seg index → forced start (rhythm)
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
    placements = placements or {}

    def _seg_idx(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    # Rhythm placement may reorder clips relative to their SRT start.
    ordered = sorted(
        final_files,
        key=lambda t: placements.get(_seg_idx(t[0]), t[1]),
    )

    with open(concat_list, "w") as f:
        for clip_path, start, end in ordered:
            if not clip_path.exists():
                log.warning(f"Missing clip, skipping: {clip_path}")
                continue
            # Place at the rhythm position if we have one, else the SRT start.
            # The gap is computed from actual_cur (real position) so cumulative
            # drift from previous clips is corrected each iteration.
            place_at = placements.get(_seg_idx(clip_path), start)
            gap = place_at - actual_cur
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
# Build trim flag if needed
    if trim_to:
        log.info(f"✂️  Trimming output video to {trim_to:.2f}s (matches SRT duration)")
        trim_flags  = ["-t", str(trim_to)]
        
        # --- NEW GPU DETECTION LOGIC ---
        try:
            # Quick check if NVENC is available in this environment
            check_gpu = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
            if "nvenc" in check_gpu.stdout.lower():
                log.info("🚀 RENDER MODE: GPU Accelerated (h264_nvenc)")
                video_codec = [
                    "-c:v", "h264_nvenc", 
                    "-preset", "p4", 
                    "-tune", "hq", 
                    "-rc", "vbr", 
                    "-cq", "19", 
                    "-b:v", "0"
                ]
            else:
                log.warning("⚠️  GPU Encoder not found! Falling back to CPU (libx264)")
                video_codec = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
        except Exception:
            video_codec = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
        # -------------------------------
    else:
        log.info("🚀 RENDER MODE: Stream Copy (No re-encode)")
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
             *video_codec, str(final), "-y", "-loglevel", "error", "-stats"],
            check=True,
        )
    else:
        subprocess.run(
            ["ffmpeg",
             "-i", str(video_path),
             "-i", str(dub_track),
             *trim_flags,
             "-map", "0:v", "-map", "1:a",
             *video_codec, str(final), "-y", "-loglevel", "error", "-stats"],
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
