#!/usr/bin/env python3
"""
run_pipeline.py — Master runner for the ASR dubbing pipeline
=============================================================
Place in ASR/ root. Runs all three steps end-to-end.

Usage:
# Basic — auto-detects source lang, processes next unprocessed video
uv run python run_pipeline.py --target-lang fr

# With trim (first 30s only)
uv run python run_pipeline.py --target-lang fr --trim 30

# Custom input/output folders
uv run python run_pipeline.py --target-lang fr --input-dir /path/to/videos --output-dir /path/to/results

# Force source language (skip Whisper detection)
uv run python run_pipeline.py --target-lang fr --language de

# Partial runs
uv run python run_pipeline.py --target-lang fr --run-mode transcribe   # Step 1 only
uv run python run_pipeline.py --target-lang fr --run-mode translate    # Steps 1-2 only
uv run python run_pipeline.py --target-lang fr --run-mode full         # All steps (default)

# Skip individual steps (when files already exist)
uv run python run_pipeline.py --target-lang fr --skip-nemo
uv run python run_pipeline.py --target-lang fr --skip-translate
uv run python run_pipeline.py --target-lang fr --skip-dub

# Voice mode (default is clone)
uv run python run_pipeline.py --target-lang fr --qwen-mode clone       # clones original speakers
uv run python run_pipeline.py --target-lang fr --qwen-mode custom      # uses fixed Qwen voices

# Faster — skip background music preservation
uv run python run_pipeline.py --target-lang fr --no-demucs

# Whisper model for language detection
uv run python run_pipeline.py --target-lang fr --whisper-model large-v3

### NEMO
# Precision
uv run python run_pipeline.py --target-lang fr --precision fp16   # older GPUs
uv run python run_pipeline.py --target-lang fr --precision fp32   # max accuracy
uv run python run_pipeline.py --target-lang fr --precision bf16   # default

# Force chunk size — auto-detected from VRAM (capped at 600s max)
# Lower if getting OOM, e.g. 120 = 2 min chunks
uv run python run_pipeline.py --target-lang fr --chunk-override 120

# VRAM tuning
uv run python run_pipeline.py --target-lang fr --reserve-gb 3.0
uv run python run_pipeline.py --target-lang fr --safety-factor 0.7

# Override model
uv run python run_pipeline.py --target-lang fr --nemo-model nvidia/parakeet-tdt-1.1b

# Combine freely
uv run python run_pipeline.py --target-lang fr --trim 30 --precision fp16 --reserve-gb 2.0

ASR/
├── run_pipeline.py          ← goes here
├── whisper/
│   ├── detect_language.py   ← goes here (new)
│   └── whisper_local.py     ← already here
├── nemo/
├── gemma-translate/
└── qwen3-tts/


"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Folder layout ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
NEMO_DIR      = ROOT / "nemo"
TRANSLATE_DIR = ROOT / "translate-gemma"
QWEN_DIR      = ROOT / "qwen3-tts"

WHISPER_DIR   = ROOT / "whisper"
NEMO_PY       = NEMO_DIR      / ".venv" / "bin" / "python"
QWEN_PY       = QWEN_DIR      / ".venv" / "bin" / "python"
WHISPER_PY    = WHISPER_DIR   / ".venv" / "bin" / "python"
TRANSLATE_PY  = TRANSLATE_DIR / ".venv" / "bin" / "python"
CLEAN_SUBS_SCRIPT = TRANSLATE_DIR / "clean_subs.py"
END_PRODUCT_DIR   = NEMO_DIR / "end_product"

OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_BIN         = os.getenv("OLLAMA_BIN", "ollama")   # native binary path
OLLAMA_DOCKER_IMAGE = os.getenv("OLLAMA_DOCKER_IMAGE", "ollama/ollama")
OLLAMA_MODELS_DIR  = os.getenv("OLLAMA_MODELS_DIR", str(Path.home() / "python-tools" / ".ollama_models"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n▶  {text}\n{bar}\n", flush=True)


def _run(cmd: list, cwd: Path, label: str) -> None:
    _banner(label)
    print(f"   cwd : {cwd}")
    print(f"   cmd : {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n❌  {label} failed (exit {result.returncode})", flush=True)
        sys.exit(result.returncode)
    print(f"\n✅  {label} done", flush=True)


def _python(venv_py: Path, fallback_dir: Path) -> list:
    """Return [python_executable] — venv directly, or uv run as fallback."""
    if venv_py.exists():
        return [str(venv_py)]
    print(f"⚠️  venv not found at {venv_py}, falling back to: uv run python")
    return ["uv", "run", "python"]


# ── Source language detection ─────────────────────────────────────────────────

def _normalize_base(s: str) -> str:
    """Lowercase + collapse any run of non-alphanumeric chars to '_' for fuzzy matching.
    'Debate 101 with Harvard\'s former...' == 'Debate_101_with_Harvard_s_former...'
    """
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _video_already_processed(video: Path, target_lang: str | None = None) -> bool:
    """Return True only if this exact video+target_lang pair has a finished run dir."""
    if not END_PRODUCT_DIR.exists():
        return False
    base = _normalize_base(re.split(r"[._]nemo|__", video.stem)[0])
    for run_dir in END_PRODUCT_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        dir_base = _normalize_base(re.split(r"[._]nemo|__", run_dir.name)[0])
        if base != dir_base:
            continue
        # If we know the target lang, only count it processed if this lang pair exists
        if target_lang and f"_to_{target_lang}" not in run_dir.name:
            continue
        return True
    return False


def _find_video(target_lang: str | None = None) -> Path | None:
    """Find the newest unprocessed video file in nemo/ dir."""
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    videos = sorted(
        (f for f in NEMO_DIR.iterdir() if f.suffix.lower() in VIDEO_EXT),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for video in videos:
        if not _video_already_processed(video, target_lang=target_lang):
            return video
    return videos[0] if videos else None


def _find_srt_for_video(video_base: str, pattern: str) -> Path | None:
    """
    Find an SRT matching `pattern` (a glob) for this video_base.
    Checks nemo/ first, then end_product/<run_dir>/ as fallback
    (clean_subs.py moves files there after a completed run).
    Uses normalized comparison so spaces vs underscores don't matter.
    """
    norm_base = _normalize_base(video_base)

    # 1. Live location — nemo/
    for srt in sorted(NEMO_DIR.glob(pattern)):
        srt_base = _normalize_base(re.split(r"[._]nemo|__", srt.stem)[0])
        if srt_base == norm_base:
            return srt

    # 2. Archived location — end_product/<any run_dir>/
    if END_PRODUCT_DIR.exists():
        for run_dir in sorted(END_PRODUCT_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            dir_base = _normalize_base(re.split(r"[._]nemo|__", run_dir.name)[0])
            if dir_base != norm_base:
                continue
            for srt in sorted(run_dir.glob(pattern)):
                srt_base = _normalize_base(re.split(r"[._]nemo|__", srt.stem)[0])
                if srt_base == norm_base:
                    return srt
    return None


def _detect_source_language(video_path: Path, whisper_model: str = "medium") -> str | None:
    """
    Use Whisper (30s forward pass) to detect spoken language.
    Runs detect_language.py in the whisper uv env.
    Prints only the 2-letter code to stdout, which we capture here.
    """
    detect_script = WHISPER_DIR / "detect_language.py"
    if not detect_script.exists():
        print(f"⚠️  detect_language.py not found at {detect_script}")
        return None

    whisper_py = str(WHISPER_PY) if WHISPER_PY.exists() else "python"

    print(f"🔍 Detecting source language from '{video_path.name}' (Whisper, 30s sample)...",
          flush=True)
    try:
        result = subprocess.run(
            [whisper_py, str(detect_script), str(video_path),
             "--model", whisper_model],
            cwd=str(WHISPER_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"⚠️  Language detection failed: {result.stderr[-200:]}")
            return None
        lang = result.stdout.strip().lower()
        if lang and len(lang) <= 3:
            print(f"✅ Detected language: '{lang}'")
            return lang
        print(f"⚠️  Unexpected output from detect_language.py: {repr(lang)}")
        return None
    except subprocess.TimeoutExpired:
        print("⚠️  Language detection timed out after 120s")
        return None
    except Exception as e:
        print(f"⚠️  Language detection error: {e}")
        return None


# ── Ollama lifecycle ──────────────────────────────────────────────────────────

def _ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _ollama_start() -> subprocess.Popen | None:
    """
    Start Ollama — three strategies tried in order:
      1. Already running → do nothing
      2. Docker available → start ollama/ollama container (local machine)
      3. Native binary → ollama serve (RunPod / server)
    Returns Popen handle of what we started, or None if already running.
    """
    if _ollama_is_running():
        print("✓ Ollama already running", flush=True)
        return None

    # ── Strategy 1: Docker ────────────────────────────────────────────────────
    if _docker_available():
        print("🐳 Starting Ollama via Docker...", flush=True)
        # Check if container already exists but is stopped
        subprocess.run(["docker", "rm", "-f", "ollama"],
                       capture_output=True)
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", "ollama",
            "-e", "OLLAMA_HOST=0.0.0.0",
            "-e", "OLLAMA_FLASH_ATTENTION=1",
            "-p", "11434:11434",
        ]
        # Add GPU flags if nvidia-smi is available
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            cmd += ["--runtime=nvidia", "--gpus", "all"]
            print("   GPU detected — enabling NVIDIA runtime", flush=True)
        except Exception:
            print("   No GPU detected — running Ollama on CPU", flush=True)
        cmd.append(OLLAMA_DOCKER_IMAGE)

        print(f"   Models dir : {OLLAMA_MODELS_DIR}", flush=True)
        if Path(OLLAMA_MODELS_DIR).exists():
            # Mount existing local models dir so Ollama finds them immediately
            cmd += ["-v", f"{OLLAMA_MODELS_DIR}:/root/.ollama"]
        else:
            # Remote/fresh machine — model is baked into the image or will be pulled
            print(f"   ℹ️  Models dir not found locally — Ollama will use image-baked or pull models", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Docker run -d exits immediately; the container runs in background
        # proc here is just the `docker run` CLI process, not the container
        started_via = "docker"

    # ── Strategy 2: Native binary (RunPod, servers) ───────────────────────────
    else:
        # Find ollama binary
        ollama_bin = OLLAMA_BIN
        if not Path(ollama_bin).exists():
            # Common install locations
            for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama",
                               str(Path.home() / ".local/bin/ollama")]:
                if Path(candidate).exists():
                    ollama_bin = candidate
                    break
            else:
                print("❌ Ollama not found. Install from https://ollama.com or set OLLAMA_BIN",
                      flush=True)
                sys.exit(1)

        print(f"🚀 Starting Ollama natively ({ollama_bin})...", flush=True)
        proc = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        started_via = "native"

    # ── Wait for ready ────────────────────────────────────────────────────────
    for i in range(40):
        time.sleep(1)
        if _ollama_is_running():
            print(f"✓ Ollama ready via {started_via} (took {i+1}s)", flush=True)
            return proc
        if i % 5 == 0:
            print(f"   waiting for Ollama... ({i+1}/40)", flush=True)

    print("❌ Ollama did not start in time")
    sys.exit(1)


def _ollama_stop(proc: subprocess.Popen | None) -> None:
    """Stop what we started — Docker container or native process."""
    if proc is None:
        return
    print("🛑 Stopping Ollama...", flush=True)
    # If docker is available, stop the container by name (cleaner than killing proc)
    if _docker_available():
        subprocess.run(["docker", "stop", "ollama"],
                       capture_output=True, timeout=15)
        print("✓ Ollama container stopped")
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            print("✓ Ollama stopped")
        except Exception as e:
            print(f"⚠️  Could not stop Ollama cleanly: {e}")


# ── Translation script finder ─────────────────────────────────────────────────

def _find_translate_script() -> Path:
    for name in ["translate_diarize.py", "translate.py"]:
        p = TRANSLATE_DIR / name
        if p.exists():
            return p
    print(f"❌ No translate script found in {TRANSLATE_DIR}")
    print("   Expected: translate_diarize.py or translate.py")
    sys.exit(1)


def _derive_run_label(source_lang: str, target_lang: str, video: Path | None = None) -> str:
    """Create a stable folder name per video/language pair."""
    # Prefer SRT that matches the current video's base name
    base = None
    if video:
        video_base_norm = _normalize_base(re.split(r"[._]nemo|__", video.stem)[0])
        for srt in sorted(NEMO_DIR.glob(f"*.nemo.{source_lang}.diarize.srt")):
            if _normalize_base(re.split(r"[._]nemo|__", srt.stem)[0]) == video_base_norm:
                base = srt.stem
                break
    if not base:
        diarize_srts = sorted(NEMO_DIR.glob(f"*.nemo.{source_lang}.diarize.srt"))
        if diarize_srts:
            base = diarize_srts[0].stem
    if not base:
        base = video.stem if video else None

    if not base:
        base = f"run_{int(time.time())}"

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    base_label = f"{slug}__{source_lang}_to_{target_lang}"
    candidate = base_label
    idx = 2
    while (END_PRODUCT_DIR / candidate).exists():
        candidate = f"{base_label}__{idx}"
        idx += 1
    return candidate


def _finalize_outputs(run_label: str, dub_workdir: Path | None = None) -> None:
    """Clean subtitles and gather all outputs into nemo/end_product/<run>."""
    if not CLEAN_SUBS_SCRIPT.exists():
        print(f"⚠️  {CLEAN_SUBS_SCRIPT.name} not found — skipping cleanup")
        return

    clean_cmd = _python(TRANSLATE_PY, TRANSLATE_DIR) + [
        CLEAN_SUBS_SCRIPT.name,
        "--run-label", run_label,
    ]
    if dub_workdir:
        clean_cmd += ["--dub-workdir", str(dub_workdir)]
    _run(clean_cmd, cwd=TRANSLATE_DIR, label="Step 4 — Clean + gather outputs")


# ── SRT validation ───────────────────────────────────────────────────────────

def _validate_translated_srt(srt_path: Path, target_lang: str) -> None:
    """
    Sanity-check a translated SRT before passing it to the dub step.
    Fails hard if the file is empty, has no text, or still contains
    the original-language speaker tags with no translated content
    (which happens when the translation model was not found).
    """
    import re as _re
    text = srt_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        print(f"❌  Translated SRT is empty: {srt_path.name}")
        sys.exit(1)

    # Count subtitle blocks (non-empty lines after timestamp lines)
    content_lines = [
        l.strip() for l in text.splitlines()
        if l.strip()
        and not l.strip().isdigit()
        and "-->" not in l
    ]
    if not content_lines:
        print(f"❌  Translated SRT has no content lines: {srt_path.name}")
        sys.exit(1)

    # Heuristic: if every content line is just a [Speaker X] tag with nothing after,
    # translation silently failed (model returned empty strings)
    speaker_only = [l for l in content_lines if _re.fullmatch(r"\[Speaker\s+\d+\]", l)]
    if len(speaker_only) == len(content_lines):
        print(f"❌  Translated SRT contains only speaker tags — translation failed silently.")
        print(f"   Check Ollama is running and 'translategemma:4b' is pulled.")
        sys.exit(1)

    print(f"✅  SRT validation passed ({len(content_lines)} content lines)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="ASR dubbing pipeline: NeMo → Gemma translate → Qwen TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target-lang",    required=True,
                   help="Target dubbing language code, e.g. fr, en, es")
    p.add_argument("--language",       default=None,
                   help="Source language code (auto-detected from SRT if omitted)")
    p.add_argument("--trim",           type=int, default=0, metavar="SEC",
                   help="Process only first N seconds of video (default: full)")
    p.add_argument("--qwen-mode",      default="clone", choices=["clone", "custom"],
                   help="Voice mode: clone (default) or custom")
    p.add_argument("--no-demucs",      action="store_true",
                   help="Skip demucs — faster, no background music preserved")
    p.add_argument("--whisper-model",  default="medium",
                   choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                   help="Whisper model for language detection (default: medium)")
    p.add_argument("--skip-nemo",      action="store_true",
                   help="Skip NeMo step (diarized SRT already exists)")
    p.add_argument("--skip-translate", action="store_true",
                   help="Skip translation (translated SRT already exists)")
    p.add_argument("--skip-dub",       action="store_true",
                   help="Skip dubbing (only transcribe + translate)")
    p.add_argument("--run-mode",       default="full",
                   choices=["transcribe", "translate", "full"],
                   help=("Convenience preset: transcribe (Step 1 only), "
                         "translate (Steps 1-2), or full (default)"))
    p.add_argument("--input-dir",      default=None, metavar="DIR",
                   help="Folder containing input video(s) (default: nemo/)")
    p.add_argument("--output-dir",     default=None, metavar="DIR",
                   help="Folder for final outputs / end_product (default: nemo/end_product/)")
    # ── NeMo tuning flags (passed through to nemo.py) ────────────────────────
    p.add_argument("--precision",      default="bf16", choices=["fp32", "fp16", "bf16"],
                   help="ASR precision (default: bf16 — use fp16 on older GPUs, fp32 for max accuracy)")
    p.add_argument("--nemo-model",     default=None, metavar="MODEL",
                   help="Override NeMo model name (default: auto-selected by language)")
    p.add_argument("--chunk-override", default=None, type=int, metavar="SEC",
                   help="Force NeMo audio chunk size in seconds (default: auto from VRAM)")
    p.add_argument("--reserve-gb",     default=None, type=float, metavar="GB",
                   help="VRAM reserve for NeMo chunk estimation (default: 1.5)")
    p.add_argument("--safety-factor",  default=None, type=float, metavar="F",
                   help="VRAM safety multiplier for NeMo chunking (default: 0.85)")
    args = p.parse_args()

    # Convenience presets for common partial runs
    if args.run_mode == "transcribe":
        args.skip_translate = True
        args.skip_dub = True
    elif args.run_mode == "translate":
        args.skip_dub = True

    # ── Apply input/output dir overrides ────────────────────────────────────
    global NEMO_DIR, END_PRODUCT_DIR
    if args.input_dir:
        NEMO_DIR = Path(args.input_dir).resolve()
        print(f"📂 Input dir  : {NEMO_DIR}")
    if args.output_dir:
        END_PRODUCT_DIR = Path(args.output_dir).resolve()
        print(f"📂 Output dir : {END_PRODUCT_DIR}")
    END_PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Validate dirs ─────────────────────────────────────────────────────────
    for name, d in [("nemo", NEMO_DIR), ("qwen3-tts", QWEN_DIR)]:
        if not d.exists():
            print(f"❌  {name}/ not found at {d}"); sys.exit(1)

    # ── Step 0: Pick video + detect source language ──────────────────────────
    source_lang = args.language

    # Pin video FIRST — every skip-check anchors off the same file.
    # _find_video skips videos that already have a run-dir for this target lang.
    video = _find_video(target_lang=args.target_lang)
    if not video:
        print(f"❌  No unprocessed video found in {NEMO_DIR} for target '{args.target_lang}'")
        sys.exit(1)

    # Stable base = stem up to the first '.nemo' or '__' marker
    # "Debate 101..." -> "Debate 101..."   "impost_trimmed_2min" -> "impost_trimmed_2min"
    video_base = re.split(r"[._]nemo|__", video.stem)[0]
    print(f"🎬 Selected video : {video.name}  (base: '{video_base}')", flush=True)

    # Try to infer source lang from an existing diarized SRT for THIS video only
    # Also checks end_product/ in case clean_subs.py already moved files there
    if not source_lang:
        srt = _find_srt_for_video(video_base, "*.nemo.*.diarize.srt")
        if srt:
            m = re.search(r"\.nemo\.([a-z]{2,3})\.diarize\.srt$", srt.name)
            if m:
                source_lang = m.group(1)
                print(f"⏭️  Source language from existing SRT: '{source_lang}' ({srt.name})")

    # Fall back to Whisper detection on the chosen video
    if not source_lang:
        source_lang = _detect_source_language(video, whisper_model=args.whisper_model)
        if not source_lang:
            print("❌  Language detection failed. Pass --language explicitly.")
            sys.exit(1)

    # ── Step 1: NeMo ──────────────────────────────────────────────────────────
    # Only skip if an SRT exists whose base name matches THIS video
    # (checks both nemo/ and end_product/ in case files were already moved)
    existing_diarize_srt = _find_srt_for_video(video_base, f"*.nemo.{source_lang}.diarize.srt")

    if args.skip_nemo or existing_diarize_srt:
        if existing_diarize_srt:
            print(f"⏭️  Skipping NeMo — SRT already exists: {existing_diarize_srt.name}")
        else:
            print("⏭️  Skipping NeMo (--skip-nemo)")
    else:
        nemo_cmd = _python(NEMO_PY, NEMO_DIR) + [
            "nemo.py", str(video),
            "--language", source_lang, "--diarize",
            "--precision", args.precision,
        ]
        if args.trim:
            nemo_cmd += ["--trim", str(args.trim)]
        if args.nemo_model:
            nemo_cmd += ["--nemo-model", args.nemo_model]
        if args.chunk_override:
            nemo_cmd += ["--chunk-override", str(args.chunk_override)]
        if args.reserve_gb:
            nemo_cmd += ["--reserve-gb", str(args.reserve_gb)]
        if args.safety_factor:
            nemo_cmd += ["--safety-factor", str(args.safety_factor)]
        _run(nemo_cmd, cwd=NEMO_DIR, label="Step 1/3 — NeMo transcription + diarization")

    print(f"\n🌐 Source: {source_lang}  →  Target: {args.target_lang}")

    # ── Step 2: Translate (Gemma via Ollama) ──────────────────────────────────
    # Only skip if the translated SRT belongs to THIS video
    # (checks both nemo/ and end_product/ in case files were already moved)
    existing_translated_srt = _find_srt_for_video(video_base, f"*.diarize_{args.target_lang}.srt")

    ollama_proc = None
    if args.skip_translate or existing_translated_srt:
        if existing_translated_srt:
            print(f"⏭️  Skipping translation — SRT already exists: {existing_translated_srt.name}")
        else:
            print("⏭️  Skipping translation (--skip-translate)")
    elif not args.skip_translate:
        translate_script = _find_translate_script()
        # Use translate-gemma venv if it exists, else fall back to system python
        translate_py = str(TRANSLATE_PY) if TRANSLATE_PY.exists() else sys.executable
        if not TRANSLATE_PY.exists():
            print(f"⚠️  No venv at {TRANSLATE_PY}, using system python (pysrt may be missing)")

        ollama_proc = _ollama_start()
        try:
            env = os.environ.copy()
            env["TARGET_LANG_CODE"] = args.target_lang
            env["SOURCE_LANG_CODE"] = source_lang

            _banner(f"Step 2/3 — Translation ({source_lang} → {args.target_lang}) via Gemma")
            print(f"   cwd : {TRANSLATE_DIR}")
            print(f"   cmd : {translate_py} {translate_script.name}\n", flush=True)
            result = subprocess.run(
                [translate_py, str(translate_script)],
                cwd=str(TRANSLATE_DIR),
                env=env,
            )
            if result.returncode != 0:
                print(f"\n❌  Translation failed (exit {result.returncode})")
                sys.exit(result.returncode)
            print("\n✅  Translation done", flush=True)
        finally:
            _ollama_stop(ollama_proc)
    # ── Step 3: Dub ───────────────────────────────────────────────────────────
    dub_workdir = QWEN_DIR / "output" / "dub" / video_base  # always defined for finalize
    if not args.skip_dub:
        # Find the translated SRT — checks nemo/ AND end_product/ (post-clean location)
        dub_srt = _find_srt_for_video(video_base, f"*.diarize_{args.target_lang}.srt")
        if dub_srt is None:
            print(f"❌  No translated SRT found for '{video_base}' in {NEMO_DIR} or {END_PRODUCT_DIR}")
            sys.exit(1)
        print(f"📄 Using SRT : {dub_srt}")

        # Validate SRT has actual translated content — catch silent translation failures
        # before wasting GPU time on dubbing an empty/broken file
        _validate_translated_srt(dub_srt, args.target_lang)

        # Per-video workdir (defined above, create it now)
        dub_workdir.mkdir(parents=True, exist_ok=True)

        dub_cmd = _python(QWEN_PY, QWEN_DIR) + [
            "dub.py",
            str(video),          # explicit video — no auto-discovery
            str(dub_srt),        # explicit SRT   — no auto-discovery
            "--language",  args.target_lang,
            "--qwen-mode", args.qwen_mode,
            "--workdir",   str(dub_workdir),
        ]
        if args.no_demucs:
            dub_cmd.append("--no-demucs")

        _run(dub_cmd, cwd=QWEN_DIR, label="Step 3/3 — Dubbing with Qwen TTS")
    else:
        print("⏭️  Skipping dub (--skip-dub)")

    run_label = _derive_run_label(source_lang, args.target_lang, video=video)
    _finalize_outputs(run_label, dub_workdir=dub_workdir if not args.skip_dub else None)

    summary_lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                  ✅  Pipeline complete!                   ║",
    ]
    if not args.skip_dub:
        summary_lines.append("║  Dub : qwen3-tts/output/dub/output/final_dub.mp4       ║")
    summary_lines.append(f"║  End : {END_PRODUCT_DIR / run_label}                      ║")
    summary_lines.append("╚══════════════════════════════════════════════════════════╝")
    print("\n" + "\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()