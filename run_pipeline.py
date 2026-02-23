#!/usr/bin/env python3
"""
run_pipeline.py — Master runner for the ASR dubbing pipeline
=============================================================
Place in ASR/ root. Runs all three steps end-to-end.

Usage:
  python run_pipeline.py --target-lang fr              # auto-detect source lang
  python run_pipeline.py --target-lang fr --trim 30   # first 30s only
  python run_pipeline.py --target-lang fr --skip-nemo  # SRT already exists
  python run_pipeline.py --target-lang fr --skip-translate
  python run_pipeline.py --target-lang fr --qwen-mode custom

Source language is auto-detected from the diarized SRT filename after NeMo runs.
If skipping NeMo, pass --language explicitly.

# Force source language manually (skips Whisper detection)
python run_pipeline.py --target-lang fr --language de

# Default now uses medium (works with all whisper builds)
python run_pipeline.py --target-lang fr --trim 30

# Explicitly choose model
python run_pipeline.py --target-lang fr --trim 30 --whisper-model small
python run_pipeline.py --target-lang fr --trim 30 --whisper-model medium

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

OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_BIN         = os.getenv("OLLAMA_BIN", "ollama")   # native binary path
OLLAMA_DOCKER_IMAGE = os.getenv("OLLAMA_DOCKER_IMAGE", "ollama/ollama")
OLLAMA_MODELS_DIR  = os.getenv("OLLAMA_MODELS_DIR", "/home/sarpk/python-tools/.ollama_models")


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

def _find_video() -> Path | None:
    """Find a video file in nemo/ dir."""
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    videos = [f for f in NEMO_DIR.iterdir() if f.suffix.lower() in VIDEO_EXT]
    return videos[0] if videos else None


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

    print(f"🔍 Detecting source language (Whisper, 30s sample)...", flush=True)
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
            "-v", f"{OLLAMA_MODELS_DIR}:/root/.ollama",
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
    args = p.parse_args()

    # ── Validate dirs ─────────────────────────────────────────────────────────
    for name, d in [("nemo", NEMO_DIR), ("qwen3-tts", QWEN_DIR)]:
        if not d.exists():
            print(f"❌  {name}/ not found at {d}"); sys.exit(1)

    # ── Step 0: Detect source language ───────────────────────────────────────
    source_lang = args.language

    # First try: parse from existing diarized SRT filename (free, instant)
    if not source_lang:
        existing = sorted(NEMO_DIR.glob("*.nemo.*.diarize.srt"))
        if existing:
            m = re.search(r"\.nemo\.([a-z]{2,3})\.diarize\.srt$", existing[0].name)
            if m:
                source_lang = m.group(1)
                print(f"⏭️  Source language from existing SRT: '{source_lang}' ({existing[0].name})")

    # Second try: Whisper detection (only if no SRT exists yet)
    if not source_lang:
        video = _find_video()
        if not video:
            print(f"❌  No video found in {NEMO_DIR}"); sys.exit(1)
        source_lang = _detect_source_language(video, whisper_model=args.whisper_model)
        if not source_lang:
            print("❌  Language detection failed. Pass --language explicitly.")
            sys.exit(1)

    # ── Step 1: NeMo ──────────────────────────────────────────────────────────
    existing_diarize_srts = sorted(NEMO_DIR.glob(f"*.nemo.{source_lang}.diarize.srt"))
    if args.skip_nemo or existing_diarize_srts:
        if existing_diarize_srts:
            print(f"⏭️  Skipping NeMo — SRT already exists: {existing_diarize_srts[0].name}")
        else:
            print("⏭️  Skipping NeMo (--skip-nemo)")
    else:
        nemo_cmd = _python(NEMO_PY, NEMO_DIR) + [
            "nemo.py", "--language", source_lang, "--diarize"
        ]
        if args.trim:
            nemo_cmd += ["--trim", str(args.trim)]
        _run(nemo_cmd, cwd=NEMO_DIR, label="Step 1/3 — NeMo transcription + diarization")

    print(f"\n🌐 Source: {source_lang}  →  Target: {args.target_lang}")

    # ── Step 2: Translate (Gemma via Ollama) ──────────────────────────────────
    existing_translated_srts = sorted(NEMO_DIR.glob(f"*.diarize_{args.target_lang}.srt"))
    ollama_proc = None
    if args.skip_translate or existing_translated_srts:
        if existing_translated_srts:
            print(f"⏭️  Skipping translation — SRT already exists: {existing_translated_srts[0].name}")
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
    if not args.skip_dub:
        dub_cmd = _python(QWEN_PY, QWEN_DIR) + [
            "dub.py",
            "--language",  args.target_lang,
            "--qwen-mode", args.qwen_mode,
        ]
        if args.no_demucs:
            dub_cmd.append("--no-demucs")

        _run(dub_cmd, cwd=QWEN_DIR, label="Step 3/3 — Dubbing with Qwen TTS")
    else:
        print("⏭️  Skipping dub (--skip-dub)")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                  ✅  Pipeline complete!                   ║
║  Output: qwen3-tts/output/dub/output/final_dub.mp4       ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
