"""
pipeline_paths.py — Path helpers, language detection, output management,
and SRT validation for the ASR dubbing pipeline.

Functions that access NEMO_DIR / END_PRODUCT_DIR accept keyword-only path
parameters (defaulting to the module-level constants from pipeline_utils)
so that main() can pass overridden paths without relying on global mutation.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

from pipeline_utils import (
    CLEAN_SUBS_SCRIPT,
    END_PRODUCT_DIR,
    NEMO_DIR,
    TRANSLATE_DIR,
    TRANSLATE_PY,
    WHISPER_DIR,
    WHISPER_PY,
    _banner,
    _python,
    _run,
)


# ── Source language detection ─────────────────────────────────────────────────

def _normalize_base(s: str) -> str:
    """Lowercase + collapse any run of non-alphanumeric chars to '_' for fuzzy matching.
    'Debate 101 with Harvard\'s former...' == 'Debate_101_with_Harvard_s_former...'
    """
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _video_already_processed(
    video: Path,
    target_lang: str | None = None,
    *,
    end_product_dir: Path | None = None,
) -> bool:
    """Return True only if this exact video+target_lang pair has a finished run dir."""
    end_product_dir = end_product_dir or END_PRODUCT_DIR
    if not end_product_dir.exists():
        return False
    base = _normalize_base(re.split(r"[._]nemo|__", video.stem)[0])
    for run_dir in end_product_dir.iterdir():
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


def _find_video(
    target_lang: str | None = None,
    *,
    nemo_dir: Path | None = None,
    end_product_dir: Path | None = None,
) -> Path | None:
    """Find the newest unprocessed video file in nemo/ dir."""
    nemo_dir = nemo_dir or NEMO_DIR
    end_product_dir = end_product_dir or END_PRODUCT_DIR
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    videos = sorted(
        (f for f in nemo_dir.iterdir() if f.suffix.lower() in VIDEO_EXT),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for video in videos:
        if not _video_already_processed(video, target_lang=target_lang,
                                        end_product_dir=end_product_dir):
            return video
    return videos[0] if videos else None


def _find_srt_for_video(
    video_base: str,
    pattern: str,
    *,
    nemo_dir: Path | None = None,
    end_product_dir: Path | None = None,
) -> Path | None:
    """
    Find an SRT matching `pattern` (a glob) for this video_base.
    Checks nemo/ first, then end_product/<run_dir>/ as fallback
    (clean_subs.py moves files there after a completed run).
    Uses normalized comparison so spaces vs underscores don't matter.
    """
    nemo_dir = nemo_dir or NEMO_DIR
    end_product_dir = end_product_dir or END_PRODUCT_DIR
    norm_base = _normalize_base(video_base)

    # 1. Live location — nemo/
    for srt in sorted(nemo_dir.glob(pattern)):
        srt_base = _normalize_base(re.split(r"[._]nemo|__", srt.stem)[0])
        if srt_base == norm_base:
            return srt

    # 2. Archived location — end_product/<any run_dir>/
    if end_product_dir.exists():
        for run_dir in sorted(end_product_dir.iterdir()):
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
            cwd=str(video_path.parent),
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


# ── Translation script finder ─────────────────────────────────────────────────

def _find_translate_script() -> Path:
    for name in ["translate_diarize.py", "translate.py"]:
        p = TRANSLATE_DIR / name
        if p.exists():
            return p
    print(f"❌ No translate script found in {TRANSLATE_DIR}")
    print("   Expected: translate_diarize.py or translate.py")
    sys.exit(1)


def _derive_run_label(
    source_lang: str,
    target_lang: str,
    video: Path | None = None,
    *,
    nemo_dir: Path | None = None,
    end_product_dir: Path | None = None,
) -> str:
    """Create a stable folder name per video/language pair."""
    nemo_dir = nemo_dir or NEMO_DIR
    end_product_dir = end_product_dir or END_PRODUCT_DIR
    # Prefer SRT that matches the current video's base name
    base = None
    if video:
        video_base_norm = _normalize_base(re.split(r"[._]nemo|__", video.stem)[0])
        for srt in sorted(nemo_dir.glob(f"*.nemo.{source_lang}.diarize.srt")):
            if _normalize_base(re.split(r"[._]nemo|__", srt.stem)[0]) == video_base_norm:
                base = srt.stem
                break
    if not base:
        diarize_srts = sorted(nemo_dir.glob(f"*.nemo.{source_lang}.diarize.srt"))
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
    while (end_product_dir / candidate).exists():
        candidate = f"{base_label}__{idx}"
        idx += 1
    return candidate


def _finalize_outputs(run_label: str, dub_workdir: Path | None = None, *, nemo_dir: Path | None = None) -> None:
    """Clean subtitles and gather all outputs into nemo/end_product/<run>."""
    nemo_dir = nemo_dir or NEMO_DIR
    if not CLEAN_SUBS_SCRIPT.exists():
        print(f"⚠️  {CLEAN_SUBS_SCRIPT.name} not found — skipping cleanup")
        return

    clean_cmd = _python(TRANSLATE_PY, TRANSLATE_DIR) + [
        str(CLEAN_SUBS_SCRIPT),
        "--run-label", run_label,
    ]
    if dub_workdir:
        clean_cmd += ["--dub-workdir", str(dub_workdir)]
    _run(clean_cmd, cwd=nemo_dir, label="Step 4 — Clean + gather outputs")


# ── SRT validation ────────────────────────────────────────────────────────────

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
        print(f"   Check Ollama is running and the translate model is pulled.")
        sys.exit(1)

    print(f"✅  SRT validation passed ({len(content_lines)} content lines)")
