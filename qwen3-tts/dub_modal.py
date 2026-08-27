"""
dub_modal.py — Qwen TTS Dubbing Pipeline on Modal
==================================================

Dubs a video using a pre-translated diarized SRT, running the full pipeline
(demucs vocal separation + Qwen TTS + stitch) on a Modal GPU.

For a choice of TTS engine (Qwen3-TTS *or* Fun-CosyVoice3-0.5B), or to run
both engines in parallel for A/B comparison, use ``dub_compare_modal.py``:

    modal run dub_compare_modal.py --video V.mp4 --srt V.nemo.LANG.diarize_fr.srt \
        --language fr --engines cosyvoice

── Prerequisites ──────────────────────────────────────────────────────────────

  1. modal setup  (or .env with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
  2. A translated diarized SRT: *.nemo.LANG.diarize_TARGETLANG.srt
     Produced by: nemo_modal_app.py --diarize  then  translate.py
  3. The matching video file

── Usage ──────────────────────────────────────────────────────────────────────

  # Auto-discover video + SRT in current directory (looks for *.diarize_??.srt)
  modal run dub_modal.py --language fr

  # Explicit files
  modal run dub_modal.py --video myvideo.mp4 --srt myvideo.nemo.de.diarize_fr.srt --language fr

  # SRT lives in NeMo output folder, video is here
  modal run dub_modal.py --language fr --search-dir ../nemo

  # Clone voices from speakers (default) vs fixed preset voices
  modal run dub_modal.py --language fr --qwen-mode clone
  modal run dub_modal.py --language fr --qwen-mode custom

  # Skip demucs (faster, replaces full audio track)
  modal run dub_modal.py --language fr --no-demucs

  # Save output to a specific file
  modal run dub_modal.py --language fr --out my_dubbed.mp4

── GPU ────────────────────────────────────────────────────────────────────────

  GPU_TYPE near the top of this file controls the GPU (default: A10G 24 GB).
  A10G fits 3 TTS workers, T4 fits 2.
"""

import os
import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME    = "qwen3-tts-dubber"
VOLUME_NAME = "qwen3-dubber-cache"
GPU_TYPE    = "A10G"          # A10G (24 GB) recommended; T4 (16 GB) also works

REMOTE_IO_PATH = Path("/app/dubber")
QWEN_VENV_PATH = "/opt/qwen3-tts"   # fake venv root so _qwen_python() finds the conda python
NEMO_SITE      = "/opt/conda/envs/nemo-env/lib/python3.12/site-packages"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "ffmpeg", "git", "curl", "ca-certificates", "bash", "sox", "libsox-fmt-all",
    )
    .env({"PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    # Install Miniconda
    .run_commands(
        "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh"
        " -o /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm -f /tmp/miniconda.sh",
    )
    # nemo-env: runs main dub.py process (demucs, ffmpeg, pysrt, tqdm)
    .run_commands(
        "bash -lc '/opt/conda/bin/conda create -n nemo-env python=3.12 -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env"
        " && pip install -U pip'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env"
        ' && pip install "numpy<2.0"\'',
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env"
        " && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env"
        " && pip install soundfile demucs==4.0.1 pysrt tqdm'",
    )
    # qwen3-tts env: runs qwen_tts_worker.py subprocess
    .run_commands(
        "bash -lc '/opt/conda/bin/conda create -n qwen3-tts python=3.12 -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts"
        " && pip install -U pip'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts"
        " && pip install transformers==4.57.3 tokenizers==0.22.2'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts"
        " && pip install -U qwen-tts'",
    )
    # Convenience symlinks + fake .venv so _qwen_python() in dub_audio.py resolves correctly
    .run_commands(
        f"mkdir -p {QWEN_VENV_PATH}/.venv/bin",
        f"ln -sf /opt/conda/envs/qwen3-tts/bin/python {QWEN_VENV_PATH}/.venv/bin/python",
        "ln -sf /opt/conda/envs/nemo-env/bin/python /usr/local/bin/python",
        "ln -sf /opt/conda/envs/nemo-env/bin/pip /usr/local/bin/pip",
        "ln -sf /opt/conda/envs/nemo-env/bin/demucs /usr/local/bin/demucs",
    )
    # Pipeline scripts (re-uploaded on every `modal deploy` / `modal run`)
    .add_local_file("dub.py",             remote_path="/root/dub.py")
    .add_local_file("dub_srt.py",         remote_path="/root/dub_srt.py")
    .add_local_file("dub_audio.py",       remote_path="/root/dub_audio.py")
    .add_local_file("qwen_tts_worker.py", remote_path="/root/qwen_tts_worker.py")
)

app = modal.App(name=APP_NAME)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={str(REMOTE_IO_PATH): volume},
    gpu=GPU_TYPE,
    timeout=60 * 60 * 2,   # 2 hours
)
def dub_remote(
    video_filename: str,
    video_data: bytes,
    srt_filename: str,
    srt_data: bytes,
    target_lang: str,
    qwen_mode: str,
    no_demucs: bool,
    tts_workers: str,
) -> tuple[bytes, bytes | None]:
    import shutil
    import subprocess as sp
    import sys

    REMOTE_IO_PATH.mkdir(parents=True, exist_ok=True)

    # Add nemo-env packages to sys.path for any direct imports (pysrt etc.)
    if NEMO_SITE not in sys.path:
        sys.path.insert(0, NEMO_SITE)

    # Extend PATH so demucs, ffmpeg etc. are found
    os.environ["PATH"] = (
        "/opt/conda/envs/nemo-env/bin:"
        "/opt/conda/envs/qwen3-tts/bin:"
        + os.environ.get("PATH", "")
    )

    # Copy pipeline scripts to a stable code dir inside the volume
    code_dir = REMOTE_IO_PATH / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("dub.py", "dub_srt.py", "dub_audio.py", "qwen_tts_worker.py"):
        src = Path("/root") / fname
        dst = code_dir / fname
        if src.exists():
            shutil.copyfile(src, dst)

    # Write input files
    video_path = REMOTE_IO_PATH / video_filename
    video_path.write_bytes(video_data)
    srt_path   = REMOTE_IO_PATH / srt_filename
    srt_path.write_bytes(srt_data)

    # Work dir persists in the volume → checkpoint reuse on reruns
    work_dir = REMOTE_IO_PATH / "work" / video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/opt/conda/envs/nemo-env/bin/python",
        str(code_dir / "dub.py"),
        str(video_path),
        str(srt_path),
        "--language",    target_lang,
        "--qwen-mode",   qwen_mode,
        "--qwen-dir",    QWEN_VENV_PATH,
        "--workdir",     str(work_dir),
        "--tts-workers", tts_workers,
    ]
    if no_demucs:
        cmd.append("--no-demucs")

    result = sp.run(cmd, env=os.environ)
    if result.returncode != 0:
        raise RuntimeError(f"dub.py failed with exit code {result.returncode}")

    out_path = work_dir / "output" / "final_dub.mp4"
    if not out_path.exists():
        # Fallback: find any mp4 in the output dir
        candidates = list((work_dir / "output").glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(
                f"No output MP4 found under {work_dir / 'output'}. "
                "Check dub.py logs above."
            )
        out_path = candidates[0]

    out_bytes = out_path.read_bytes()

    # Dub SRT written by dub.py next to the MP4
    dub_srt_candidates = list((work_dir / "output").glob("*_dub.srt"))
    dub_srt_bytes = dub_srt_candidates[0].read_bytes() if dub_srt_candidates else None

    # Clean up input files; keep work dir for checkpoint reuse
    video_path.unlink(missing_ok=True)
    srt_path.unlink(missing_ok=True)

    return out_bytes, dub_srt_bytes


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _find_srt_and_video(search_dir: str | None) -> tuple:
    """
    Auto-discover the most recently modified translated SRT + matching video.

    Search order:
      1. Explicit --search-dir (if given)
      2. ../nemo/end_product  — subdirs whose name contains "nemo", newest first
      3. Current directory (flat)
    """
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    local_dir = Path(".").resolve()

    def _srts_in(folder: Path) -> list[Path]:
        return sorted(folder.glob("*.diarize_??.srt"))

    def _video_for_srt(srt: Path) -> Path | None:
        stem_match = re.match(r"^(.+?)\.nemo\.", srt.name)
        video_stem = stem_match.group(1) if stem_match else None
        for search in (srt.parent, local_dir):
            if video_stem:
                matches = [
                    f for f in search.iterdir()
                    if f.stem == video_stem and f.suffix.lower() in VIDEO_EXT
                ]
                if matches:
                    return matches[0]
        return None

    # ── 1. Explicit search dir ───────────────────────────────────────────────
    if search_dir:
        base = Path(search_dir).resolve()
        srts = _srts_in(base)
        if srts:
            return srts[0], _video_for_srt(srts[0])

    # ── 2. ../nemo/end_product — subdirs with "nemo" in name, newest first ──
    end_product = (local_dir / "../nemo/end_product").resolve()
    if end_product.is_dir():
        nemo_subdirs = sorted(
            [d for d in end_product.iterdir() if d.is_dir() and "nemo" in d.name],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for subdir in nemo_subdirs:
            srts = _srts_in(subdir)
            if srts:
                return srts[0], _video_for_srt(srts[0])

    # ── 3. Current dir flat ──────────────────────────────────────────────────
    srts = _srts_in(local_dir)
    if srts:
        return srts[0], _video_for_srt(srts[0])

    return None, None


@app.local_entrypoint()
def main(
    video: str = None,
    srt: str = None,
    language: str = "fr",
    qwen_mode: str = "clone",
    no_demucs: bool = False,
    search_dir: str = None,
    out: str = "output/final_dub.mp4",
    tts_workers: str = "auto",
):
    """
    Dub a video on Modal using Qwen TTS.

    Auto-discovers video + translated SRT from ../nemo/end_product (newest
    subfolder whose name contains "nemo"), then current dir as fallback.
    SRT pattern: *.nemo.LANG.diarize_TARGETLANG.srt

    Examples:
      modal run dub_modal.py --language fr
      modal run dub_modal.py --language fr --search-dir ../nemo/end_product/myfolder
      modal run dub_modal.py --video clip.mp4 --srt clip.nemo.de.diarize_fr.srt --language fr
      modal run dub_modal.py --language fr --qwen-mode custom --no-demucs
    """
    local_dir = Path(".").resolve()
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    # ── Locate SRT ──────────────────────────────────────────────────────────
    if srt is None:
        chosen_srt, auto_video = _find_srt_and_video(search_dir)
        if chosen_srt is None:
            print("No translated SRTs found.")
            print("Searched: ../nemo/end_product/*nemo*/ and current directory.")
            print("Expected pattern: *.nemo.LANG.diarize_TARGETLANG.srt")
            print("Run nemo_modal_app.py --diarize, then translate.py first.")
            return
    else:
        p = Path(srt)
        chosen_srt = p if p.is_absolute() else local_dir / srt
        auto_video = None

    if not chosen_srt.exists():
        print(f"SRT not found: {chosen_srt}")
        return

    # ── Locate video ────────────────────────────────────────────────────────
    if video is None:
        chosen_video = auto_video
        if chosen_video is None:
            videos = [f for f in local_dir.iterdir() if f.suffix.lower() in VIDEO_EXT]
            if not videos:
                print(f"No video files found for SRT: {chosen_srt.name}")
                return
            chosen_video = videos[0]
    else:
        p = Path(video)
        chosen_video = p if p.is_absolute() else local_dir / video

    if not chosen_video.exists():
        print(f"Video not found: {chosen_video}")
        return

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Qwen TTS Dubbing Pipeline  (Modal / {GPU_TYPE})")
    print(f"{'=' * 60}")
    print(f"Video      : {chosen_video}")
    print(f"SRT        : {chosen_srt.name}")
    print(f"Language   : {language}")
    print(f"Mode       : {qwen_mode}")
    print(f"Demucs     : {'disabled' if no_demucs else 'enabled'}")
    print(f"TTS workers: {tts_workers}")
    print(f"{'=' * 60}\n")

    print("Reading video...")
    video_data = chosen_video.read_bytes()
    print(f"  {len(video_data) / 1024 / 1024:.1f} MB")

    print("Reading SRT...")
    srt_data = chosen_srt.read_bytes()

    print("Submitting to Modal...\n")
    out_bytes, dub_srt_bytes = dub_remote.remote(
        video_filename=chosen_video.name,
        video_data=video_data,
        srt_filename=chosen_srt.name,
        srt_data=srt_data,
        target_lang=language,
        qwen_mode=qwen_mode,
        no_demucs=no_demucs,
        tts_workers=tts_workers,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out_bytes)
    print(f"\nDone!  Output: {out_path.resolve()}")

    if dub_srt_bytes:
        srt_out = out_path.with_suffix(".srt")
        srt_out.write_bytes(dub_srt_bytes)
        print(f"       SRT:    {srt_out.resolve()}")
