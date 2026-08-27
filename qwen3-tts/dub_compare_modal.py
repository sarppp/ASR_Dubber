"""
dub_compare_modal.py — Dub one clip with several TTS engines in parallel on Modal
================================================================================

Runs the SAME dub.py pipeline (demucs + clone-mode TTS + speed-fit + stitch) once
per engine, each on its own T4 GPU, in parallel.  Every engine consumes the same
translated diarized SRT so the only variable is the TTS backend.

Outputs land in a local folder, one subdir per engine:

    comparison/<clip-stem>/qwen/final_dub.mp4      + .srt + timing json
    comparison/<clip-stem>/cosyvoice/final_dub.mp4 + .srt + timing json

── Prerequisites ──────────────────────────────────────────────────────────────

  1. modal auth (`modal setup`, or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in .env)
  2. A translated diarized SRT for the clip. Produce it with, e.g.:
       cd ../nemo && uv run --env-file .env modal run nemo_modal_app.py \
         --video-filename china_job_3min.mp4 --language en --translate fr --diarize
     → nemo/china_job_3min.nemo.fr.diarize.srt

── Usage ──────────────────────────────────────────────────────────────────────

  modal run dub_compare_modal.py \
    --video ../nemo/china_job_3min.mp4 \
    --srt   ../nemo/china_job_3min.nemo.fr.diarize.srt \
    --language fr

  # only one engine
  modal run dub_compare_modal.py --video ... --srt ... --engines cosyvoice
"""

import os
from pathlib import Path

import modal

APP_NAME = "dub-engine-compare"
GPU_TYPE = "T4"

QWEN_VOLUME = modal.Volume.from_name("qwen3-dubber-cache", create_if_missing=True)
COSY_VOLUME = modal.Volume.from_name("cosyvoice-dubber-cache", create_if_missing=True)

REMOTE_IO = Path("/app/dubber")
QWEN_VENV_PATH = "/opt/qwen3-tts"          # fake venv root so _qwen_python() resolves
COSY_VENV_PATH = "/opt/cosyvoice-venv"     # ditto for the cosyvoice worker
COSY_REPO_DIR = "/opt/CosyVoice"
COSY_MODEL_DIR = "/app/dubber/models/Fun-CosyVoice3-0.5B"

PIPELINE_FILES = ("dub.py", "dub_srt.py", "dub_audio.py",
                  "qwen_tts_worker.py", "cosyvoice_tts_worker.py")

# ---------------------------------------------------------------------------
# Shared base: conda + nemo-env (runs dub.py: demucs, ffmpeg, pysrt, tqdm)
# ---------------------------------------------------------------------------

def _base_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.12"
        )
        .apt_install("ffmpeg", "git", "curl", "ca-certificates", "bash",
                     "sox", "libsox-fmt-all", "unzip")
        .env({"PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .run_commands(
            "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh"
            " -o /tmp/miniconda.sh",
            "bash /tmp/miniconda.sh -b -p /opt/conda",
            "rm -f /tmp/miniconda.sh",
        )
        .run_commands(
            "bash -lc '/opt/conda/bin/conda create -n nemo-env python=3.12 -y'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install -U pip'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install \"numpy<2.0\"'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env"
            " && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env"
            " && pip install soundfile demucs==4.0.1 pysrt tqdm librosa'",
            "ln -sf /opt/conda/envs/nemo-env/bin/python /usr/local/bin/python",
            "ln -sf /opt/conda/envs/nemo-env/bin/demucs /usr/local/bin/demucs",
        )
    )


def _with_pipeline(img: modal.Image) -> modal.Image:
    for f in PIPELINE_FILES:
        img = img.add_local_file(f, remote_path=f"/root/{f}")
    return img


# ---------------------------------------------------------------------------
# Qwen image  (adds qwen3-tts conda env)
# ---------------------------------------------------------------------------

qwen_image = _with_pipeline(
    _base_image()
    .run_commands(
        "bash -lc '/opt/conda/bin/conda create -n qwen3-tts python=3.12 -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts && pip install -U pip'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts"
        " && pip install transformers==4.57.3 tokenizers==0.22.2'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts && pip install -U qwen-tts'",
        f"mkdir -p {QWEN_VENV_PATH}/.venv/bin",
        f"ln -sf /opt/conda/envs/qwen3-tts/bin/python {QWEN_VENV_PATH}/.venv/bin/python",
    )
)

# ---------------------------------------------------------------------------
# CosyVoice image  (adds cosyvoice conda env + repo checkout)
# ---------------------------------------------------------------------------

cosy_image = _with_pipeline(
    _base_image()
    .run_commands(
        f"git clone --recursive https://github.com/FunAudioLLM/CosyVoice {COSY_REPO_DIR}",
        "bash -lc '/opt/conda/bin/conda create -n cosyvoice python=3.10 -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice && pip install -U pip'",
        # torch first, pinned to the CosyVoice-tested version
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice"
        " && pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121'",
        # the rest of CosyVoice deps, minus tensorrt / deepspeed (inference-only, load_trt=False)
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate cosyvoice && pip install "
        "conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 HyperPyYAML==1.2.3 inflect==7.3.1 "
        "librosa==0.10.2 lightning==2.2.4 modelscope==1.20.0 networkx==3.1 numpy==1.26.4 omegaconf==2.3.0 "
        "onnx==1.16.0 onnxruntime-gpu==1.18.0 openai-whisper==20231117 protobuf==4.25 pyarrow==18.1.0 "
        "pydantic==2.7.0 pyworld==0.3.4 rich==13.7.1 soundfile==0.12.1 transformers==4.51.3 "
        "x-transformers==2.11.24 wetext==0.0.4 gdown==5.1.0 wget==3.2 huggingface_hub'",
        f"mkdir -p {COSY_VENV_PATH}/.venv/bin",
        f"ln -sf /opt/conda/envs/cosyvoice/bin/python {COSY_VENV_PATH}/.venv/bin/python",
    )
    .env({"COSYVOICE_REPO_DIR": COSY_REPO_DIR, "COSYVOICE_MODEL_DIR": COSY_MODEL_DIR})
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Shared dub runner
# ---------------------------------------------------------------------------

def _run_dub(engine: str, tts_python: str,
             video_filename: str, video_data: bytes,
             srt_filename: str, srt_data: bytes,
             target_lang: str, no_demucs: bool) -> dict:
    import shutil
    import subprocess as sp
    import time

    REMOTE_IO.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = (
        "/opt/conda/envs/nemo-env/bin:" + os.environ.get("PATH", "")
    )

    code_dir = REMOTE_IO / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for f in PIPELINE_FILES:
        src = Path("/root") / f
        if src.exists():
            shutil.copyfile(src, code_dir / f)

    video_path = REMOTE_IO / video_filename
    video_path.write_bytes(video_data)
    srt_path = REMOTE_IO / srt_filename
    srt_path.write_bytes(srt_data)

    work_dir = REMOTE_IO / "work" / engine / video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/opt/conda/envs/nemo-env/bin/python", str(code_dir / "dub.py"),
        str(video_path), str(srt_path),
        "--language", target_lang,
        "--tts-engine", engine,
        "--qwen-mode", "clone",
        "--qwen-dir", tts_python,
        "--workdir", str(work_dir),
        "--tts-workers", "2",
    ]
    if no_demucs:
        cmd.append("--no-demucs")

    t0 = time.perf_counter()
    result = sp.run(cmd, env=os.environ)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"[{engine}] dub.py failed rc={result.returncode}")

    out_mp4 = work_dir / "output" / "final_dub.mp4"
    if not out_mp4.exists():
        cand = list((work_dir / "output").glob("*.mp4"))
        if not cand:
            raise FileNotFoundError(f"[{engine}] no output mp4 under {work_dir/'output'}")
        out_mp4 = cand[0]

    srt_out = next(iter((work_dir / "output").glob("*_dub.srt")), None)
    timing = next(iter((work_dir / "output").glob("*_timing_dub.json")), None)

    video_path.unlink(missing_ok=True)
    return {
        "engine": engine,
        "elapsed_sec": round(elapsed, 1),
        "mp4": out_mp4.read_bytes(),
        "srt": srt_out.read_bytes() if srt_out else None,
        "timing": timing.read_bytes() if timing else None,
    }


@app.function(image=qwen_image, gpu=GPU_TYPE, volumes={str(REMOTE_IO): QWEN_VOLUME},
              timeout=60 * 60 * 2)
def dub_qwen(video_filename: str, video_data: bytes, srt_filename: str,
             srt_data: bytes, target_lang: str, no_demucs: bool) -> dict:
    return _run_dub("qwen", QWEN_VENV_PATH, video_filename, video_data,
                    srt_filename, srt_data, target_lang, no_demucs)


@app.function(image=cosy_image, gpu=GPU_TYPE, volumes={str(REMOTE_IO): COSY_VOLUME},
              timeout=60 * 60 * 2)
def dub_cosyvoice(video_filename: str, video_data: bytes, srt_filename: str,
                  srt_data: bytes, target_lang: str, no_demucs: bool) -> dict:
    # First run downloads the weights into the volume; later runs reuse them.
    from huggingface_hub import snapshot_download
    Path(COSY_MODEL_DIR).parent.mkdir(parents=True, exist_ok=True)
    if not (Path(COSY_MODEL_DIR) / "cosyvoice3.yaml").exists():
        print("[cosyvoice] downloading Fun-CosyVoice3-0.5B weights…")
        snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=COSY_MODEL_DIR)
        COSY_VOLUME.commit()
    return _run_dub("cosyvoice", COSY_VENV_PATH, video_filename, video_data,
                    srt_filename, srt_data, target_lang, no_demucs)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(video: str, srt: str, language: str = "fr",
         engines: str = "qwen,cosyvoice", no_demucs: bool = False,
         out_dir: str = "comparison"):
    video_path = Path(video).resolve()
    srt_path = Path(srt).resolve()
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")
    if not srt_path.exists():
        raise SystemExit(f"srt not found: {srt_path}")

    engine_list = [e.strip() for e in engines.split(",") if e.strip()]
    fn_map = {"qwen": dub_qwen, "cosyvoice": dub_cosyvoice}
    for e in engine_list:
        if e not in fn_map:
            raise SystemExit(f"unknown engine {e!r}; choose from {list(fn_map)}")

    video_data = video_path.read_bytes()
    srt_data = srt_path.read_bytes()
    print(f"Clip : {video_path.name}  ({len(video_data)/1e6:.1f} MB)")
    print(f"SRT  : {srt_path.name}")
    print(f"Engines (parallel, {GPU_TYPE} each): {', '.join(engine_list)}\n")

    handles = {
        e: fn_map[e].spawn(video_path.name, video_data, srt_path.name,
                           srt_data, language, no_demucs)
        for e in engine_list
    }

    base = Path(out_dir) / video_path.stem
    for e, h in handles.items():
        try:
            res = h.get()
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {e}: {exc}")
            continue
        dest = base / e
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "final_dub.mp4").write_bytes(res["mp4"])
        if res["srt"]:
            (dest / "final_dub.srt").write_bytes(res["srt"])
        if res["timing"]:
            (dest / "timing_dub.json").write_bytes(res["timing"])
        print(f"✅ {e}: {dest}/final_dub.mp4   (dub stage {res['elapsed_sec']}s)")

    print(f"\nCompare the mp4s under: {base.resolve()}")
