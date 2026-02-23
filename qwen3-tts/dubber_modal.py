import os
from pathlib import Path

import modal


APP_NAME = "qwen3-dubber"
REMOTE_IO_PATH = Path("/app/dubber")
VOLUME_NAME = "qwen3-dubber-cache"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _image():
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
            add_python="3.12",
        )
        .apt_install(
            "ffmpeg",
            "git",
            "curl",
            "ca-certificates",
            "bash",
            "sox",
            "libsox-fmt-all",
        )
        .env({"PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .run_commands(
            "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -o /tmp/miniconda.sh",
            "bash /tmp/miniconda.sh -b -p /opt/conda",
            "rm -f /tmp/miniconda.sh",
        )
        .run_commands(
            "bash -lc '/opt/conda/bin/conda create -n nemo-env python=3.12 -y'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install -U pip'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install \"numpy<2.0\"'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install soundfile demucs==4.0.1 deep-translator==1.11.4 python-dotenv nemo_toolkit[asr] whisperx'",
        )
        .run_commands(
            "bash -lc '/opt/conda/bin/conda create -n qwen3-tts python=3.12 -y'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts && pip install -U pip'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts && pip install transformers==4.57.3 tokenizers==0.22.2'",
            "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts && pip install -U qwen-tts'",
            "bash -lc 'ln -sf /opt/conda/envs/nemo-env/bin/python /usr/local/bin/python'",
            "bash -lc 'ln -sf /opt/conda/envs/nemo-env/bin/pip /usr/local/bin/pip'",
        )
        .add_local_file("fish_qwen_copy.py", remote_path="/root/fish_qwen_copy.py")
        .add_local_file("nemo_diarization_report.py", remote_path="/root/nemo_diarization_report.py")
        .add_local_file("whisper_nemo_report.py", remote_path="/root/whisper_nemo_report.py")
        .add_local_file("qwen_tts_worker.py", remote_path="/root/qwen_tts_worker.py")
    )


image = _image()

app = modal.App(name=APP_NAME)


@app.function(
    image=image,
    volumes={REMOTE_IO_PATH: volume},
    gpu="T4",
    timeout=60 * 60,
)
def dub_video_remote(
    video_filename: str,
    video_data: bytes,
    target_lang: str,
    qwen_mode: str,
) -> bytes:
    import shutil
    import site
    import sys

    REMOTE_IO_PATH.mkdir(parents=True, exist_ok=True)

    os.environ["PATH"] = "/opt/conda/envs/nemo-env/bin:" + os.environ.get("PATH", "")

    site.addsitedir("/opt/conda/envs/nemo-env/lib/python3.12/site-packages")

    sys.executable = "/opt/conda/envs/nemo-env/bin/python"

    code_dir = Path(__file__).resolve().parent
    for helper_name in ("fish_qwen_copy.py", "nemo_diarization_report.py", "whisper_nemo_report.py", "qwen_tts_worker.py"):
        src = code_dir / helper_name
        dst = REMOTE_IO_PATH / helper_name
        if src.exists():
            shutil.copyfile(src, dst)

    sys.path.insert(0, str(REMOTE_IO_PATH))
    from fish_qwen_copy import DockerDubber

    video_path = REMOTE_IO_PATH / video_filename
    with open(video_path, "wb") as f:
        f.write(video_data)

    dubber = DockerDubber(base_dir=str(REMOTE_IO_PATH))
    dubber.run(str(video_path), target_lang=target_lang, qwen_mode=qwen_mode)

    out_path = (REMOTE_IO_PATH / "output" / "final_dub.mp4")
    if not out_path.exists():
        out_path = (REMOTE_IO_PATH / "output" / "final_dub.mp4")

    if not out_path.exists():
        raise FileNotFoundError(f"Expected output not found: {out_path}")

    out_bytes = out_path.read_bytes()

    try:
        video_path.unlink(missing_ok=True)
    except Exception:
        pass

    return out_bytes


@app.local_entrypoint()
def main(
    video: str = "impost_trimmed_2min.mp4",
    target_lang: str = "fr",
    qwen_mode: str = "custom",
    out: str = "final_dub.mp4",
):
    local_video_path = Path(video)
    if not local_video_path.exists():
        raise FileNotFoundError(f"Video not found: {local_video_path.resolve()}")

    with open(local_video_path, "rb") as f:
        video_data = f.read()

    out_bytes = dub_video_remote.remote(
        local_video_path.name,
        video_data,
        target_lang,
        qwen_mode,
    )

    out_path = Path(out)
    out_path.write_bytes(out_bytes)
