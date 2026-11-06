"""Parakeet Ultra support (moondream/parakeet-ultra).

Ultra is a transformers/kestrel model, not a NeMo one, so it cannot be loaded
by the NeMo code path. It needs torch >= 2.8; `.venv-local-asr` now runs torch
2.8.0+cu128, so in-process loading is the normal path.

Two execution paths, chosen automatically:

  1. in-process, when `moondream` is importable (the installed environment);
  2. otherwise, by running the transcription inside the local Docker image
     (`ultra-asr:local`, built from ultra_trial/Dockerfile). Still fully local,
     no network, no API.

Either way the model returns word-level timestamps, so the existing
`_words_to_segs` cutter produces subtitle-sized cues.

Photon posts usage telemetry to api.moondream.ai; `_offline_moondream()` stubs
the reporter so the pipeline never contacts a vendor endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("parakeet_ultra")

ULTRA_ALIASES = {"parakeet-ultra", "parakeet_ultra", "moondream/parakeet-ultra"}
ULTRA_MODEL_ID = "moondream/parakeet-ultra"
DOCKER_IMAGE = os.getenv("ULTRA_DOCKER_IMAGE", "ultra-asr:local")
HF_VOLUME = os.getenv("ULTRA_HF_VOLUME", "ultra_hf")


def _is_parakeet_ultra(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "parakeet-ultra" in name or "parakeet_ultra" in name


def _moondream_available() -> bool:
    try:
        _offline_moondream()
        import moondream  # noqa: F401
        return True
    except Exception:
        return False


def _offline_moondream() -> None:
    """Keep Photon off the network.

    moondream/kestrel posts usage telemetry to api.moondream.ai; the base URL is
    whitelisted so an env override is ignored. Inference is fully local and the
    weights come from the local HF cache, so the upload is disabled after import
    by stubbing the reporter's flush loop and startup validation.
    """
    try:
        from kestrel import photon as _photon
    except Exception:  # moondream not installed: Docker path handles it
        return

    async def _no_op_loop(self) -> None:
        return None

    async def _no_op_flush(self, *, rotate: bool = True):
        return None

    async def _no_op_validate(self) -> bool:
        return False

    reporter = _photon.PhotonReporter
    reporter._telemetry_loop = _no_op_loop
    reporter._flush_window = _no_op_flush
    reporter.validate_api_key = _no_op_validate



class UltraDockerModel:
    """Transcribes through the local ultra-asr Docker image."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        if shutil.which("docker") is None:
            raise RuntimeError(
                "parakeet-ultra needs either the `moondream` package or Docker. "
                "Build the image with: docker build -t ultra-asr:local "
                "nemo/new_guys/crispr/ultra_trial"
            )

    def transcribe(self, audio_path: str, timestamps: str = "word") -> dict:
        module_dir = Path(__file__).resolve().parent
        audio_dir = Path(audio_path).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asr.json"
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "--name", f"ultra-{os.getpid()}",
                "-v", f"{audio_dir}:/audio:ro",
                "-v", f"{tmp}:/out",
                "-v", f"{HF_VOLUME}:/models/hf",
                "-v", f"{module_dir}:/app:ro",
                "-w", "/app",
                "-e", "PYTHONPATH=/app",
                DOCKER_IMAGE,
                "python", "-m", "parakeet_ultra",
                "--audio", f"/audio/{Path(audio_path).name}",
                "--out", "/out/asr.json",
            ]
            log.info("parakeet-ultra: running in Docker (%s)", DOCKER_IMAGE)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not out.exists():
                raise RuntimeError(
                    "parakeet-ultra container failed: "
                    f"{(result.stderr or result.stdout)[-600:]}"
                )
            return json.loads(out.read_text(encoding="utf-8"))


class UltraInProcessModel:
    """Keeps one Photon session open for repeated chunk calls."""

    def __init__(self, model_id: str):
        _offline_moondream()
        import moondream as md

        # Callers may use the pipeline alias; Photon needs the full HF id.
        model_id = ULTRA_MODEL_ID if _is_parakeet_ultra(model_id) else model_id
        self.model_id = model_id
        runtime = {}
        if os.getenv("PARAKEET_ULTRA_MODEL_PATH"):
            runtime["model_path"] = os.environ["PARAKEET_ULTRA_MODEL_PATH"]
        self._ctx = md.photon(model_id, **runtime)
        self._speech = self._ctx.__enter__()

    def transcribe(self, audio_path: str, timestamps: str = "word") -> dict:
        return self._speech.transcribe(audio=str(audio_path), timestamps=timestamps)

    def close(self) -> None:
        try:
            if self._ctx is not None:
                self._ctx.__exit__(None, None, None)
        finally:
            self._ctx = self._speech = None


def _load_parakeet_ultra(model_name: str, device: str = "cuda", precision: str = "fp32"):
    if _moondream_available():
        log.info("parakeet-ultra: loading in-process (%s)", model_name)
        return UltraInProcessModel(model_name)
    log.info("parakeet-ultra: moondream not installed here, using Docker")
    return UltraDockerModel(model_name)


def _transcribe_parakeet_ultra(model, audio_path: str, offset: float = 0.0):
    """Return (words, segs) with timestamps shifted by `offset` seconds."""
    from nemo_audio import _words_to_segs

    result = model.transcribe(audio_path)
    words: list[dict] = []
    for seg in result.get("segments") or []:
        seg_words = seg.get("words") or []
        if seg_words:
            for word in seg_words:
                text = (word.get("word") or "").strip()
                if not text:
                    continue
                words.append({
                    "word": text,
                    "start": round(float(word.get("start", 0.0)) + offset, 3),
                    "end": round(float(word.get("end", 0.0)) + offset, 3),
                })
        else:
            # Segment without word timings: spread evenly so nothing is lost.
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            parts = text.split()
            start = float(seg.get("start", 0.0)) + offset
            end = float(seg.get("end", start)) + offset
            step = (end - start) / max(1, len(parts))
            for i, part in enumerate(parts):
                words.append({"word": part,
                              "start": round(start + i * step, 3),
                              "end": round(start + (i + 1) * step, 3)})

    if not words:
        return [], []

    words.sort(key=lambda w: w["start"])
    segs = _words_to_segs(words, max_w=10, max_dur=5.0, max_ch=80)
    return words, segs


def _unload_parakeet_ultra(model) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


# ── Standalone entry point (used inside the container: python -m parakeet_ultra)

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe with Parakeet Ultra")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=ULTRA_MODEL_ID)
    args = parser.parse_args()

    _offline_moondream()
    import moondream as md

    with md.photon(args.model) as speech:
        result = speech.transcribe(audio=args.audio, timestamps="word")
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _main()
