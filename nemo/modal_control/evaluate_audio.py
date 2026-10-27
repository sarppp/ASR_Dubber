import modal, json, io, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "speech_model_benchmark_2026-10-27"
app = modal.App("speech-benchmark-evaluation")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("onnx-asr[cpu,hub]", "soundfile", "numpy", "jiwer")
    .env(
        {
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "HF_HOME": "/cache/huggingface",
        }
    )
)
volume = modal.Volume.from_name("speech-model-benchmark-20261027")


@app.function(
    serialized=True,
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={"/cache": volume},
)
def evaluate(files, texts):
    print('MC_EVENT {"kind":"worker_ready"}', flush=True)
    import tempfile, soundfile as sf, onnx_asr, jiwer
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        "istupakov/parakeet-tdt-0.6b-v3-onnx",
        revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        allow_patterns=["*.int8.onnx", "nemo128.onnx", "vocab.txt", "config.json"],
    )
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", local, quantization="int8")

    def norm(s):
        return re.sub(r"[^\w\s]", " ", s.lower()).replace("_", " ").split()

    def n(s):
        return " ".join(norm(s))

    rows = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        for i, (data, ref) in enumerate(zip(files, texts)):
            p = Path(td) / f"{i}.wav"
            wav, sr = sf.read(io.BytesIO(data))
            sf.write(p, wav, sr)
            hyp = model.recognize(str(p))
            r = jiwer.process_words(n(ref), n(hyp))
            rows.append(
                {
                    "index": i,
                    "expected": ref,
                    "recognized": hyp,
                    "wer": r.wer,
                    "substitutions": r.substitutions,
                    "deletions": r.deletions,
                    "insertions": r.insertions,
                }
            )
            print("evaluated", i + 1, len(files), flush=True)
    score = jiwer.process_words(
        [n(s) for s in texts], [n(r["recognized"]) for r in rows]
    )
    return {
        "status": "complete",
        "metric": "ASR round-trip intelligibility proxy, not human accuracy",
        "evaluator": "Parakeet v3 ONNX INT8",
        "reference": "common German script supplied to TTS",
        "wer": score.wer,
        "cer": jiwer.cer([n(s) for s in texts], [n(r["recognized"]) for r in rows]),
        "substitutions": score.substitutions,
        "deletions": score.deletions,
        "insertions": score.insertions,
        "evaluation_seconds": time.perf_counter() - start,
        "segments": rows,
    }


@app.local_entrypoint()
def main(key: str, smoke: bool = False):
    import subprocess, tempfile

    d = ROOT / "runs" / key / "tts"
    segs = json.loads((d / "segments.json").read_text())
    data = []
    if smoke:
        segs = segs[:1]
    with tempfile.TemporaryDirectory() as td:
        for i, s in enumerate(segs):
            p = Path(td) / f"{i}.flac"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(d / s["path"]),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(p),
                ],
                check=True,
            )
            data.append(p.read_bytes())
    result = evaluate.remote(data, [s["text"] for s in segs])
    (d / ("quality_smoke.json" if smoke else "quality_proxy.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(key, "WER_PROXY", result["wer"])
