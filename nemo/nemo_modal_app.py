'''
NeMo ASR Transcription Pipeline on Modal (v2 — local-parity engine)
=====================================================================

Identical transcription quality to nemo3.py (local runner).
Uses the same adaptive VRAM chunking, OOM-retry, canary token merging,
cumulative-word handling, CUDA-graph disabling, and layer-by-layer model loading.

Auto-selects the best NeMo model based on language:
  en         -> nvidia/parakeet-tdt-0.6b-v2  (best English accuracy)
  fr/de/es   -> nvidia/canary-1b-v2          (multilingual)

# English (auto-selects parakeet)
modal run nemo_modal_app_v2.py --language en

# French / German / Spanish (auto-selects canary)
modal run nemo_modal_app_v2.py --language fr
modal run nemo_modal_app_v2.py --language de
modal run nemo_modal_app_v2.py --language es

# With speaker diarization (who said what)
modal run nemo_modal_app_v2.py --language en --diarize

# Translate to English (canary only)
modal run nemo_modal_app_v2.py --language de --translate

# Trim to first N seconds
modal run nemo_modal_app_v2.py --language en --trim 300

# Specify video file
modal run nemo_modal_app_v2.py --video-filename momo.mp4 --language en

# Override NeMo model manually
modal run nemo_modal_app_v2.py --language en --nemo-model nvidia/canary-1b-v2

# Precision (bf16 default)
modal run nemo_modal_app_v2.py --language en --precision fp32

# Force chunk size (canary models only)
modal run nemo_modal_app_v2.py --language de --chunk-override 720

uv run --env-file .env modal run nemo_modal_app.py --language de

GPU Pricing (Modal):
  Nvidia T4    ~$0.000164 / sec
  Nvidia A10G  ~$0.000306 / sec
'''

import modal
from pathlib import Path
import os
import sys
import gc
import time
import wave
import subprocess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOTE_IO_PATH = Path("/app/nemo_asr")
NEMO_CONDA_ENV = "nemo-env"
CONDA_PYTHON = f"/opt/conda/envs/{NEMO_CONDA_ENV}/bin/python"
GPU_TYPE = "T4"  # Change to "A10G" for long videos

NEMO_MODEL_EN = "nvidia/parakeet-tdt-0.6b-v2"
NEMO_MODEL_MULTI = "nvidia/canary-1b-v2"
NEMO_MULTI_LANGS = {"fr", "de", "es"}
DEFAULT_NEMO_MODEL = NEMO_MODEL_EN

CHUNK_OVERLAP_SEC = 2


def select_nemo_model(language: str, user_model: str = None) -> str:
    if user_model and user_model != DEFAULT_NEMO_MODEL:
        return user_model
    return NEMO_MODEL_MULTI if language in NEMO_MULTI_LANGS else NEMO_MODEL_EN


# ---------------------------------------------------------------------------
# Modal image (same proven conda setup as original)
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "git", "curl", "ca-certificates", "bash")
    .env({
        "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/opt/conda/envs/nemo-env/lib/python3.12/site-packages",
    })
    .run_commands(
        "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -o /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm -f /tmp/miniconda.sh",
    )
    .run_commands(
        "bash -lc '/opt/conda/bin/conda create -n nemo-env python=3.12 -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install -U pip'",
        'bash -lc \'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install "numpy<2.0"\'',
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install soundfile librosa nemo_toolkit[asr]'",
    )
    .run_commands(
        "bash -lc 'ln -sf /opt/conda/envs/nemo-env/bin/python /usr/local/bin/python'",
        "bash -lc 'ln -sf /opt/conda/envs/nemo-env/bin/pip /usr/local/bin/pip'",
    )
)

app = modal.App(name="nemo-asr-transcriber-v2")

# ---------------------------------------------------------------------------
# Lazy imports for torch / NeMo (so local client doesn't need them)
# ---------------------------------------------------------------------------

torch = None
_torch_mp = None
nemo_asr = None
OmegaConf = None
ClusteringDiarizer = None


NEMO_ENV_SITE_PACKAGES = "/opt/conda/envs/nemo-env/lib/python3.12/site-packages"


def _ensure_remote_imports():
    global torch, _torch_mp, nemo_asr, OmegaConf, ClusteringDiarizer
    if torch is not None:
        return

    if NEMO_ENV_SITE_PACKAGES not in sys.path:
        sys.path.insert(0, NEMO_ENV_SITE_PACKAGES)

    import torch as _torch
    import torch.multiprocessing as _mp
    import nemo.collections.asr as _nemo_asr
    from omegaconf import OmegaConf as _OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer as _ClusteringDiarizer

    torch = _torch
    _torch_mp = _mp
    nemo_asr = _nemo_asr
    OmegaConf = _OmegaConf
    ClusteringDiarizer = _ClusteringDiarizer

    # PyTorch 2.6 weights_only fix (best-effort)
    try:
        _original_load = torch.load

        def _safe_load_wrapper(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _safe_load_wrapper
    except Exception:
        pass

    # Avoid /dev/shm bus errors in containers
    try:
        _torch_mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    # CUDA memory optimizations
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")


# ---------------------------------------------------------------------------
# Helpers — timestamp / SRT formatting
# ---------------------------------------------------------------------------

def _fmt_ts(s: float) -> str:
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def _fmt_dur(s: float) -> str:
    return f"{int(s // 60)}m{int(s % 60):02d}s" if s >= 60 else f"{s:.1f}s"


# ---------------------------------------------------------------------------
# Segment builders (from nemo3.py — fixes the skip-then-restart bug)
# ---------------------------------------------------------------------------

def _words_to_segments(words, max_words=10, max_dur=5.0, max_chars=80):
    segs, cw, ct, cs = [], [], "", None
    for w in words:
        word = w.get("word", "").strip()
        if not word:
            continue
        ws, we = w.get("start", 0.0), w.get("end", 0.0)
        if cs is None:
            cs = ws
        cand = (ct + " " + word).strip() if ct else word
        split = (
            len(cw) >= max_words
            or (we - cs) > max_dur
            or len(cand) > max_chars
            or (ct and ct[-1] in ".!?" and len(cw) >= 3)
        )
        if split and cw:
            segs.append({"start": cs, "end": cw[-1].get("end", cs), "text": ct})
            cw, ct, cs = [], "", ws
            continue
        cw.append(w)
        ct = cand
    if cw and ct.strip():
        segs.append({"start": cs, "end": cw[-1].get("end", cs), "text": ct})
    return segs


def _words_to_segments_diarized(words, max_words=10, max_dur=5.0, max_chars=80):
    segs, cw, ct, cs, cspk = [], [], "", None, None
    for w in words:
        word = w.get("word", "").strip()
        if not word:
            continue
        ws, we, spk = w.get("start", 0.0), w.get("end", 0.0), w.get("speaker", "unknown")
        if cs is None:
            cs, cspk = ws, spk
        cand = (ct + " " + word).strip() if ct else word
        split = (
            len(cw) >= max_words
            or (we - cs) > max_dur
            or len(cand) > max_chars
            or (ct and ct[-1] in ".!?" and len(cw) >= 3)
            or (spk != cspk and cw)
        )
        if split and cw:
            segs.append({"start": cs, "end": cw[-1].get("end", cs), "text": ct, "speaker": cspk})
            cw, ct, cs, cspk = [], "", ws, spk
            continue
        cw.append(w)
        ct = cand
    if cw and ct.strip():
        segs.append({"start": cs, "end": cw[-1].get("end", cs), "text": ct, "speaker": cspk})
    return segs


def _segs_to_srt(segs):
    lines, idx, prev = [], 0, None
    for s in segs:
        t = s["text"].strip()
        if not t or t == prev:
            continue
        idx += 1
        lines += [str(idx), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}", t, ""]
        prev = t
    return "\n".join(lines)


def _segs_to_srt_diarized(segs):
    spk_list = sorted({s.get("speaker", "unknown") for s in segs})
    spk_map = {s: f"Speaker {i + 1}" for i, s in enumerate(spk_list)}
    lines, idx = [], 0
    for s in segs:
        t = s["text"].strip()
        if not t:
            continue
        idx += 1
        label = spk_map.get(s.get("speaker", "unknown"), "Speaker ?")
        lines += [str(idx), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}", f"[{label}] {t}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token merging (from nemo3.py — handles canary BPE / sentencepiece tokens
# and parakeet cumulative-text output)
# ---------------------------------------------------------------------------

def _merge_canary_tokens(words: list) -> list:
    if not words:
        return words
    has_spm = any("▁" in (w.get("word") or "") for w in words)
    has_g = any((w.get("word") or "").startswith("Ġ") for w in words)
    if not has_spm and not has_g:
        return words
    merged, cur = [], None
    for w in words:
        tok = str(w.get("word") or "")
        if not tok:
            continue
        starts_new = tok.startswith("▁") or tok.startswith("Ġ")
        piece = tok.lstrip("▁")
        if tok.startswith("Ġ"):
            piece = tok[1:]
        if starts_new or cur is None:
            if cur is not None and cur.get("word"):
                merged.append(cur)
            cur = {
                "word": piece,
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", w.get("start", 0.0))),
            }
            if "speaker" in w:
                cur["speaker"] = w["speaker"]
        else:
            cur["word"] = cur.get("word", "") + piece
            cur["end"] = float(w.get("end", cur.get("end", 0.0)))
    if cur is not None and cur.get("word"):
        merged.append(cur)
    return merged


def _merge_cumulative_words(words: list) -> list:
    """Handle parakeet's cumulative text output (each token = sentence so far)."""
    if not words or len(words) < 2:
        return words
    merged, prev_text = [], ""
    for w in words:
        current_text = w.get("word", "").strip()
        if not current_text:
            continue
        if prev_text and current_text.startswith(prev_text):
            delta = current_text[len(prev_text):].strip()
            if delta:
                new_word = {"word": delta, "start": w.get("start", 0.0), "end": w.get("end", 0.0)}
                if "speaker" in w:
                    new_word["speaker"] = w["speaker"]
                merged.append(new_word)
        elif not prev_text:
            merged.append(w.copy())
        else:
            merged.append(w.copy())
        prev_text = current_text
    return merged


def _dedup_words(words: list) -> list:
    if not words:
        return words
    out = [words[0]]
    for w in words[1:]:
        p = out[-1]
        if w["start"] < p["end"] - 0.05 and w["word"] == p["word"]:
            continue
        if w["start"] < p["start"]:
            continue
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _audio_duration(path: str) -> float:
    import wave
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _extract_audio(video_path: str, audio_path: str, trim_sec: int = 0) -> None:
    cpu_count = os.cpu_count() or 4
    cmd = ["ffmpeg", "-y", "-threads", str(cpu_count), "-i", video_path]
    if trim_sec > 0:
        cmd += ["-t", str(trim_sec)]
    cmd += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-threads", str(cpu_count), audio_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _chunk_audio(audio_path: str, work_dir: Path, chunk_sec: int, overlap_sec: int = CHUNK_OVERLAP_SEC) -> list:
    cpu_count = os.cpu_count() or 4
    duration = _audio_duration(audio_path)
    if duration <= chunk_sec + 5:
        return [(audio_path, 0.0)]
    chunks, step, offset, idx = [], chunk_sec - overlap_sec, 0.0, 0
    while offset < duration:
        cp = str(work_dir / f"_chunk_{idx:04d}.wav")
        dur = min(chunk_sec, duration - offset)
        subprocess.run(
            ["ffmpeg", "-y", "-threads", str(cpu_count), "-ss", str(offset), "-i", audio_path,
             "-t", str(dur), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", cp],
            check=True, capture_output=True,
        )
        chunks.append((cp, offset))
        offset += step
        idx += 1
    return chunks


def _cleanup_chunks(manifest: list, original_audio: str) -> None:
    for entry in manifest or []:
        path = entry.get("path")
        if not path or path == original_audio:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _vram_gb() -> tuple:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    return free_b / 1024 ** 3, total_b / 1024 ** 3


def _compute_max_chunk_sec(model_name: str, safety_factor: float = 0.85, reserve_gb: float = 1.5) -> int:
    free_gb, _ = _vram_gb()
    if free_gb <= 0:
        return 300
    usable_gb = max(0.0, free_gb - reserve_gb) * safety_factor
    if usable_gb <= 0:
        return 60
    gb_per_minute = 0.28 if "parakeet" in model_name.lower() else 0.50
    max_minutes = usable_gb / gb_per_minute if gb_per_minute > 0 else 1.0
    secs = int(max_minutes * 60)
    return max(30, min(secs, 600))


def _calibrate_chunk_size(
    model, audio_path: str, model_name: str, language: str, target_lang: str,
    initial_guess_sec: int, reserve_gb: float, safety_factor: float,
) -> int:
    if not torch.cuda.is_available():
        return initial_guess_sec

    audio_dur = _audio_duration(audio_path)
    test_sec = max(60, min(int(audio_dur * 0.1), int(audio_dur)) if audio_dur > 0 else initial_guess_sec)

    work_dir = Path(audio_path).parent
    test_chunk = work_dir / "_calibration_chunk.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "0", "-i", audio_path, "-t", str(test_sec),
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(test_chunk)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        return initial_guess_sec

    try:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() if hasattr(torch.cuda, "memory_allocated") else 0
        _transcribe_manifest(model, [{"path": str(test_chunk), "offset": 0.0}], model_name, language, target_lang)
        peak_bytes = torch.cuda.max_memory_allocated() if hasattr(torch.cuda, "max_memory_allocated") else baseline
        delta_bytes = max(0, peak_bytes - baseline)
        vram_used_gb = delta_bytes / 1024 ** 3 if delta_bytes else 0.0
        gb_per_sec = vram_used_gb / test_sec if test_sec > 0 else 0.0
        if gb_per_sec < 0.001:
            return initial_guess_sec
        free_now, _ = _vram_gb()
        usable_gb = max(0.0, free_now - reserve_gb) * safety_factor
        projected_sec = int(usable_gb / gb_per_sec)
        return max(60, min(projected_sec, 600))
    finally:
        test_chunk.unlink(missing_ok=True)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model loading (layer-by-layer, from nemo3.py)
# ---------------------------------------------------------------------------

def _load_model(model_name: str, precision: str, device: str):
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        gc.collect()
        free_before, total_gb = _vram_gb()
        print(f"VRAM before load: {free_before:.2f}/{total_gb:.2f} GB free")
        min_vram_gb = 4.0 if precision == "fp32" else 2.5
        if free_before < min_vram_gb:
            raise RuntimeError(f"Insufficient VRAM: {free_before:.2f} GB free (need {min_vram_gb:.1f} GB)")

    t0 = time.perf_counter()
    print(f"Loading model: {model_name}…")

    map_loc = "cpu" if device == "cpu" else None
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_loc)
    except Exception as e:
        if device == "cuda":
            print(f"GPU load failed ({e}); retrying on CPU then moving to GPU")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
        else:
            raise

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        target_dtype = torch.float32
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            target_dtype = torch.bfloat16
            print("Moving to GPU with bfloat16 (layer-by-layer)…")
        elif precision == "fp16":
            target_dtype = torch.float16
            print("Moving to GPU with float16 (layer-by-layer)…")
        elif precision == "bf16":
            print("bfloat16 unsupported on this GPU — using fp32")
        else:
            print("Moving to GPU with float32…")

        try:
            for name, module in model.named_children():
                module.to(target_dtype).to(device)
                torch.cuda.empty_cache()
            model = model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"OOM during model transfer. Try --precision fp16.") from e
            raise
    else:
        print("Precision: float32 (CPU)")

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    load_sec = time.perf_counter() - t0

    if device == "cuda":
        free_after, _ = _vram_gb()
        used_gb = free_before - free_after
        print(f"Model loaded {load_sec:.1f}s | model VRAM {used_gb:.2f} GB | free VRAM {free_after:.2f} GB")

        free_gb, _ = _vram_gb()
        if free_gb > 1.0:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile(reduce-overhead) active — first chunk warms up")
            except Exception:
                pass
        else:
            print(f"Skipping torch.compile (only {free_gb:.2f} GB VRAM free)")
    else:
        print(f"Model loaded {load_sec:.1f}s (CPU mode)")

    return model


# ---------------------------------------------------------------------------
# Core transcription (from nemo3.py — CUDA graph fix, token merging, etc.)
# ---------------------------------------------------------------------------

def _transcribe_manifest(model, manifest: list, model_name: str, language: str, target_lang: str) -> tuple:
    # Disable CUDA graphs (avoids invalid getCurrentStream errors on some GPUs)
    if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
        for attr in ("use_cuda_graphs", "cuda_graphs", "use_cuda_graph"):
            if attr in model.cfg.decoding:
                model.cfg.decoding[attr] = False
    if hasattr(model, "decoding"):
        for attr in ("use_cuda_graphs", "cuda_graphs", "use_cuda_graph"):
            if hasattr(model.decoding, attr):
                setattr(model.decoding, attr, False)
        dc = getattr(model.decoding, "decoding_computer", None)
        if dc:
            for attr in ("use_cuda_graphs", "cuda_graphs", "use_cuda_graph"):
                if hasattr(dc, attr):
                    setattr(dc, attr, False)
            if hasattr(dc, "cuda_graphs_impl"):
                dc.cuda_graphs_impl = None

    all_words, all_segs, text_parts = [], [], []
    n = len(manifest)

    for ci, entry in enumerate(manifest):
        path, offset = entry["path"], entry["offset"]
        prev_cum = ""
        with torch.inference_mode():
            kw = {"timestamps": True, "batch_size": 1}
            if "canary" in model_name.lower():
                kw["source_lang"] = language
                kw["target_lang"] = target_lang
            out = model.transcribe([path], **kw)

        if n > 1:
            free_gb, _ = _vram_gb()
            print(f"  chunk {ci + 1}/{n} | VRAM free {free_gb:.2f} GB")

        if hasattr(out, "__len__") and len(out) > 0:
            hyp = out[0]
            text = hyp.text if hasattr(hyp, "text") else (hyp if isinstance(hyp, str) else "")
            text_parts.append(text or "")
            ts = getattr(hyp, "timestamp", None)
            words_str = getattr(hyp, "words", None) or []

            if ts and isinstance(ts, dict) and "word" in ts:
                word_items = ts.get("word") or []
                for i, td in enumerate(word_items):
                    if not isinstance(td, dict):
                        continue
                    s = td.get("start", 0.0)
                    e = td.get("end", 0.0)
                    w = td.get("word", None)
                    if not w and i < len(words_str):
                        w = words_str[i]
                    w = str(w or "").strip()
                    if ("canary" in model_name.lower() or "parakeet" in model_name.lower()) and w:
                        cum = w
                        if prev_cum and cum.startswith(prev_cum):
                            delta = cum[len(prev_cum):].strip()
                            if not delta:
                                prev_cum = cum
                                continue
                            w = delta
                        prev_cum = cum
                    if "\u00a0" in w:
                        w = w.replace("\u00a0", " ")
                    if " " in w:
                        w = w.split()[-1]
                    if not w:
                        continue
                    all_words.append({"word": w, "start": float(s) + offset, "end": float(e) + offset})

            if ts and isinstance(ts, dict) and "segment" in ts:
                for seg in ts["segment"]:
                    all_segs.append({
                        "text": str(seg.get("segment", "")),
                        "start": float(seg.get("start", 0.0)) + offset,
                        "end": float(seg.get("end", 0.0)) + offset,
                    })

            if not all_words and not all_segs and ts and isinstance(ts, dict):
                for key in ts:
                    items = ts[key]
                    if isinstance(items, list) and items and isinstance(items[0], dict):
                        for item in items:
                            w = (item.get("word") or item.get("char") or item.get("label") or item.get("segment") or "")
                            if w:
                                all_words.append({
                                    "word": str(w),
                                    "start": float(item.get("start", 0.0)) + offset,
                                    "end": float(item.get("end", 0.0)) + offset,
                                })
                        if all_words:
                            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    if "canary" in model_name.lower() and all_words:
        all_words = _merge_canary_tokens(all_words)
    elif "parakeet" in model_name.lower() and all_words:
        all_words = _merge_cumulative_words(all_words)

    return all_words, all_segs, text_parts


# ---------------------------------------------------------------------------
# OOM-retry transcription with adaptive chunk halving (from nemo3.py)
# ---------------------------------------------------------------------------

def _transcribe_with_retry(model, audio_path: str, offset: float, model_name: str,
                            language: str, target_lang: str, initial_chunk_sec: int):
    chunk_sec = max(30, initial_chunk_sec)
    work_dir = Path(audio_path).parent
    duration = _audio_duration(audio_path)

    while chunk_sec >= 30:
        manifest = []
        try:
            if duration <= chunk_sec:
                manifest = [{"path": audio_path, "offset": offset}]
                print(f"Single-pass — full {_fmt_dur(duration)}")
            else:
                raw = _chunk_audio(audio_path, work_dir, chunk_sec=chunk_sec)
                manifest = [{"path": cp, "offset": off} for cp, off in raw]
                print(f"Chunk size {_fmt_dur(chunk_sec)} → {len(manifest)} chunk(s)")
            words, segs, txt = _transcribe_manifest(model, manifest, model_name, language, target_lang)
            return words, segs, txt, manifest
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            old, chunk_sec = chunk_sec, max(30, chunk_sec // 2)
            print(f"OOM at chunk={_fmt_dur(old)} → retrying with {_fmt_dur(chunk_sec)}")
            _cleanup_chunks(manifest, audio_path)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                old, chunk_sec = chunk_sec, max(30, chunk_sec // 2)
                print(f"OOM (RuntimeError) → retrying with {_fmt_dur(chunk_sec)}")
                _cleanup_chunks(manifest, audio_path)
            else:
                _cleanup_chunks(manifest, audio_path)
                raise

    raise RuntimeError("Could not fit even 30s chunks in VRAM.")


# ---------------------------------------------------------------------------
# Diarization (inline NeMo ClusteringDiarizer, no subprocess)
# ---------------------------------------------------------------------------

def _run_diarization(audio_path: str, work_dir: Path) -> list:
    import json, shutil
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running speaker diarization…")

    ddir = work_dir / "_diarize"
    ddir.mkdir(parents=True, exist_ok=True)

    mpath = ddir / "manifest.json"
    mpath.write_text(
        json.dumps({
            "audio_filepath": str(Path(audio_path).resolve()),
            "offset": 0, "duration": None, "label": "infer",
            "text": "", "num_speakers": None, "rttm_filepath": "", "uem_filepath": "",
        }) + "\n",
        encoding="utf-8",
    )

    cfg = OmegaConf.create({
        "name": "ClusterDiarizer",
        "num_workers": 0, "sample_rate": 16000, "batch_size": 16,
        "device": device, "verbose": True,
        "diarizer": {
            "manifest_filepath": str(mpath),
            "out_dir": str(ddir),
            "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "window_length_in_sec": 0.63, "shift_length_in_sec": 0.01,
                    "smoothing": False, "overlap": 0.5, "onset": 0.9, "offset": 0.5,
                    "pad_onset": 0.0, "pad_offset": 0.0, "min_duration_on": 0.0,
                    "min_duration_off": 0.6, "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.0, 0.5],
                    "shift_length_in_sec": [0.75, 0.5, 0.25],
                    "multiscale_weights": [1, 1, 1], "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False, "max_num_speakers": 8,
                    "enhanced_count_thres": 80, "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30, "maj_vote_spk_count": False,
                    "chunk_cluster_count": 50, "embeddings_per_chunk": 10000,
                },
            },
        },
    })

    ClusteringDiarizer(cfg=cfg).to(device).diarize()

    rttm_files = list((ddir / "pred_rttms").glob("*.rttm"))
    if not rttm_files:
        rttm_files = list(ddir.rglob("*.rttm"))

    turns = []
    if rttm_files:
        with open(rttm_files[0], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
                    try:
                        start = float(parts[3])
                        dur = float(parts[4])
                        spk = str(parts[7])
                        turns.append({"speaker": spk, "start": start, "end": start + dur})
                    except (ValueError, IndexError) as e:
                        print(f"Skipping malformed RTTM line: {line.strip()} | {e}")

    turns.sort(key=lambda t: t["start"])
    print(f"Diarization done — {len({t['speaker'] for t in turns})} speaker(s), {len(turns)} turns")
    shutil.rmtree(ddir, ignore_errors=True)
    return turns


# ---------------------------------------------------------------------------
# Speaker assignment (words and segments)
# ---------------------------------------------------------------------------

def _assign_speakers(words: list, turns: list) -> list:
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        best_spk, best_ov = None, 0.0
        for t in turns:
            ov = max(0.0, min(w["end"], t["end"]) - max(w["start"], t["start"]))
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
            if best_spk is None and t["start"] <= mid <= t["end"]:
                best_spk = t["speaker"]
        w["speaker"] = best_spk or "unknown"
    return words


def _assign_speakers_to_segments(segments: list, turns: list) -> list:
    for seg in segments:
        seg_start, seg_end = seg.get("start", 0.0), seg.get("end", 0.0)
        best_spk, best_score = "unknown", 0.0
        for turn in turns:
            overlap = max(0.0, min(seg_end, turn["end"]) - max(seg_start, turn["start"]))
            seg_dur = seg_end - seg_start
            if seg_dur > 0:
                pct = overlap / seg_dur
                if pct > best_score:
                    best_score, best_spk = pct, turn["speaker"]
        seg["speaker"] = best_spk
    return segments


# ---------------------------------------------------------------------------
# Main pipeline (runs inside Modal container)
# ---------------------------------------------------------------------------

def _run_pipeline(
    video_path: str,
    language: str,
    nemo_model: str,
    precision: str,
    translate: bool,
    diarize: bool,
    trim_sec: int,
    safety_factor: float,
    reserve_gb: float,
    chunk_override_sec: int | None,
) -> str:
    _ensure_remote_imports()
    wall_t0 = time.perf_counter()
    work_dir = Path(video_path).parent
    stem = Path(video_path).stem
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_lang = "en" if translate else language

    trim_tag = f"trim{int(trim_sec)}" if trim_sec and trim_sec > 0 else "full"
    audio_path = str(work_dir / f"{stem}_nemo_16k_{trim_tag}.wav")

    print("Extracting 16 kHz mono WAV…")
    _extract_audio(video_path, audio_path, trim_sec=trim_sec)

    audio_dur = _audio_duration(audio_path)
    print(f"Audio ready | duration {_fmt_dur(audio_dur)}")

    print(f"Loading model: {nemo_model} [{device}]")
    model = _load_model(nemo_model, precision, device)

    # Determine chunk size
    forced_chunk = None
    if chunk_override_sec and chunk_override_sec >= 30:
        if "canary" in nemo_model.lower():
            forced_chunk = max(30, min(int(chunk_override_sec), 900))
            print(f"Manual chunk override: {_fmt_dur(forced_chunk)}")
        else:
            print("Chunk override ignored — canary models only")

    if forced_chunk:
        optimal_chunk = forced_chunk
    elif device == "cuda":
        initial_chunk = _compute_max_chunk_sec(nemo_model, safety_factor, reserve_gb)
        print(f"VRAM-estimated chunk: {_fmt_dur(initial_chunk)}")
        if audio_dur > initial_chunk * 1.5:
            try:
                optimal_chunk = _calibrate_chunk_size(
                    model, audio_path, nemo_model, language, target_lang,
                    initial_chunk, reserve_gb, safety_factor,
                )
                print(f"Calibrated chunk: {_fmt_dur(optimal_chunk)}")
            except Exception as exc:
                print(f"Calibration failed ({exc}); using {_fmt_dur(initial_chunk)}")
                optimal_chunk = initial_chunk
        else:
            optimal_chunk = initial_chunk
    else:
        optimal_chunk = 300

    print(f"Transcribing | chunk target: {_fmt_dur(optimal_chunk)}")
    t_asr = time.perf_counter()
    words, segs, _, manifest = _transcribe_with_retry(
        model, audio_path, 0.0, nemo_model, language, target_lang, optimal_chunk,
    )
    _cleanup_chunks(manifest, audio_path)
    asr_elapsed = time.perf_counter() - t_asr
    rtf = asr_elapsed / audio_dur if audio_dur > 0 else 0
    print(f"Transcription done {asr_elapsed:.1f}s (RTF {rtf:.2f}x)")

    if len(manifest) > 1 and words:
        words = _dedup_words(words)
    if "canary" in nemo_model.lower() and segs:
        words = []
    if not words and not segs:
        raise RuntimeError("NeMo returned no timestamps.")

    print(f"Got {len(words) if words else len(segs)} {'word' if words else 'segment'} timestamps")

    if diarize:
        turns = _run_diarization(audio_path, work_dir)
        if words:
            words = _assign_speakers(words, turns)
            final_segs = _words_to_segments_diarized(words)
        elif segs:
            final_segs = _assign_speakers_to_segments(segs, turns)
        else:
            final_segs = []
        srt = _segs_to_srt_diarized(final_segs)
    else:
        final_segs = _words_to_segments(words) if words else segs
        srt = _segs_to_srt(final_segs)

    wall_elapsed = time.perf_counter() - wall_t0
    print(
        f"{'=' * 55}\n"
        f"  Total wall time   : {_fmt_dur(wall_elapsed)}\n"
        f"  Audio duration    : {_fmt_dur(audio_dur)}\n"
        f"  ASR time          : {_fmt_dur(asr_elapsed)}\n"
        f"  Real-time factor  : {rtf:.2f}x\n"
        f"  Subtitle segments : {len(final_segs)}\n"
        f"{'=' * 55}"
    )

    # Cleanup audio
    try:
        Path(audio_path).unlink(missing_ok=True)
    except Exception:
        pass

    return srt


# ---------------------------------------------------------------------------
# Modal remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
)
def transcribe_nemo_remote(
    video_filename: str,
    video_data: bytes,
    language: str = "en",
    trim_sec: int = 0,
    nemo_model: str = DEFAULT_NEMO_MODEL,
    precision: str = "bf16",
    translate: bool = False,
    diarize: bool = False,
    safety_factor: float = 0.85,
    reserve_gb: float = 1.5,
    chunk_override_sec: int | None = None,
) -> bytes:
    _ensure_remote_imports()
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
        ).strip().splitlines()
        if gpu_name:
            print(f"GPU: {gpu_name[0]}")
    except Exception:
        pass

    REMOTE_IO_PATH.mkdir(parents=True, exist_ok=True)
    video_path = REMOTE_IO_PATH / video_filename

    with open(video_path, "wb") as f:
        f.write(video_data)

    srt_content = _run_pipeline(
        video_path=str(video_path),
        language=language,
        nemo_model=nemo_model,
        precision=precision,
        translate=translate,
        diarize=diarize,
        trim_sec=trim_sec,
        safety_factor=safety_factor,
        reserve_gb=reserve_gb,
        chunk_override_sec=chunk_override_sec,
    )

    if video_path.exists():
        video_path.unlink()

    return srt_content.encode("utf-8")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    video_filename: str = None,
    language: str = "en",
    trim: int = 0,
    nemo_model: str = DEFAULT_NEMO_MODEL,
    precision: str = "bf16",
    translate: bool = False,
    diarize: bool = False,
    safety_factor: float = 0.85,
    reserve_gb: float = 1.5,
    chunk_override: int = None,
):
    """
    Transcribe or translate video using NeMo ASR (local-parity engine).

    Languages (auto-selects best model):
        en           -> nvidia/parakeet-tdt-0.6b-v2
        fr / de / es -> nvidia/canary-1b-v2

    Examples:
        modal run nemo_modal_app_v2.py --language en --diarize
        modal run nemo_modal_app_v2.py --language de --translate
        modal run nemo_modal_app_v2.py --language en --trim 300 --precision fp16
        modal run nemo_modal_app_v2.py --language de --chunk-override 720
    """
    import time as _time

    local_io_path = Path(".")
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}

    if translate and language == "en":
        print("⚠️  --translate with --language en makes no sense. Ignoring --translate.")
        translate = False
    if translate:
        resolved_model = NEMO_MODEL_MULTI
    else:
        resolved_model = select_nemo_model(language, nemo_model)

    if video_filename is None:
        found = [f for f in local_io_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT]
        if not found:
            print("❌ No video files found in current directory")
            return

        if diarize:
            srt_suffix = f".nemo.{language}.diarize.srt"
        elif translate:
            srt_suffix = ".nemo.en.srt"
        else:
            srt_suffix = f".nemo.{language}.srt"

        pending = [v for v in found if not (v.parent / (v.stem + srt_suffix)).exists()]
        if not pending:
            print("✅ All videos already have NeMo SRT files!")
            return
        local_video_path = pending[0]
        print(f"🎯 Auto-detected video: {local_video_path.name}")
    else:
        local_video_path = local_io_path / video_filename
        if not local_video_path.exists():
            print(f"❌ Video not found: {local_video_path.absolute()}")
            return

    task_label = "Translation" if translate else "Transcription"
    print(f"\n{'=' * 60}")
    print(f"🎙️  NeMo ASR {task_label} Pipeline (v2 — local-parity)")
    print(f"{'=' * 60}")
    print(f"📹 Video     : {local_video_path.absolute()}")
    print(f"🧠 Model     : {resolved_model}")
    print(f"🌍 Language  : {language}" + (" → English" if translate else ""))
    print(f"🎯 Precision : {precision}")
    print(f"🗣️  Diarize   : {'✅' if diarize else '❌'}")
    print(f"✂️  Trim      : {_fmt_dur(trim) if trim > 0 else 'full video'}")
    print(f"⚙️  Safety    : {safety_factor:.0%}  | Reserve: {reserve_gb:.1f} GB")
    if chunk_override:
        print(f"📦 Chunk override: {_fmt_dur(chunk_override)} (canary-only)")
    print(f"🖥️  GPU       : {GPU_TYPE}")
    print(f"{'=' * 60}\n")

    print("📤 Reading video file…")
    with open(local_video_path, "rb") as f:
        video_data = f.read()
    print(f"   Size: {len(video_data) / 1024 / 1024:.1f} MB")

    wall_t0 = time.time()
    print("🚀 Sending job to Modal…\n")

    srt_bytes = transcribe_nemo_remote.remote(
        video_filename=local_video_path.name,
        video_data=video_data,
        language=language,
        trim_sec=trim,
        nemo_model=resolved_model,
        precision=precision,
        translate=translate,
        diarize=diarize,
        safety_factor=safety_factor,
        reserve_gb=reserve_gb,
        chunk_override_sec=chunk_override,
    )

    wall_elapsed = time.time() - wall_t0

    if translate:
        srt_suffix = ".nemo.en.srt"
    elif diarize:
        srt_suffix = f".nemo.{language}.diarize.srt"
    else:
        srt_suffix = f".nemo.{language}.srt"

    srt_path = local_io_path / (local_video_path.stem + srt_suffix)
    srt_path.write_bytes(srt_bytes)

    print(f"\n{'=' * 60}")
    print(f"✅ {task_label} complete! (wall time: {_fmt_dur(wall_elapsed)})")
    print(f"📄 SRT saved: {srt_path.absolute()}")
    print(f"{'=' * 60}")

    srt_text = srt_bytes.decode("utf-8")
    lines = srt_text.split("\n")
    preview = lines[:min(16, len(lines))]
    print("\n📋 Preview (first segments):")
    print("-" * 40)
    for line in preview:
        print(f"  {line}")
    seg_count = len([l for l in lines if l.strip() and l.strip().isdigit()])
    if len(lines) > 16:
        print(f"  … ({seg_count} segments total)")