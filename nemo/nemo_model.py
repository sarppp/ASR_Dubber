"""
nemo_model.py — NeMo model loading and transcription.

No top-level nemo.collections imports — _import_nemo_asr() strips the script
directory from sys.path at call time so the real `nemo` package is found
instead of the local nemo.py file.
"""

import gc
import importlib
import logging
import shutil
import sys
import time
from pathlib import Path

import torch

from nemo_audio import (
    CHUNK_OVERLAP_SEC,
    _audio_duration,
    _chunk_audio,
    _cleanup_chunks,
    _fmt_dur,
    _strip_asr_repetition,
    _strip_special_tokens,
    _vram_gb,
)

log = logging.getLogger("nemo_local")


# ── Import helpers ────────────────────────────────────────────────────────────

def _import_nemo_asr():
    """Load nemo.collections.asr even if this script is named nemo.py."""
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)

    def _is_script_dir(entry: str | None) -> bool:
        if entry is None:
            return False
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            return False
        return resolved == script_dir

    try:
        sys.path = [entry for entry in original_path if not _is_script_dir(entry)]
        nemo_asr = importlib.import_module("nemo.collections.asr")
    finally:
        sys.path = original_path

    return nemo_asr


# ── Hypothesis helpers ────────────────────────────────────────────────────────

def _hyp_timestamps(hyp) -> dict | None:
    if hyp is None:
        return None
    for attr in ("timestamp", "timestep", "timestamps"):
        if isinstance(hyp, dict):
            ts = hyp.get(attr)
        else:
            ts = getattr(hyp, attr, None)
        if ts:
            return ts
    return None

def _hyp_field(hyp, attr: str, default=None):
    if hyp is None:
        return default
    if isinstance(hyp, dict):
        return hyp.get(attr, default)
    return getattr(hyp, attr, default)

def _looks_like_hyp(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return any(k in obj for k in ("text", "words", "timestamp", "timestep", "timestamps"))
    return any(hasattr(obj, attr) for attr in ("text", "words", "timestamp", "timestep", "timestamps"))

def _extract_first_hypothesis(batch_output):
    if batch_output is None:
        return None
    if _looks_like_hyp(batch_output):
        return batch_output
    if isinstance(batch_output, (list, tuple)):
        for item in batch_output:
            hyp = _extract_first_hypothesis(item)
            if hyp is not None:
                return hyp
    return None


# ── Model loading ─────────────────────────────────────────────────────────────

def _clear_nemo_cache(model_name: str) -> bool:
    """Delete the NeMo HF-hub cache for model_name so it re-downloads cleanly."""
    base = Path("/root/.cache/torch/NeMo")
    if not base.exists():
        return False
    # model_name is e.g. "nvidia/canary-qwen-2.5b" → org="nvidia", slug="canary-qwen-2.5b"
    parts = model_name.split("/")
    if len(parts) == 2:
        org, slug = parts
    else:
        org, slug = "", parts[-1]
    cleared = False
    for hf_dir in base.glob("*/hf_hub_cache"):
        target = hf_dir / org / slug if org else hf_dir / slug
        if target.exists():
            log.warning(f"Clearing corrupt NeMo cache: {target}")
            shutil.rmtree(target)
            cleared = True
    return cleared


def _from_pretrained_with_cache_retry(nemo_asr, model_name: str, device: str):
    """Call ASRModel.from_pretrained, clearing corrupt cache and retrying once on FileNotFoundError."""
    map_loc = None if device == "cuda" else "cpu"
    try:
        return nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_loc)
    except FileNotFoundError:
        if _clear_nemo_cache(model_name):
            log.info("Retrying model download after cache clear…")
            return nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_loc)
        raise


def _load_model(model_name: str, precision: str, device: str):
    from qwen3_asr import _is_qwen3_asr, _load_qwen3_asr
    if _is_qwen3_asr(model_name):
        return _load_qwen3_asr(model_name, device, precision)

    nemo_asr = _import_nemo_asr()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache(); gc.collect()
        free, total = _vram_gb()
        log.info(f"VRAM before load : {free:.2f}/{total:.2f} GB free")
        min_gb = 4.0 if precision == "fp32" else 2.5
        if free < min_gb:
            raise RuntimeError(f"Only {free:.2f} GB VRAM free — need {min_gb:.1f} GB for {precision}")

    t0 = time.perf_counter()
    log.info("Loading model…")
    try:
        model = _from_pretrained_with_cache_retry(nemo_asr, model_name, device)
    except Exception as e:
        if device != "cuda": raise
        log.warning(f"Direct GPU load failed ({e}); loading on CPU first")
        model = _from_pretrained_with_cache_retry(nemo_asr, model_name, "cpu")

    if device == "cuda":
        torch.cuda.empty_cache(); gc.collect()
        dtype = (torch.bfloat16 if precision == "bf16" and torch.cuda.is_bf16_supported()
                 else torch.float16 if precision == "fp16" else torch.float32)
        log.info(f"Moving to GPU [{dtype}] layer-by-layer…")
        for _, module in model.named_children():
            module.to(dtype).to(device); torch.cuda.empty_cache()
        model = model.to(device)
        log.info(f"Precision : {dtype}")

    model.eval()
    torch.cuda.empty_cache(); gc.collect()
    load_sec = time.perf_counter() - t0
    free_after, _ = _vram_gb()
    log.info(f"Model loaded {load_sec:.1f} s | VRAM used {free - free_after:.2f} GB | free {free_after:.2f} GB")

    if device == "cuda" and free_after > 1.0:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("torch.compile(reduce-overhead) active — first chunk warms up")
        except Exception: pass

    return model


# ── Transcription: Parakeet ───────────────────────────────────────────────────

def _transcribe_parakeet(model, audio_path: str, offset: float) -> tuple[list, list]:
    with torch.inference_mode():
        out = model.transcribe([audio_path], batch_size=1, timestamps=True)

    hyp = _extract_first_hypothesis(out)
    if hyp is None:
        log.error("Parakeet returned no recognizable hypothesis")
        return [], []

    text = _hyp_field(hyp, "text", "")
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)

    all_words = []
    ts = _hyp_timestamps(hyp)
    words_str = _hyp_field(hyp, "words", []) or []

    if ts and isinstance(ts, dict) and "word" in ts:
        prev_cum = ""
        for i, td in enumerate(ts.get("word") or []):
            if not isinstance(td, dict):
                continue
            s = td.get("start", 0.0)
            e = td.get("end", 0.0)
            w = td.get("word", None)
            if not w and i < len(words_str):
                w = words_str[i]
            w = str(w or "").strip()
            if w:
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

    if not all_words and ts and isinstance(ts, dict):
        log.warning("No 'word' key in Parakeet timestamps; scanning all keys")
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

    if not all_words and ts and isinstance(ts, dict) and "segment" in ts:
        segs_out = []
        for seg in ts["segment"]:
            segs_out.append({
                "text": str(seg.get("segment", "")),
                "start": float(seg.get("start", 0.0)) + offset,
                "end": float(seg.get("end", 0.0)) + offset,
            })
        return [], segs_out

    return all_words, []


# ── Transcription: Canary ─────────────────────────────────────────────────────

def _transcribe_canary(model, audio_path: str, offset: float, src_lang: str, tgt_lang: str) -> tuple[list, list]:
    from canary_patch import patch_canary2_eos_assert, patch_manifest_lang, build_transcription_config

    patch_canary2_eos_assert()
    patch_manifest_lang(src_lang, tgt_lang)
    cfg = build_transcription_config(src_lang, tgt_lang)

    with torch.inference_mode():
        try:
            out = (model.transcribe([audio_path], override_config=cfg) if cfg
                   else model.transcribe([audio_path], batch_size=1,
                                         source_lang=src_lang, target_lang=tgt_lang))
        except Exception as exc:
            log.error(f"Canary transcribe failed: {exc}"); raise

    # NeMo 2.1+ returns list[str] directly
    text = ""
    if out:
        first = out[0]
        if isinstance(first, str):
            text = first
        else:
            for attr in ("text", "pred_text", "transcription"):
                v = first.get(attr) if isinstance(first, dict) else getattr(first, attr, None)
                if v and isinstance(v, str): text = v; break
    text = _strip_special_tokens(text)
    text = _strip_asr_repetition(text)
    log.info(f"Canary output: {len(text)} chars | {text[:80]!r}")

    audio_dur = _audio_duration(audio_path)
    seg = {"text": text, "start": offset,
           "end": offset + (audio_dur if audio_dur > 0 else max(1.0, len(text.split()) * 0.4))}
    return [], [seg]


# ── Chunked transcription with OOM retry ─────────────────────────────────────

def _transcribe_chunked(model, audio_path: str, model_name: str,
                         src_lang: str, tgt_lang: str, chunk_sec: int):
    from qwen3_asr import _is_qwen3_asr, _transcribe_qwen3_asr
    is_canary  = "canary" in model_name.lower()
    is_qwen3   = _is_qwen3_asr(model_name)
    work_dir = Path(audio_path).parent
    dur = _audio_duration(audio_path)

    while chunk_sec >= 30:
        manifest = []
        try:
            if dur <= chunk_sec:
                manifest = [{"path": audio_path, "offset": 0.0}]
                log.info(f"Single-pass — full {_fmt_dur(dur)} fits in one chunk")
            else:
                raw = _chunk_audio(audio_path, work_dir, chunk_sec)
                manifest = [{"path": p, "offset": off} for p, off in raw]
                log.info(f"Chunk size {_fmt_dur(chunk_sec)} → {len(manifest)} chunk(s)")

            all_words, all_segs = [], []
            for ci, entry in enumerate(manifest):
                path, offset = entry["path"], entry["offset"]
                t1 = time.perf_counter()
                if is_qwen3:
                    words, segs = _transcribe_qwen3_asr(model, path, offset, src_lang)
                elif is_canary:
                    words, segs = _transcribe_canary(model, path, offset, src_lang, tgt_lang)
                else:
                    words, segs = _transcribe_parakeet(model, path, offset)
                elapsed = time.perf_counter() - t1
                if len(manifest) > 1:
                    free, _ = _vram_gb()
                    log.info(f"  chunk {ci+1}/{len(manifest)}: {elapsed:.1f} s | VRAM free {free:.2f} GB")
                all_words.extend(words); all_segs.extend(segs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); gc.collect()

            # Dedup overlapping words from chunk overlap
            if len(manifest) > 1 and all_words:
                out, prev = [all_words[0]], all_words[0]
                for w in all_words[1:]:
                    if not (w["start"] < prev["end"] - 0.05 and w["word"] == prev["word"]):
                        out.append(w); prev = w
                all_words = out

            return all_words, all_segs, manifest

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                _cleanup_chunks(manifest, audio_path); raise
            torch.cuda.empty_cache(); gc.collect()
            old = chunk_sec; chunk_sec = max(30, chunk_sec // 2)
            log.warning(f"OOM at chunk={_fmt_dur(old)} → retrying with {_fmt_dur(chunk_sec)}")
            _cleanup_chunks(manifest, audio_path)

    raise RuntimeError("Could not fit even 30s chunks in VRAM")


def _estimate_chunk_sec(model_name: str, safety: float, reserve_gb: float) -> int:
    free, _ = _vram_gb()
    if free <= 0: return 300
    usable = max(0.0, free - reserve_gb) * safety
    if usable <= 0: return 60

    from qwen3_asr import _is_qwen3_asr
    is_canary   = "canary"   in model_name.lower()
    is_parakeet = "parakeet" in model_name.lower()
    is_qwen3    = _is_qwen3_asr(model_name)
    gb_per_min  = 0.28 if is_parakeet else 0.50

    if is_canary:
        # Encoder-decoder trained on ≤40s segments: quality collapses above 60s.
        secs = 60
    elif is_qwen3:
        # LLM-based encoder-decoder: handles longer context than canary.
        # Cap at 120s; OOM retry halves if needed.
        secs = 120
    else:
        # CTC/TDT models (Parakeet v2/v3): quality unaffected by chunk length.
        secs = max(30, min(int(usable / gb_per_min * 60), 3600))

    log.info(f"VRAM {free:.2f} GB free → usable {usable:.2f} GB → chunk target {_fmt_dur(secs)}")
    return secs
