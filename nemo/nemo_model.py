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

# All known CUDA-graph config keys across NeMo model variants.
# Parakeet TDT actual keys (confirmed from config dump):
#   greedy.use_cuda_graph_decoder — main TDT decoder loop graph (~17 GB on A10G)
#   greedy.loop_labels            — label-loop CUDA graph variant
#   beam.allow_cuda_graphs        — beam decoder graphs
# Generic fallbacks for older/other NeMo model variants:
#   use_cuda_graphs / greedy.use_cuda_graphs
_CUDA_GRAPH_KEYS = (
    "use_cuda_graphs",
    "greedy.use_cuda_graphs",
    "greedy.use_cuda_graph_decoder",
    "greedy.loop_labels",
    "beam.allow_cuda_graphs",
)


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


# ── CUDA graph helpers ────────────────────────────────────────────────────────

def _patch_greedy_tdt_no_cuda_graphs() -> bool:
    """Patch GreedyBatchedTDTInfer at the CLASS level to disable CUDA graphs.

    Must be called BEFORE from_pretrained so even the first decoder construction
    (inside NeMo's model restore) already has CUDA graphs disabled.

    Why class-level and not instance-level:
      - NeMo rebuilds GreedyBatchedTDTInfer from frozen config at the start of
        every transcribe() call (rnnt_models.py:315).  Instance patches are wiped.
      - use_cuda_graph_decoder is consumed during __init__ to branch between
        creating a CUDA-graph sub-object vs a plain decoder; it is NOT stored as
        a plain attribute afterward, so post-construction patching has no effect.
    """
    _CANDIDATE_MODULES = (
        "nemo.collections.asr.parts.submodules.rnnt_greedy_decoding",
        "nemo.collections.asr.modules.rnnt_greedy_decoding",
        "nemo.collections.asr.parts.submodules.cuda_graph_rnnt_greedy_decoding",
    )
    _CLASS_NAMES = ("GreedyBatchedTDTInfer", "GreedyBatchedRNNTInfer")
    patched_any = False
    for mod_path in _CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        for cls_name in _CLASS_NAMES:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            _orig_init = cls.__init__

            def _make_patched_init(orig):
                def _patched_init(self, *args, **kwargs):
                    kwargs["use_cuda_graph_decoder"] = False
                    if "loop_labels" in kwargs:
                        kwargs["loop_labels"] = False
                    orig(self, *args, **kwargs)
                    log.debug(f"[GRAPH-PATCH] {type(self).__name__} created: "
                              f"use_cuda_graph_decoder={getattr(self, 'use_cuda_graph_decoder', '?')} "
                              f"loop_labels={getattr(self, 'loop_labels', '?')}")
                return _patched_init

            cls.__init__ = _make_patched_init(_orig_init)
            log.info(f"[GRAPH-PATCH] Patched {mod_path}.{cls_name}.__init__")
            patched_any = True
    if not patched_any:
        log.warning("[GRAPH-PATCH] Warning: could not find GreedyBatchedTDTInfer to patch")
    return patched_any


def _disable_cuda_graphs_in_decoder(model) -> None:
    """Belt-and-suspenders: disable CUDA graph flags on the live decoder object.

    Called after every change_decoding_strategy() (including NeMo's internal
    call at the start of each transcribe()).  The class-level __init__ patch is
    the primary fix; this is the fallback in case a decoder variant slips through.
    """
    _GRAPH_BOOL_ATTRS = (
        "use_cuda_graphs",
        "use_cuda_graph_decoder",
        "loop_labels",
        "allow_cuda_graphs",
    )

    dec = getattr(model, "decoding", None)
    if dec is None:
        return

    def _fix_obj(obj, label: str):
        changed = []
        for attr in _GRAPH_BOOL_ATTRS:
            if hasattr(obj, attr) and getattr(obj, attr) is True:
                setattr(obj, attr, False)
                changed.append(attr)
        if hasattr(obj, "cuda_graphs_impl") and getattr(obj, "cuda_graphs_impl") is not None:
            obj.cuda_graphs_impl = None
            changed.append("cuda_graphs_impl→None")
        if changed:
            log.info(f"[CDS-FIX] disabled in {label}: {changed}")

    _fix_obj(dec, "model.decoding")
    for dc_name in ("decoding", "decoding_computer", "_decoding_computer", "greedy_decoding"):
        inner = getattr(dec, dc_name, None)
        if inner is not None and inner is not dec:
            _fix_obj(inner, f"model.decoding.{dc_name}")


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
    """Call ASRModel.from_pretrained, clearing corrupt cache and retrying once on FileNotFoundError.

    NOTE: Models stored on HuggingFace in safetensors format (no .nemo archive)
    cannot be loaded by NeMo ≤2.1 — they lack the model_config.yaml that NeMo's
    restore path requires.  Retrying won't help; use parakeet-v3 or qwen3-asr instead.
    """
    map_loc = None if device == "cuda" else "cpu"
    try:
        return nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_loc)
    except FileNotFoundError as exc:
        missing = str(exc)
        if "model_config.yaml" in missing:
            raise RuntimeError(
                f"Cannot load '{model_name}': model_config.yaml not found in the downloaded "
                f"HuggingFace repo. This model is stored in safetensors format which NeMo ≤2.1 "
                f"cannot restore. Use parakeet-v3 (EN) or qwen3-asr (multilingual) instead, "
                f"or set NEMO_MODEL_EN=parakeet-v3 in docker-compose.yml."
            ) from exc
        # Generic FileNotFoundError → try clearing corrupt cache and re-downloading once
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

    # Patch GreedyBatchedTDTInfer BEFORE from_pretrained so the very first
    # decoder construction (inside NeMo's model restore) already has CUDA graphs
    # disabled.  Applies to parakeet v2 and v3 (both use TDT/RNNT decoders).
    # Canary uses an encoder-decoder — no GreedyBatchedTDTInfer involved.
    is_parakeet = "parakeet" in model_name.lower()
    if device == "cuda" and is_parakeet:
        _patch_greedy_tdt_no_cuda_graphs()

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
    if device == "cuda":
        log.info(f"Model loaded {load_sec:.1f} s | VRAM used {free - free_after:.2f} GB | free {free_after:.2f} GB")
    else:
        log.info(f"Model loaded {load_sec:.1f} s (CPU)")

    # Disable NeMo's CUDA graph capture for parakeet models.
    # GreedyBatchedTDTInfer captures a CUDA graph on the first transcribe() call.
    # After empty_cache() between chunks, the graph's backing pages are freed;
    # the next chunk replays the stale graph onto freed addresses →
    # XID 31 MMU fault → SIGABRT (exit code 134).
    # Primary fix: class-level __init__ patch above (_patch_greedy_tdt_no_cuda_graphs).
    # Belt-and-suspenders: also try OmegaConf config keys and monkey-patch
    # change_decoding_strategy so NeMo's internal re-invocations are intercepted.
    if device == "cuda" and is_parakeet and hasattr(model, "cfg"):
        try:
            from omegaconf import OmegaConf
            failed_keys = []
            for key_path in _CUDA_GRAPH_KEYS:
                try:
                    OmegaConf.update(model.cfg.decoding, key_path, False)
                except Exception as _ke:
                    failed_keys.append(key_path)
            if failed_keys:
                log.warning(f"[WARN] CUDA graph keys not writable (frozen config): {failed_keys}")
            if hasattr(model, "change_decoding_strategy"):
                model.change_decoding_strategy(model.cfg.decoding)
            log.info("NeMo decoder CUDA graphs disabled")
        except Exception as exc:
            log.warning(f"Could not disable NeMo CUDA graphs: {exc}")

    if device == "cuda" and is_parakeet and hasattr(model, "change_decoding_strategy"):
        _orig_cds = model.change_decoding_strategy

        def _patched_cds(cfg=None, **kw):
            result = _orig_cds(cfg, **kw) if cfg is not None else _orig_cds(**kw)
            _disable_cuda_graphs_in_decoder(model)
            return result

        model.change_decoding_strategy = _patched_cds
        _disable_cuda_graphs_in_decoder(model)

    # torch.compile(reduce-overhead) uses CUDA graph capture internally:
    #   1. Captures ~17 GB VRAM during warmup for large encoder models
    #   2. Crashes (XID 31 MMU fault) when replaying after empty_cache() between chunks
    # Variable-length chunked ASR inference is incompatible with reduce-overhead.
    log.info("Skipping torch.compile — incompatible with variable-length chunked ASR")

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
                    # Belt-and-suspenders: ensure CUDA graphs are off before every
                    # parakeet chunk (primary fix is the class-level __init__ patch).
                    _disable_cuda_graphs_in_decoder(model)
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
        # QUALITY cap — trained on ≤40s segments; decoder collapses above 60s.
        secs = 60
    elif is_qwen3:
        # LLM-based ASR: offline inference handles long context.  No quality
        # cap needed; let available VRAM drive the chunk size.
        gb_per_min = 0.35
        secs = int(usable / gb_per_min * 60)
    elif "parakeet" in model_name.lower() and "v3" in model_name.lower():
        # parakeet-v3 uses global self-attention (att_context_size=[-1,-1]):
        # memory is O(T²), not O(T).  On a 16 GB GPU the linear formula gives
        # ~37 min which OOMs even with CUDA graphs disabled.  Hard cap at 600 s
        # (10 min); OOM retry will halve further if needed.
        # parakeet-v2 uses local windowed attention — no cap needed there.
        secs = max(30, min(int(usable / gb_per_min * 60), 600))
    else:
        # CTC/TDT models (Parakeet v2): quality unaffected by chunk length.
        secs = int(usable / gb_per_min * 60)
    if not is_canary and "v3" not in model_name.lower():
        # No model-specific cap — free VRAM is the only constraint.
        # 7200s absolute ceiling; OOM retry halves if we ever overshoot.
        secs = max(30, min(secs, 7200))

    log.info(f"VRAM {free:.2f} GB free → usable {usable:.2f} GB → chunk target {_fmt_dur(secs)}")
    return secs
