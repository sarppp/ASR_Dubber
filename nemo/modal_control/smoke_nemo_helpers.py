"""CPU/GPU model helpers extracted from nemo_modal_app; no Modal allocations."""
import sys, os, time, gc, wave, subprocess
from pathlib import Path
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

def _ensure_remote_imports():
    pass

def _fmt_dur(s: float) -> str:
    return f"{int(s // 60)}m{int(s % 60):02d}s" if s >= 60 else f"{s:.1f}s"

def _words_to_segs(words, max_w=10, max_dur=5.0, max_ch=80, diarized=False):
    """Unified segment builder. Fixed: boundary word is no longer dropped."""
    segs, cur_w, cur_t, cur_s, cur_spk = [], [], "", None, None
    for w in words:
        word = w.get("word", "").strip()
        if not word:
            continue
        ws, we = w.get("start", 0.0), w.get("end", 0.0)
        spk = w.get("speaker", "unknown") if diarized else None
        if cur_s is None:
            cur_s, cur_spk = ws, spk
        cand = (cur_t + " " + word).strip() if cur_t else word
        split = (
            len(cur_w) >= max_w
            or (we - cur_s) > max_dur
            or len(cand) > max_ch
            or (cur_t and cur_t[-1] in ".!?" and len(cur_w) >= 3)
            or (diarized and spk != cur_spk and cur_w)
        )
        if split and cur_w:
            seg = {"start": cur_s, "end": cur_w[-1].get("end", cur_s), "text": cur_t}
            if diarized:
                seg["speaker"] = cur_spk
            segs.append(seg)
            cur_w, cur_t, cur_s, cur_spk = [], "", ws, spk
            cand = word  # boundary word becomes start of next segment
        cur_w.append(w)
        cur_t = cand
    if cur_w and cur_t.strip():
        seg = {"start": cur_s, "end": cur_w[-1].get("end", cur_s), "text": cur_t}
        if diarized:
            seg["speaker"] = cur_spk
        segs.append(seg)
    return segs

def _words_to_segments_diarized(words, max_words=10, max_dur=5.0, max_chars=80):
    return _words_to_segs(words, max_w=max_words, max_dur=max_dur, max_ch=max_chars, diarized=True)

def _strip_asr_repetition(text: str, min_unit_words: int = 5, min_reps: int = 3) -> str:
    """Remove hallucinated repetition loops from Canary/Whisper ASR output."""
    words = text.split()
    n = len(words)
    if n < min_unit_words * min_reps:
        return text
    for start in range(n - min_unit_words * min_reps + 1):
        for unit_len in range(min_unit_words, (n - start) // min_reps + 1):
            unit = words[start:start + unit_len]
            reps, pos = 1, start + unit_len
            while pos + unit_len <= n and words[pos:pos + unit_len] == unit:
                reps += 1
                pos += unit_len
            if reps >= min_reps:
                return " ".join(words[:start + unit_len])
    return text

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

def _is_qwen3_asr(model_name: str) -> bool:
    return "Qwen3-ASR" in model_name or "qwen3-asr" in model_name.lower()

def _audio_duration(path: str) -> float:
    import wave
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0

def _vram_gb() -> tuple:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    return free_b / 1024 ** 3, total_b / 1024 ** 3

def _disable_cuda_graphs_in_decoder(model) -> None:
    """Disable CUDA graph capture in NeMo's live decoder object.

    Called after every change_decoding_strategy() — including NeMo's own internal
    call inside transcribe() — because:
      1. model.cfg.decoding is a frozen OmegaConf StructuredConfig; OmegaConf.update
         raises on every key, so we cannot patch the config object itself.
      2. NeMo rebuilds GreedyBatchedTDTInfer from that frozen config at the start of
         each transcribe(), re-enabling use_cuda_graph_decoder=True.
      3. CUDA graph capture at chunk N locks specific GPU memory pages; empty_cache()
         between chunks frees those pages; chunk N+1 replays the stale graph onto
         freed addresses → XID 31 MMU fault → SIGABRT.

    We bypass the frozen config and directly mutate the Python flags on the live
    decoder objects instead.
    """
    _GRAPH_BOOL_ATTRS = (
        "use_cuda_graphs",
        "use_cuda_graph_decoder",  # GreedyBatchedTDTInfer main flag
        "loop_labels",             # label-loop CUDA graph variant
        "allow_cuda_graphs",       # beam decoder flag
    )

    dec = getattr(model, "decoding", None)
    if dec is None:
        return

    def _fix_obj(obj, label: str) -> list:
        """Set all CUDA-graph bool flags to False on obj; return list of changed attrs."""
        changed = []
        for attr in _GRAPH_BOOL_ATTRS:
            if hasattr(obj, attr) and getattr(obj, attr) is True:
                setattr(obj, attr, False)
                changed.append(attr)
        if hasattr(obj, "cuda_graphs_impl") and getattr(obj, "cuda_graphs_impl") is not None:
            obj.cuda_graphs_impl = None
            changed.append("cuda_graphs_impl→None")
        if changed:
            print(f"[CDS-FIX] disabled in {label}: {changed}")
        return changed

    # Top-level decoder wrapper (RNNTDecoding / TDTDecoding)
    _fix_obj(dec, "model.decoding")

    # NeMo's AbstractRNNTDecoding stores the inner GreedyBatchedTDTInfer as
    # self.decoding (same name, one level deeper).  Other names tried as fallback.
    inner_names = ("decoding", "decoding_computer", "_decoding_computer", "greedy_decoding")
    found_inner = False
    for dc_name in inner_names:
        inner = getattr(dec, dc_name, None)
        if inner is None or inner is dec:
            continue
        _fix_obj(inner, f"model.decoding.{dc_name}")
        found_inner = True

    # Broad fallback: scan every attribute on dec for objects that carry the
    # CUDA-graph flags, in case NeMo's naming changes across versions.
    if not found_inner:
        for dc_name in vars(dec):
            if dc_name.startswith("__"):
                continue
            inner = getattr(dec, dc_name, None)
            if inner is None or inner is dec or not isinstance(inner, object):
                continue
            if any(hasattr(inner, a) for a in _GRAPH_BOOL_ATTRS):
                _fix_obj(inner, f"model.decoding.{dc_name}[fallback]")

def _transcribe_manifest(model, manifest: list, model_name: str, language: str, target_lang: str) -> tuple:
    # Disable CUDA graphs in the live decoder (belt-and-suspenders; the real
    # fix is the _patched_cds monkey-patch which runs after every NeMo-internal
    # change_decoding_strategy() call inside transcribe()).
    _disable_cuda_graphs_in_decoder(model)

    all_words, all_segs, text_parts = [], [], []
    n = len(manifest)

    is_qwen3 = _is_qwen3_asr(model_name)

    for ci, entry in enumerate(manifest):
        path, offset = entry["path"], entry["offset"]

        if is_qwen3:
            words, segs = _transcribe_qwen3_asr(model, path, offset, language)
            all_words.extend(words)
            all_segs.extend(segs)
            if n > 1:
                free_gb, _ = _vram_gb()
                print(f"  chunk {ci + 1}/{n} | VRAM free {free_gb:.2f} GB")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            continue

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

    # Strip hallucination loops from Canary segment text
    if "canary" in model_name.lower() and all_segs:
        for seg in all_segs:
            seg["text"] = _strip_asr_repetition(seg["text"])

    return all_words, all_segs, text_parts

def _run_diarization(audio_path: str, work_dir: Path) -> list:
    import json, shutil
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running speaker diarization…")

    ddir = work_dir / "_diarize"
    ddir.mkdir(parents=True, exist_ok=True)

    try:
        # NeMo uses the WAV stem as the key for internal files (VAD, RTTM, etc.)
        # Spaces/apostrophes in the stem cause silent RTTM lookup failures.
        # Always copy to a clean fixed name.
        safe_wav = ddir / "input_16k_mono.wav"
        shutil.copy2(audio_path, safe_wav)
        print(f"Copied WAV to safe path: {safe_wav.name}")

        mpath = ddir / "manifest.json"
        mpath.write_text(
            json.dumps({
                "audio_filepath": str(safe_wav.resolve()),
                "offset": 0, "duration": None, "label": "infer",
                "text": "", "num_speakers": None, "rttm_filepath": "", "uem_filepath": "",
            }) + "\n",
            encoding="utf-8",
        )

        # Scale batch_size for long audio to avoid STFT VRAM spikes
        audio_dur = _audio_duration(audio_path)
        if audio_dur > 600:
            batch_size = 1
        elif audio_dur > 300:
            batch_size = 4
        else:
            batch_size = 16
        print(f"Audio {_fmt_dur(audio_dur)} → diarization batch_size={batch_size}")

        cfg = OmegaConf.create({
            "name": "ClusterDiarizer",
            "num_workers": 0, "sample_rate": 16000, "batch_size": batch_size,
            "device": device, "verbose": True,
            "diarizer": {
                "manifest_filepath": str(mpath),
                "out_dir": str(ddir),
                "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "window_length_in_sec": 0.63, "shift_length_in_sec": 0.01,
                        "smoothing": False, "overlap": 0.5, "onset": 0.5, "offset": 0.3,
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

        try:
            ClusteringDiarizer(cfg=cfg).to(device).diarize()
        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"Diarization OOM on {_fmt_dur(audio_dur)} audio — "
                    "VAD+TitaNet exceeded VRAM. Try a shorter clip or use A10G GPU."
                ) from e
            raise

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
        return turns

    finally:
        shutil.rmtree(ddir, ignore_errors=True)

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
