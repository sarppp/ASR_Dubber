"""
nemo_diarize.py — Speaker diarization and main pipeline orchestration.

_run_diarization uses a deferred `from nemo.collections.asr.models import ClusteringDiarizer`
inside the function body. This works because by the time it's called, _load_model() →
_import_nemo_asr() has already loaded and cached `nemo` in sys.modules.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo_audio import (
    _audio_duration,
    _cleanup_chunks,
    _extract_audio,
    _fmt_dur,
    _segs_to_srt,
    _split_coarse_segs,
    _vram_gb,
    _words_to_segs,
)
from nemo_model import _estimate_chunk_sec, _transcribe_chunked

log = logging.getLogger("nemo_local")


# ── Diarization ───────────────────────────────────────────────────────────────

def _run_diarization(audio_path: str, work_dir: Path) -> list:
    from nemo.collections.asr.models import ClusteringDiarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Running speaker diarization…")
    ddir = work_dir / "_diarize"
    ddir.mkdir(parents=True, exist_ok=True)

    # NeMo uses the WAV stem as the key for all internal files (VAD output, RTTM, etc.)
    # Spaces/apostrophes in the stem cause silent mismatches — diarization "succeeds"
    # but RTTM lookup fails, returning 1 speaker. Always use a clean fixed name.
    safe_wav = ddir / "input_16k_mono.wav"
    shutil.copy2(audio_path, safe_wav)
    log.info(f"Copied WAV to safe path: {safe_wav.name}")

    mpath = ddir / "manifest.json"
    mpath.write_text(json.dumps({
        "audio_filepath": str(safe_wav.resolve()),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "",
        "num_speakers": None,
        "rttm_filepath": "",
        "uem_filepath": "",
    }) + "\n", encoding="utf-8")
    # Build config using setdefault pattern — avoids "Multiscale parameters not properly setup"
    # caused by OmegaConf treating inline lists differently from post-construction assignments.
    # Mirrors the approach in nemo_diarization_report.py which reliably finds all speakers.
    cfg = {
        "name": "ClusterDiarizer",
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 16,
        "device": device,
        "verbose": True,
        "diarizer": {
            "manifest_filepath": str(mpath),
            "out_dir": str(ddir),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {"model_path": "vad_multilingual_marblenet"},
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {"save_embeddings": False},
            },
            "clustering": {},
        },
    }

    cfg["diarizer"]["vad"].setdefault("parameters", {
        "window_length_in_sec": 0.63, "shift_length_in_sec": 0.01,
        "smoothing": False, "overlap": 0.5,
        "onset": 0.9, "offset": 0.5,
        "pad_onset": 0.0, "pad_offset": 0.0,
        "min_duration_on": 0.0, "min_duration_off": 0.6,
        "filter_speech_first": True,
    })

    spk = cfg["diarizer"]["speaker_embeddings"]["parameters"]
    spk.setdefault("window_length_in_sec", [1.5, 1.0, 0.5])
    spk.setdefault("shift_length_in_sec",  [0.75, 0.5, 0.25])
    spk.setdefault("multiscale_weights",   [1, 1, 1])

    cfg["diarizer"]["clustering"].setdefault("parameters", {
        "oracle_num_speakers": False,
        "max_num_speakers": 8,
        "enhanced_count_thres": 80,
        "max_rp_threshold": 0.25,
        "sparse_search_volume": 30,
        "maj_vote_spk_count": False,
        "chunk_cluster_count": 50,
        "embeddings_per_chunk": 10000,
    })

    cfg = OmegaConf.create(cfg)
    ClusteringDiarizer(cfg=cfg).to(device).diarize()
    rttm_files = list((ddir / "pred_rttms").glob("*.rttm")) or list(ddir.rglob("*.rttm"))
    turns = []
    if rttm_files:
        for line in rttm_files[0].read_text().splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
                try:
                    s, d = float(parts[3]), float(parts[4])
                    turns.append({"speaker": parts[7], "start": s, "end": s + d})
                except (ValueError, IndexError): pass
    turns.sort(key=lambda t: t["start"])
    log.info(f"Diarization done — {len({t['speaker'] for t in turns})} speaker(s), {len(turns)} turns")
    shutil.rmtree(ddir, ignore_errors=True)
    return turns

def _assign_speakers(items: list, turns: list) -> list:
    for item in items:
        s, e = item.get("start", 0.0), item.get("end", 0.0)
        best_spk, best_ov = "unknown", 0.0
        for t in turns:
            ov = max(0.0, min(e, t["end"]) - max(s, t["start"]))
            if ov > best_ov: best_ov, best_spk = ov, t["speaker"]
        item["speaker"] = best_spk
    return items


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run_with_model(model, video_path: str, language: str, model_name: str,
                     translate: bool, diarize: bool, trim_sec: int,
                     safety_factor: float, reserve_gb: float, chunk_override_sec) -> str:
    t0 = time.perf_counter()
    work_dir = Path(video_path).parent
    stem = Path(video_path).stem
    src_lang, tgt_lang = language, ("en" if translate else language)
    is_canary = "canary" in model_name.lower()
    free_before, _ = _vram_gb()

    # Audio
    trim_tag = f"trim{trim_sec}" if trim_sec else "full"
    audio_path = str(work_dir / f"{stem}_nemo_16k_{trim_tag}.wav")
    if Path(audio_path).exists():
        log.info(f"Reusing cached audio: {Path(audio_path).name}")
    else:
        log.info("Extracting 16 kHz mono WAV…")
        _extract_audio(video_path, audio_path, trim_sec)
        log.info(f"Audio extracted {time.perf_counter() - t0:.1f} s")
    audio_dur = _audio_duration(audio_path)
    log.info(f"Audio ready | duration {_fmt_dur(audio_dur)}")

    # Chunk size
    if chunk_override_sec and is_canary:
        chunk_sec = max(30, min(int(chunk_override_sec), 900))
        log.info(f"Manual chunk override: {_fmt_dur(chunk_sec)}")
    else:
        chunk_sec = _estimate_chunk_sec(model_name, safety_factor, reserve_gb)
    log.info(f"Transcribing with {_fmt_dur(chunk_sec)} chunk target…")

    # Transcribe
    t_asr = time.perf_counter()
    words, segs, manifest = _transcribe_chunked(model, audio_path, model_name,
                                                  src_lang, tgt_lang, chunk_sec)
    _cleanup_chunks(manifest, audio_path)
    asr_elapsed = time.perf_counter() - t_asr
    rtf = asr_elapsed / audio_dur if audio_dur > 0 else 0
    log.info(f"Transcription done  {asr_elapsed:.1f} s  (RTF {rtf:.2f}x)")

    if is_canary: words = []
    if not words and not segs:
        raise RuntimeError("NeMo returned no output.")
    log.info(f"Got {len(segs)} segment timestamps" if segs else f"Got {len(words)} word timestamps")

    # Build subtitles
    if diarize:
        turns = _run_diarization(audio_path, work_dir)
        final_segs = (_words_to_segs(_assign_speakers(words, turns), diarized=True) if words
                      else _split_coarse_segs(_assign_speakers(segs, turns)))
        spk_counts: dict = {}
        for seg in final_segs: spk_counts[seg.get("speaker", "?")] = spk_counts.get(seg.get("speaker", "?"), 0) + 1
        log.info(f"Built {len(final_segs)} diarized subtitle segments")
        for spk, n in sorted(spk_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {spk}: {n} segments ({n/len(final_segs)*100:.0f}%)")
        srt = _segs_to_srt(final_segs, diarized=True)
    else:
        final_segs = _words_to_segs(words) if words else _split_coarse_segs(segs)
        log.info(f"Built {len(final_segs)} subtitle segments")
        srt = _segs_to_srt(final_segs)

    wall = time.perf_counter() - t0
    log.info(
        f"{'='*55}\n"
        f"  Total wall time   : {_fmt_dur(wall)}\n"
        f"  Audio duration    : {_fmt_dur(audio_dur)}\n"
        f"  ASR time          : {_fmt_dur(asr_elapsed)}\n"
        f"  Real-time factor  : {rtf:.2f}x  (< 1.0 = faster than real-time)\n"
        f"  Subtitle segments : {len(final_segs)}\n"
        f"{'='*55}"
    )
    return srt
