"""
nemo_diarize.py — Speaker diarization and main pipeline orchestration.

_run_diarization imports ClusteringDiarizer with the same sys.path trick
as nemo_model._import_nemo_asr() — strips the script directory so the local
nemo.py file doesn't shadow the real nemo package.
"""

import gc
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
    _srt_last_timestamp,
    _vram_gb,
    _words_to_segs,
)
from nemo_model import _estimate_chunk_sec, _transcribe_chunked

log = logging.getLogger("nemo_local")


# ── Checkpoint validation ──────────────────────────────────────────────────────

def _validate_checkpoint(checkpoint_file: Path, audio_path: str,
                          trim_sec: int = 0, tolerance: float = 0.10) -> bool:
    """
    Return True if the checkpoint is valid for the current run.

    A checkpoint is STALE (returns False) if any of:
      - JSON is corrupt or unreadable
      - words/segs keys are missing or both are empty
      - stored trim_sec != current trim_sec  (catches --trim 40 vs full run)
      - audio_path exists and stored audio_duration differs by > tolerance

    If trim_sec was never stored (old checkpoint) and audio_path doesn't exist,
    we cannot detect staleness via duration — the function returns True and the
    caller will discover the problem at SRT coverage checks instead.
    """
    try:
        td = json.loads(checkpoint_file.read_text())
    except Exception as e:
        log.warning(f"Checkpoint corrupt ({e}): {checkpoint_file.name} — will re-transcribe")
        return False

    if td.get("words") is None or td.get("segs") is None:
        log.warning(f"Checkpoint missing words/segs keys: {checkpoint_file.name} — will re-transcribe")
        return False

    if not td.get("words") and not td.get("segs"):
        log.warning(f"Checkpoint has empty transcription output: {checkpoint_file.name} — will re-transcribe")
        return False

    # Trim mismatch: --trim 40 checkpoint loaded for full run (or vice versa)
    cp_trim = td.get("trim_sec")
    if cp_trim is not None and cp_trim != trim_sec:
        log.warning(
            f"Stale checkpoint: stored trim_sec={cp_trim} ≠ current trim_sec={trim_sec} "
            f"— discarding {checkpoint_file.name}"
        )
        return False

    # Duration mismatch: compare against actual WAV when it already exists
    cp_dur = float(td.get("audio_duration", 0.0))
    actual_dur = _audio_duration(audio_path) if Path(audio_path).exists() else 0.0
    if cp_dur > 0 and actual_dur > 0:
        ratio = abs(cp_dur - actual_dur) / actual_dur
        if ratio > tolerance:
            log.warning(
                f"Stale checkpoint: stored duration {cp_dur:.1f}s ≠ actual {actual_dur:.1f}s "
                f"({ratio * 100:.0f}% diff) — discarding {checkpoint_file.name}"
            )
            return False

    return True


# ── Diarization helpers ──────────────────────────────────────────────────────

_DIAR_WINDOW_SEC  =  90   # process audio in 90-s windows
_DIAR_OVERLAP_SEC =  30   # 30-s overlap for speaker-ID alignment
_DIAR_STEP_SEC    = _DIAR_WINDOW_SEC - _DIAR_OVERLAP_SEC


def _make_diarizer_cfg(safe_wav: Path, out_dir: Path, batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mpath  = out_dir / "manifest.json"
    mpath.write_text(
        json.dumps({
            "audio_filepath": str(safe_wav.resolve()),
            "offset": 0, "duration": None, "label": "infer",
            "text": "", "num_speakers": None,
            "rttm_filepath": "", "uem_filepath": "",
        }) + "\n",
        encoding="utf-8",
    )
    cfg = {
        "name": "ClusterDiarizer",
        "num_workers": 0, "sample_rate": 16000,
        "batch_size": batch_size, "device": device, "verbose": False,
        "diarizer": {
            "manifest_filepath": str(mpath),
            "out_dir": str(out_dir),
            "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "window_length_in_sec": 0.63, "shift_length_in_sec": 0.01,
                    "smoothing": False, "overlap": 0.5,
                    "onset": 0.5, "offset": 0.3,
                    "pad_onset": 0.0, "pad_offset": 0.0,
                    "min_duration_on": 0.0, "min_duration_off": 0.6,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.0, 0.5],
                    "shift_length_in_sec":  [0.75, 0.5, 0.25],
                    "multiscale_weights":   [1, 1, 1],
                    "save_embeddings":      False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers":  False,
                    "max_num_speakers":     8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold":     0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count":   False,
                    "chunk_cluster_count":  50,
                    "embeddings_per_chunk": 10000,
                },
            },
        },
    }
    return OmegaConf.create(cfg)


def _parse_rttm_dir(pred_rttm_dir: Path) -> list:
    """Parse turns from pred_rttms/*.rttm.  Returns [] if nothing found."""
    files = list(pred_rttm_dir.glob("*.rttm")) or list(pred_rttm_dir.parent.rglob("*.rttm"))
    turns = []
    for line in (files[0].read_text().splitlines() if files else []):
        parts = line.split()
        if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
            try:
                s, d = float(parts[3]), float(parts[4])
                turns.append({"speaker": parts[7], "start": s, "end": s + d})
            except (ValueError, IndexError):
                pass
    return sorted(turns, key=lambda t: t["start"])


def _diarize_one_window(
    safe_wav: Path, win_dir: Path,
    ClusteringDiarizer, device: str, batch_size: int = 64,
) -> list:
    """Run ClusteringDiarizer on one (short) audio window. Returns local turns."""
    win_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_diarizer_cfg(safe_wav, win_dir, batch_size)
    ClusteringDiarizer(cfg=cfg).to(device).diarize()
    return _parse_rttm_dir(win_dir / "pred_rttms")


def _align_speakers(prev_turns: list, curr_turns: list,
                    overlap_start: float, overlap_end: float) -> dict:
    """
    Build a mapping {curr_speaker → prev_speaker} by computing temporal
    overlap between the two sets of turns within the overlap window.
    Unmatched curr speakers are NOT included — caller assigns new global IDs.
    """
    scores: dict = {}   # (prev_spk, curr_spk) → total overlap seconds
    for pt in prev_turns:
        if pt["end"] <= overlap_start or pt["start"] >= overlap_end:
            continue
        for ct in curr_turns:
            if ct["end"] <= overlap_start or ct["start"] >= overlap_end:
                continue
            ov = max(0.0, min(pt["end"], ct["end"]) - max(pt["start"], ct["start"]))
            if ov > 0:
                k = (pt["speaker"], ct["speaker"])
                scores[k] = scores.get(k, 0.0) + ov

    mapping: dict = {}       # curr_spk → prev_spk
    used_prev: set = set()
    for (ps, cs), _ in sorted(scores.items(), key=lambda x: -x[1]):
        if cs not in mapping and ps not in used_prev:
            mapping[cs] = ps
            used_prev.add(ps)
    return mapping


def _merge_split_speakers(turns: list) -> list:
    """
    Post-process windowed diarization output to undo three kinds of artefacts.

    Pass 1 — sequential label-switch (dominant speakers):
      Two "substantial" speakers (≥ 5 % of total speech, multiple turns) whose
      turn ranges are strictly sequential are merged (alignment chain broke at a
      silent window boundary and the same voice got a fresh ID).

    Pass 2 — false split of dominant (long 1-turn speakers):
      A speaker with exactly 1 turn and duration > _SHORT_TURN_SEC is a
      mis-labelled monologue segment of the dominant speaker.  Merge into the
      speaker with the highest total duration.

    Pass 3 — rare-speaker fragmentation (short 1-turn speakers):
      The rare speaker (e.g. interviewer) appears several times in different
      windows and gets a fresh ID each time because the alignment only threads
      the *dominant* speaker.  Group all single-turn speakers with duration
      ≤ _SHORT_TURN_SEC under one ID (first appearance).
    """
    from collections import defaultdict

    _SHORT_TURN_SEC = 15.0   # turns shorter than this are rare-speaker fragments

    if not turns or len({t["speaker"] for t in turns}) < 2:
        return turns

    dur:   dict = defaultdict(float)
    times: dict = defaultdict(list)
    cnt:   dict = defaultdict(int)
    for t in sorted(turns, key=lambda x: x["start"]):
        s = t["speaker"]
        dur[s]  += max(0.0, t["end"] - t["start"])
        times[s].append(t["start"])
        cnt[s]  += 1

    total_dur = sum(dur.values()) or 1.0
    merge_map: dict = {}

    # ── Pass 1: sequential dominant label-switch ─────────────────────────────
    substantial = sorted(
        [s for s, d in dur.items() if d / total_dur >= 0.05 and cnt[s] > 1],
        key=lambda s: -dur[s],
    )
    merged: set = set()
    if len(substantial) >= 2:
        q75 = lambda lst: lst[max(0, int(len(lst) * 0.75) - 1)]
        q25 = lambda lst: lst[max(0, int(len(lst) * 0.25) - 1)]
        for i, s1 in enumerate(substantial):
            if s1 in merged:
                continue
            for s2 in substantial[i + 1:]:
                if s2 in merged:
                    continue
                t1, t2 = sorted(times[s1]), sorted(times[s2])
                early, late = (s1, s2) if sum(t1)/len(t1) < sum(t2)/len(t2) else (s2, s1)
                if q75(sorted(times[early])) < q25(sorted(times[late])):
                    merge_map[s2] = s1
                    merged.add(s2)
                    log.info(
                        f"  [merge] sequential label-switch: "
                        f"{s2} ({dur[s2]:.0f}s) → {s1} ({dur[s1]:.0f}s)"
                    )

    # ── Pass 2: long 1-turn speakers ─────────────────────────────────────────
    # A speaker with exactly 1 turn and duration > _SHORT_TURN_SEC is either:
    #   (a) A false split of the dominant speaker (single occurrence) → merge into dominant
    #   (b) A real secondary speaker appearing in separate windows with long turns
    #       (multiple occurrences, each gets its own ID) → group together
    dominant = max(
        (s for s in dur if s not in merge_map),
        key=lambda s: dur[s],
        default=None,
    )
    long_singles = sorted(
        [s for s in dur
         if s not in merge_map and s != dominant
         and cnt[s] == 1 and dur[s] > _SHORT_TURN_SEC],
        key=lambda s: min(times[s]),
    )
    if len(long_singles) == 1:
        merge_map[long_singles[0]] = dominant
        log.info(
            f"  [merge] long 1-turn false split: "
            f"{long_singles[0]} ({dur[long_singles[0]]:.0f}s, 1 turn) → {dominant}"
        )
    elif len(long_singles) >= 2:
        keep = long_singles[0]
        for s in long_singles[1:]:
            merge_map[s] = keep
            log.info(
                f"  [merge] long 1-turn secondary fragment: "
                f"{s} ({dur[s]:.0f}s) → {keep}"
            )

    # ── Pass 3: short 1-turn rare-speaker fragments → one ID ─────────────────
    fragments = sorted(
        [s for s in dur
         if s not in merge_map
         and cnt[s] <= 2
         and dur[s] <= _SHORT_TURN_SEC],
        key=lambda s: min(times[s]),
    )
    if len(fragments) >= 2:
        keep = fragments[0]
        for frag in fragments[1:]:
            merge_map[frag] = keep
            log.info(
                f"  [merge] rare-speaker fragment: "
                f"{frag} ({dur[frag]:.1f}s) → {keep}"
            )

    if not merge_map:
        return turns

    # Apply (follow chains: A→B→C becomes A→C)
    def resolve(s):
        while s in merge_map:
            s = merge_map[s]
        return s

    for t in turns:
        t["speaker"] = resolve(t["speaker"])

    # Re-number by first appearance
    order: dict = {}
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in order:
            order[t["speaker"]] = f"speaker_{len(order)}"
    for t in turns:
        t["speaker"] = order[t["speaker"]]

    return turns


def _diarize_windowed(
    safe_wav: Path,
    ddir: Path,
    ClusteringDiarizer,
    device: str,
    audio_dur: float,
    batch_size: int = 64,
) -> list:
    """
    Diarize long audio in overlapping 3-min windows and merge speaker IDs.

    Why: NeMo's spectral clustering collapses a minority speaker (<1% of
    embeddings) to the dominant cluster for audio longer than ~5 min.  Short
    windows preserve the relative proportion so default params work.
    """
    import subprocess

    n_windows = max(1, -(-int(audio_dur) // _DIAR_STEP_SEC))  # ceiling div
    log.info(
        f"Windowed diarization: {n_windows} × {_fmt_dur(_DIAR_WINDOW_SEC)} windows "
        f"({_fmt_dur(_DIAR_OVERLAP_SEC)} overlap)"
    )

    all_turns: list = []
    next_id:   list = [0]   # mutable int via list

    for i in range(n_windows):
        win_start = i * _DIAR_STEP_SEC
        win_dur   = min(_DIAR_WINDOW_SEC, audio_dur - win_start)
        if win_dur < 5.0:
            break

        win_dir = ddir / f"win_{i:03d}"
        win_wav = win_dir / "input_16k_mono.wav"
        win_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(safe_wav),
             "-ss", str(win_start), "-t", str(win_dur),
             "-ar", "16000", "-ac", "1", str(win_wav)],
            check=True, capture_output=True,
        )

        log.info(
            f"  Window {i+1}/{n_windows}: "
            f"[{_fmt_dur(win_start)} – {_fmt_dur(win_start + win_dur)}]"
        )
        local_turns = _diarize_one_window(win_wav, win_dir, ClusteringDiarizer,
                                          device, batch_size)

        # Shift local timestamps to global
        for t in local_turns:
            t["start"] += win_start
            t["end"]   += win_start

        if not local_turns:
            continue

        # Build local→global speaker mapping
        local_ids = sorted(set(t["speaker"] for t in local_turns))
        if i == 0:
            id_map = {lid: f"speaker_{next_id[0] + j}" for j, lid in enumerate(local_ids)}
            next_id[0] += len(local_ids)
        else:
            # Use the full previous step + overlap as context (wider = more robust)
            ctx_start = max(0.0, win_start - _DIAR_STEP_SEC)
            ctx_end   = win_start + _DIAR_OVERLAP_SEC
            raw_map = _align_speakers(all_turns, local_turns, ctx_start, ctx_end)
            id_map  = {}
            for lid in local_ids:
                if lid in raw_map:
                    id_map[lid] = raw_map[lid]
                else:
                    id_map[lid] = f"speaker_{next_id[0]}"
                    next_id[0] += 1

        for t in local_turns:
            t["speaker"] = id_map[t["speaker"]]

        # Only keep turns from the non-overlap portion of this window
        keep_start = win_start + (_DIAR_OVERLAP_SEC if i > 0 else 0.0)
        keep_end   = (win_start + win_dur
                      if i == n_windows - 1
                      else win_start + _DIAR_STEP_SEC + _DIAR_OVERLAP_SEC)
        all_turns.extend(
            t for t in local_turns
            if t["start"] >= keep_start and t["start"] < keep_end
        )

    all_turns.sort(key=lambda t: t["start"])
    all_turns = _merge_split_speakers(all_turns)
    all_turns.sort(key=lambda t: t["start"])
    spk_set = {t["speaker"] for t in all_turns}
    log.info(f"Windowed diarization complete — {len(spk_set)} speaker(s), {len(all_turns)} turns")
    return all_turns


# ── Diarization ───────────────────────────────────────────────────────────────

def _run_diarization(audio_path: str, work_dir: Path) -> list:
    # Import via sys.path trick — when using Qwen3-ASR, _import_nemo_asr() was
    # never called so 'nemo' isn't cached in sys.modules yet. The local nemo.py
    # file would shadow the real nemo package without this workaround.
    import importlib, sys
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    try:
        sys.path = [e for e in original_path
                    if not (e and Path(e).resolve() == script_dir)]
        ClusteringDiarizer = importlib.import_module(
            "nemo.collections.asr.models"
        ).ClusteringDiarizer
    finally:
        sys.path = original_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Running speaker diarization…")
    ddir = work_dir / "_diarize"
    ddir.mkdir(parents=True, exist_ok=True)

    try:
        # NeMo uses the WAV stem as the key for all internal files (VAD output, RTTM, etc.)
        # Spaces/apostrophes in the stem cause silent mismatches — diarization "succeeds"
        # but RTTM lookup fails, returning 1 speaker. Always use a clean fixed name.
        safe_wav = ddir / "input_16k_mono.wav"
        shutil.copy2(audio_path, safe_wav)
        log.info(f"Copied WAV to safe path: {safe_wav.name}")

        audio_dur = _audio_duration(audio_path)
        batch_size = 64 if audio_dur > 600 else (128 if audio_dur > 300 else 256)
        log.info(f"Audio {_fmt_dur(audio_dur)} → diarization batch_size={batch_size}")

        try:
            if audio_dur > _DIAR_WINDOW_SEC:
                # Long audio: use windowed approach so minority speakers aren't
                # buried by the dominant speaker's embeddings in global clustering.
                turns = _diarize_windowed(
                    safe_wav, ddir, ClusteringDiarizer, device, audio_dur, batch_size
                )
            else:
                # Short audio: single run is fine.
                single_dir = ddir / "single"
                single_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(safe_wav, single_dir / "input_16k_mono.wav")
                turns = _diarize_one_window(
                    single_dir / "input_16k_mono.wav",
                    single_dir, ClusteringDiarizer, device, batch_size,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"Diarization OOM on {_fmt_dur(audio_dur)} audio — "
                    "ASR model was offloaded but VAD+TitaNet still exceeded VRAM. "
                    "Try --chunk-override to reduce audio length or use a smaller GPU reserve."
                ) from e
            raise

        turns.sort(key=lambda t: t["start"])
        log.info(f"Diarization done — {len({t['speaker'] for t in turns})} speaker(s), {len(turns)} turns")
        return turns

    finally:
        shutil.rmtree(ddir, ignore_errors=True)


def _assign_speakers(items: list, turns: list) -> list:
    for item in items:
        s, e = item.get("start", 0.0), item.get("end", 0.0)
        best_spk, best_ov = "unknown", 0.0
        for t in turns:
            ov = max(0.0, min(e, t["end"]) - max(s, t["start"]))
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
        item["speaker"] = best_spk
    return items


def _build_srt(words: list, segs: list, turns: list, diarize: bool) -> str:
    """Merge transcription + diarization results into SRT text."""
    if diarize:
        items = _assign_speakers(words if words else segs, turns)
        final_segs = (_words_to_segs(items, diarized=True) if words
                      else _split_coarse_segs(items))
        spk_counts: dict = {}
        for seg in final_segs:
            spk_counts[seg.get("speaker", "?")] = spk_counts.get(seg.get("speaker", "?"), 0) + 1
        log.info(f"Built {len(final_segs)} diarized subtitle segments")
        for spk, n in sorted(spk_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {spk}: {n} segments ({n / len(final_segs) * 100:.0f}%)")
        return _segs_to_srt(final_segs, diarized=True)
    else:
        final_segs = _words_to_segs(words) if words else _split_coarse_segs(segs)
        log.info(f"Built {len(final_segs)} subtitle segments")
        return _segs_to_srt(final_segs)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run_with_model(model, video_path: str, language: str, model_name: str,
                     translate: bool, diarize: bool, trim_sec: int,
                     safety_factor: float, reserve_gb: float, chunk_override_sec) -> str:
    t0 = time.perf_counter()
    work_dir = Path(video_path).parent
    stem = Path(video_path).stem
    src_lang = language
    tgt_lang = "en" if translate else language
    is_canary = "canary" in model_name.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    free_before, _ = _vram_gb()

    trim_tag = f"trim{trim_sec}" if trim_sec else "full"
    is_wav_input = Path(video_path).suffix.lower() == ".wav"
    if is_wav_input and trim_sec == 0:
        # WAV with no trim: use directly, no FFmpeg needed
        audio_path = video_path
        log.info(f"WAV input — skipping FFmpeg extraction: {Path(video_path).name}")
    else:
        # Non-WAV OR WAV with trim: always produce a processed output file
        audio_path = str(work_dir / f"{stem}_nemo_16k_{trim_tag}.wav")

    # Checkpoint file paths (defined once, used throughout)
    transcript_file  = work_dir / f"{stem}_nemo_{src_lang}_transcript.json"
    diarization_file = work_dir / f"{stem}_nemo_{src_lang}_diarization.json"

    # ── Fast resume: both checkpoints exist ──────────────────────────────────
    if transcript_file.exists() and (not diarize or diarization_file.exists()):
        if not _validate_checkpoint(transcript_file, audio_path, trim_sec):
            log.info("Discarding stale checkpoint(s) — will re-transcribe from scratch")
            transcript_file.unlink(missing_ok=True)
            diarization_file.unlink(missing_ok=True)
        else:
            log.info("Resuming from cached intermediate results — skipping ASR and diarization")
            td = json.loads(transcript_file.read_text())
            words, segs = td["words"], td["segs"]
            audio_dur, asr_elapsed, rtf = td["audio_duration"], td["asr_elapsed"], td["rtf"]
            turns = json.loads(diarization_file.read_text())["turns"] if diarize else []
            log.info(f"Loaded {len(words)} words / {len(segs)} segs | ASR={asr_elapsed:.1f}s RTF={rtf:.2f}x")
            srt = _build_srt(words, segs, turns, diarize)
            wall = time.perf_counter() - t0
            log.info(f"Resume complete in {_fmt_dur(wall)} (no GPU used)")
            return srt

    # ── Extract audio ─────────────────────────────────────────────────────────
    if Path(audio_path).exists():
        log.info(f"Reusing cached audio: {Path(audio_path).name}")
    elif audio_path != video_path:
        action = f"Trimming WAV to first {_fmt_dur(trim_sec)}…" if (is_wav_input and trim_sec) else "Extracting 16 kHz mono WAV…"
        log.info(action)
        _extract_audio(video_path, audio_path, trim_sec)
        log.info(f"Audio ready {time.perf_counter() - t0:.1f}s")
    audio_dur = _audio_duration(audio_path)
    log.info(f"Audio ready | duration {_fmt_dur(audio_dur)}")

    # ── Transcribe (or load checkpoint) ──────────────────────────────────────
    if transcript_file.exists():
        if not _validate_checkpoint(transcript_file, audio_path, trim_sec):
            log.info("Discarding stale transcript checkpoint — will re-transcribe")
            transcript_file.unlink(missing_ok=True)
        else:
            log.info(f"Resuming: loading cached transcription ({transcript_file.name})")
            td = json.loads(transcript_file.read_text())
            words, segs = td["words"], td["segs"]
            asr_elapsed, rtf = td["asr_elapsed"], td["rtf"]
            log.info(f"ASR was {asr_elapsed:.1f}s RTF={rtf:.2f}x — skipping to diarization")

    if not transcript_file.exists():
        if chunk_override_sec:
            if is_canary:
                # Cap Canary at 60s even with manual override — quality collapses above.
                chunk_sec = max(30, min(int(chunk_override_sec), 60))
                log.info(f"Manual chunk override (Canary cap 60s): {_fmt_dur(chunk_sec)}")
            else:
                chunk_sec = max(30, int(chunk_override_sec))
                log.info(f"Manual chunk override: {_fmt_dur(chunk_sec)}")
        else:
            chunk_sec = _estimate_chunk_sec(model_name, safety_factor, reserve_gb)
        log.info(f"Transcribing with {_fmt_dur(chunk_sec)} chunk target…")

        t_asr = time.perf_counter()
        manifest = []
        try:
            words, segs, manifest = _transcribe_chunked(model, audio_path, model_name,
                                                          src_lang, tgt_lang, chunk_sec)
        finally:
            _cleanup_chunks(manifest, audio_path)
        asr_elapsed = time.perf_counter() - t_asr
        rtf = asr_elapsed / audio_dur if audio_dur > 0 else 0
        log.info(f"Transcription done {asr_elapsed:.1f}s (RTF {rtf:.2f}x)")

        if is_canary:
            words = []
        if not words and not segs:
            raise RuntimeError("NeMo returned no output.")
        log.info(f"Got {'words' if words else 'segs'}: {len(words) if words else len(segs)} items")

        # Save checkpoint so a diarization OOM doesn't lose the ASR work
        # trim_sec is stored so a subsequent run with a different --trim detects staleness
        transcript_file.write_text(
            json.dumps({"words": words, "segs": segs,
                        "audio_duration": audio_dur, "asr_elapsed": asr_elapsed, "rtf": rtf,
                        "trim_sec": trim_sec},
                       indent=2),
            encoding="utf-8",
        )
        log.info(f"✓ Saved ASR checkpoint: {transcript_file.name}")

    # ── Diarize (or load checkpoint) ──────────────────────────────────────────
    turns = []
    if diarize:
        if diarization_file.exists():
            log.info(f"Resuming: loading cached diarization ({diarization_file.name})")
            turns = json.loads(diarization_file.read_text())["turns"]
        else:
            # Offload the ASR model from VRAM before loading VAD + TitaNet.
            # Without this, Parakeet (~2GB) or Canary (~5GB) + diarization models
            # + STFT tensors for long audio exceed GPU memory.
            log.info("Offloading ASR model to CPU to free VRAM for diarization…")
            try:
                model.cpu()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            free, total = _vram_gb()
            log.info(f"VRAM after offload: {free:.1f}/{total:.1f} GB free")

            turns = _run_diarization(audio_path, work_dir)

            # Save diarization checkpoint
            diarization_file.write_text(
                json.dumps({"turns": turns}, indent=2), encoding="utf-8")
            log.info(f"✓ Saved diarization checkpoint: {diarization_file.name}")

            # Restore model to GPU — important if --all processes multiple videos
            log.info("Restoring ASR model to GPU…")
            try:
                model.to(device)
            except Exception:
                pass

    # ── Build SRT ─────────────────────────────────────────────────────────────
    srt = _build_srt(words, segs, turns, diarize)

    # ── Coverage sanity check ─────────────────────────────────────────────────
    # Catches silent failures: stale checkpoint, ASR returning only first chunk,
    # _strip_asr_repetition false-positive, audio extraction cut short, etc.
    if audio_dur > 60:
        last_ts = _srt_last_timestamp(srt)
        coverage = last_ts / audio_dur if audio_dur > 0 else 0.0
        if coverage < 0.50:
            log.error(
                f"⚠️  SRT COVERAGE CRITICAL: last timestamp {_fmt_dur(last_ts)} "
                f"vs audio {_fmt_dur(audio_dur)} ({coverage * 100:.0f}%) — "
                f"transcription appears incomplete. Delete the checkpoint JSON and re-run."
            )
        elif coverage < 0.85:
            log.warning(
                f"⚠️  SRT coverage low: {_fmt_dur(last_ts)} / {_fmt_dur(audio_dur)} "
                f"({coverage * 100:.0f}%) — last segment may be missing."
            )

    wall = time.perf_counter() - t0
    seg_count = srt.count("\n\n") + 1 if srt.strip() else 0
    log.info(
        f"{'='*55}\n"
        f"  Total wall time   : {_fmt_dur(wall)}\n"
        f"  Audio duration    : {_fmt_dur(audio_dur)}\n"
        f"  ASR time          : {_fmt_dur(asr_elapsed)}\n"
        f"  Real-time factor  : {rtf:.2f}x  (< 1.0 = faster than real-time)\n"
        f"  Subtitle segments : {seg_count}\n"
        f"{'='*55}"
    )
    timing = {
        "total_sec":    round(wall,        1),
        "asr_sec":      round(asr_elapsed, 1),
        "audio_dur_sec": round(audio_dur,  1),
        "rtf":          round(rtf,         3),
        "segments":     seg_count,
    }
    return srt, timing
