"""
nemo_diarization_report.py — NeMo local ASR/translation runner

Usage:
cd nemo && uv run python nemo_diarization_report.py \
    --input "input_name.wav"
"""

#!/usr/bin/env python3

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
DEFAULT_WORKDIR = "output/nemo_diarization"

try:
    import torch.multiprocessing as _tmp_mp

    _tmp_mp.set_sharing_strategy("file_system")
except Exception:
    pass


# --- PYTORCH 2.6 FIX (best-effort) ---
try:
    _original_load = torch.load

    def safe_load_wrapper(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = safe_load_wrapper
except Exception:
    pass


@dataclass
class Turn:
    speaker: str
    start: float
    end: float


def _run_ffmpeg_extract_wav(input_path: Path, wav_path: Path, sr: int = 16000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
        "-y",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)


def _parse_time_seconds(s: str) -> float:
    s = s.strip()
    if not s:
        raise ValueError("empty time")
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid time format: {s}")
        m = float(parts[0])
        sec = float(parts[1])
        return m * 60.0 + sec
    return float(s)


def _parse_window(spec: str) -> Tuple[float, float]:
    spec = spec.strip().lower()
    if spec in {"first10", "first_10", "first10s", "first_10s"}:
        return 0.0, 10.0
    if "-" not in spec:
        raise ValueError(f"invalid window spec: {spec} (expected start-end)")
    a, b = spec.split("-", 1)
    start = _parse_time_seconds(a)
    end = _parse_time_seconds(b)
    if end <= start:
        raise ValueError(f"invalid window spec: {spec} (end must be > start)")
    return start, end


def _median_pitch_for_window(wav_path: Path, start_s: float, end_s: float) -> Optional[float]:
    try:
        import librosa
    except Exception:
        return None

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    a = max(0, int(start_s * sr))
    b = min(int(end_s * sr), y.shape[0])
    if b <= a:
        return None
    clip = y[a:b]
    if clip.size < sr * 0.20:
        return None

    try:
        f0 = librosa.yin(clip, fmin=50, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            return None
        return float(np.median(f0))
    except Exception:
        return None


def _parse_rttm(rttm_path: Path) -> List[Turn]:
    turns: List[Turn] = []
    if not rttm_path.exists():
        return turns

    # RTTM format (simplified):
    # SPEAKER <file-id> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            if parts[0].upper() != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = str(parts[7])
            turns.append(Turn(speaker=spk, start=start, end=start + dur))

    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def _summarize_turns(turns: List[Turn]) -> Dict[str, Dict[str, float]]:
    per: Dict[str, Dict[str, float]] = {}
    for t in turns:
        dur = max(0.0, t.end - t.start)
        if t.speaker not in per:
            per[t.speaker] = {"duration": 0.0, "turns": 0.0}
        per[t.speaker]["duration"] += dur
        per[t.speaker]["turns"] += 1.0
    return per


def _gender_estimate_pitch(wav_path: Path, turns: List[Turn], max_seconds_per_speaker: float = 12.0) -> Dict[str, Dict[str, float | str]]:
    try:
        import librosa
    except Exception:
        return {}

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)

    out: Dict[str, Dict[str, float | str]] = {}
    speakers = sorted({t.speaker for t in turns})
    for spk in speakers:
        remaining = float(max_seconds_per_speaker)
        clips: List[np.ndarray] = []

        for t in turns:
            if t.speaker != spk:
                continue
            if remaining <= 0:
                break
            seg_dur = max(0.0, t.end - t.start)
            if seg_dur <= 0:
                continue
            take = min(seg_dur, remaining)
            a = max(0, int(t.start * sr))
            b = min(int((t.start + take) * sr), y.shape[0])
            if b <= a:
                continue
            clip = y[a:b]
            if clip.size < int(sr * 0.25):
                continue
            clips.append(clip)
            remaining -= take

        if not clips:
            continue
        clip = np.concatenate(clips)

        try:
            f0 = librosa.yin(clip, fmin=50, fmax=400, sr=sr)
            f0 = f0[np.isfinite(f0)]
            if f0.size == 0:
                continue
            med = float(np.median(f0))
            guess = "female" if med >= 165 else "male"
            out[spk] = {"median_f0_hz": med, "gender_guess": guess}
        except Exception:
            continue

    return out


def _hash_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _source_fingerprint(path: Path) -> Dict[str, int | str]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": _hash_file(path),
    }


def _cache_key(source_fp: Dict[str, int | str]) -> str:
    payload = json.dumps(source_fp, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cached_turns(turns_path: Path) -> List[Turn]:
    if not turns_path.exists():
        return []
    try:
        raw = json.loads(turns_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    turns: List[Turn] = []
    for item in raw:
        try:
            turns.append(Turn(**item))
        except TypeError:
            continue
    return turns


def _store_cached_turns(turns_path: Path, turns: List[Turn]) -> None:
    turns_path.write_text(
        json.dumps([t.__dict__ for t in turns], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_diarized_srt(video_path: Path) -> Optional[Path]:
    pattern = f"{video_path.stem}.nemo.*.diarize.srt"
    candidates = sorted(
        (p for p in video_path.parent.glob(pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_srt_segments(srt_path: Path) -> list:
    if not srt_path.exists():
        return []
    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(r"\d+\s+([0-9:,\s-]+)\s+\[Speaker (\d+)\]", re.MULTILINE)
    segments = []
    for block in content.split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        if not re.match(r"^\d+$", lines[0]):
            continue
        timing = lines[1]
        label_match = re.search(r"\[Speaker (\d+)\]", " ".join(lines[2:]))
        if not label_match:
            continue
        speaker_idx = label_match.group(1)
        time_match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) \-\-> (\d{2}):(\d{2}):(\d{2}),(\d{3})", timing)
        if not time_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        segments.append({"speaker": f"Speaker {speaker_idx}", "start": start, "end": end})
    return segments


def _analyze_srt_distribution(segments: list, srt_label_map: Dict[str, str] = None) -> Dict[str, Dict[str, float]]:
    if not segments:
        return {}
    counts: Dict[str, int] = {}
    for seg in segments:
        key = seg["speaker"]
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    # Build reverse map: srt_label -> nemo_id
    reverse_map: Dict[str, str] = {}
    if srt_label_map:
        for nemo_id, srt_label in srt_label_map.items():
            reverse_map[srt_label] = nemo_id
    return {
        spk: {
            "nemo_id": reverse_map.get(spk),
            "segments": count,
            "percentage": (count / total) * 100 if total else 0.0,
        }
        for spk, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    }

def _map_srt_to_diar_speakers(srt_segments: list, turns: list) -> Dict[str, str]:
    if not srt_segments or not turns:
        return {}
    overlap_matrix: Dict[str, Dict[str, float]] = {}
    for seg in srt_segments:
        srt_spk = seg["speaker"]
        overlap_matrix.setdefault(srt_spk, {})
        for turn in turns:
            ov = max(0.0, min(seg["end"], turn.end) - max(seg["start"], turn.start))
            if ov <= 0:
                continue
            overlap_matrix[srt_spk][turn.speaker] = overlap_matrix[srt_spk].get(turn.speaker, 0.0) + ov

    mapping: Dict[str, str] = {}
    used_diar = set()
    for srt_spk, overlaps in sorted(overlap_matrix.items(), key=lambda kv: max(kv[1].values(), default=0), reverse=True):
        diar_spk = max((spk for spk in overlaps if spk not in used_diar), key=lambda spk: overlaps[spk], default=None)
        if diar_spk:
            mapping[diar_spk] = srt_spk
            used_diar.add(diar_spk)
    return mapping


def _import_nemo_module(module_path: str):
    """Import a NeMo submodule even if this repo defines nemo.py."""
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)

    def _is_script_dir(entry: Optional[str]) -> bool:
        if entry is None:
            return False
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            return False
        return resolved == script_dir

    try:
        sys.path = [p for p in original_path if not _is_script_dir(p)]
        return importlib.import_module(module_path)
    finally:
        sys.path = original_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker diarization report using NVIDIA NeMo")
    parser.add_argument("--input", default="", help="Audio/video input path (auto-detect latest video if omitted)")
    parser.add_argument(
        "--search-dir",
        default="",
        help="Directory to scan when --input is not supplied (default: repo root or <repo>/videos)",
    )
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Work/output directory (defaults to <video>_diar when --input auto-detected)")
    parser.add_argument("--out", default="", help="Output report json path (default: <workdir>/nemo_diarization_report.json)")

    # NeMo knobs (simple + safe defaults)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Use 0 to avoid /dev/shm bus errors in containers.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for NeMo diarizer (lower is safer).")
    parser.add_argument("--vad-model", default="vad_multilingual_marblenet", help="NeMo VAD model name or .nemo")
    parser.add_argument("--spk-embed-model", default="titanet_large", help="Speaker embedding model name or .nemo")

    # Optional hints (do not force exact count)
    parser.add_argument("--min-speakers", type=int, default=None, help="Hint: minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Hint: maximum number of speakers")

    # Pitch windows (same as your other tool)
    parser.add_argument("--pitch-window", action="append", default=[], help="Repeatable: start-end in seconds or mm:ss. e.g. 23-27, 1:55-2:00, first10")
    parser.add_argument("--pitch-defaults", action="store_true", help="Print pitch for default windows: first10, 23-27, 30-35, 1:55-2:00")

    parser.add_argument("--no-gender", action="store_true", help="Disable pitch-based gender heuristic")
    parser.add_argument("--cache", action="store_true", help="Persist WAV + RTTM to reuse on reruns")
    parser.add_argument("--reuse-cache", action="store_true", help="Skip diarization if cached RTTM exists")
    parser.add_argument("--cache-dir", default="", help="Optional cache directory (default: <workdir>/cache)")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.input:
        inp = Path(args.input)
    else:
        if args.search_dir:
            search_root_candidate = Path(args.search_dir)
        else:
            repo_videos = script_dir / "videos"
            search_root_candidate = repo_videos if repo_videos.exists() else script_dir
        if search_root_candidate.is_absolute():
            search_root = search_root_candidate
        else:
            cwd_candidate = Path.cwd() / search_root_candidate
            script_candidate = script_dir / search_root_candidate
            repo_root_candidate = script_dir
            if cwd_candidate.exists():
                search_root = cwd_candidate
            elif script_candidate.exists():
                search_root = script_candidate
            elif repo_root_candidate.exists():
                search_root = repo_root_candidate
            else:
                search_root = cwd_candidate
        if not search_root.exists():
            raise FileNotFoundError(f"Search directory not found: {search_root}")
        candidates = [
            p
            for p in sorted(search_root.rglob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.is_file() and p.suffix.lower() in VIDEO_EXT
        ]
        if not candidates:
            raise FileNotFoundError(f"No video files found under {search_root}")
        inp = candidates[0]
        print(f"🎯 Auto-selected input: {inp}")
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    inp = inp.resolve()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    try:
        from omegaconf import OmegaConf
        nemo_asr_models = _import_nemo_module("nemo.collections.asr.models")
        ClusteringDiarizer = getattr(nemo_asr_models, "ClusteringDiarizer")
    except Exception as e:
        print(f"❌ NeMo not available: {e}")
        print("Install (example): pip install nemo_toolkit[asr]")
        print("Note: NeMo diarization downloads models on first run.")
        print("Note: Installing NeMo can change versions of core deps (e.g. transformers). If Qwen TTS breaks after install, restore transformers==4.57.3 in your Qwen env.")
        return 1

    if args.workdir == DEFAULT_WORKDIR:
        workdir = inp.parent / f"{inp.stem}_diar"
    else:
        workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cache_enabled = bool(args.cache or args.reuse_cache)
    cache_root = Path(args.cache_dir) if args.cache_dir else (workdir / "cache")
    cache_dir = cache_root
    cache_meta_path = cache_root / "meta.json"
    cache_turns_path = cache_root / "turns.json"
    cache_rttm_path = cache_root / "cached.rttm"
    diar_out_dir = workdir
    manifest = workdir / "manifest.json"
    wav_path = workdir / "input_16k_mono.wav"

    if cache_enabled:
        cache_root.mkdir(parents=True, exist_ok=True)
        source_fp = _source_fingerprint(inp)
        cache_key = _cache_key(source_fp)
        cache_dir = cache_root / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_meta_path = cache_dir / "meta.json"
        cache_turns_path = cache_dir / "turns.json"
        cache_rttm_path = cache_dir / "cached.rttm"
        diar_out_dir = cache_dir / "nemo_out"
        diar_out_dir.mkdir(parents=True, exist_ok=True)
        manifest = cache_dir / "manifest.json"
        wav_path = cache_dir / "input_16k_mono.wav"

        if cache_meta_path.exists():
            try:
                meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            if meta.get("source") != source_fp:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                diar_out_dir = cache_dir / "nemo_out"
                diar_out_dir.mkdir(parents=True, exist_ok=True)
                manifest = cache_dir / "manifest.json"
                wav_path = cache_dir / "input_16k_mono.wav"

        cache_meta_path.write_text(json.dumps({"source": source_fp}, indent=2), encoding="utf-8")

    if cache_enabled and wav_path.exists():
        print("♻️  Reusing cached WAV")
    else:
        _run_ffmpeg_extract_wav(inp, wav_path, sr=16000)

    # Optional pitch analysis
    pitch_specs = list(args.pitch_window or [])
    if args.pitch_defaults:
        pitch_specs.extend(["first10", "23-27", "30-35", "1:55-2:00"])
    if pitch_specs:
        for spec in pitch_specs:
            try:
                a_s, b_s = _parse_window(spec)
            except Exception as e:
                print(f"⚠️  Skipping pitch window '{spec}': {e}")
                continue
            med = _median_pitch_for_window(wav_path, a_s, b_s)
            if med is None:
                print(f"pitch≈(unavailable) for {spec} ({a_s:.2f}-{b_s:.2f}s)")
            else:
                print(f"pitch≈{med:.0f}Hz for {spec} ({a_s:.2f}-{b_s:.2f}s)")

    # NeMo expects a manifest
    file_id = wav_path.stem
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": str(wav_path.resolve()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "",
                "num_speakers": None,
                "rttm_filepath": "",
                "uem_filepath": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Minimal *valid* config for NeMo clustering diarizer.
    # NeMo expects certain keys to exist (e.g. diarizer.oracle_vad), otherwise it errors.
    # Based on NeMo's example config (diar_infer_meeting.yaml), but trimmed.
    cfg = {
        "name": "ClusterDiarizer",
        "num_workers": int(args.num_workers),
        "sample_rate": 16000,
        "batch_size": int(args.batch_size),
        "device": args.device,
        "verbose": True,
        "diarizer": {
            "manifest_filepath": str(manifest),
            "out_dir": str(diar_out_dir),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {"model_path": args.vad_model},
            "speaker_embeddings": {
                "model_path": args.spk_embed_model,
                "parameters": {"save_embeddings": False},
            },
            "clustering": {},
        },
    }

    # Provide minimal parameter blocks that NeMo expects to exist.
    cfg["diarizer"]["vad"].setdefault(
        "parameters",
        {
            "window_length_in_sec": 0.63,
            "shift_length_in_sec": 0.01,
            "smoothing": False,
            "overlap": 0.5,
            "onset": 0.9,
            "offset": 0.5,
            "pad_onset": 0.0,
            "pad_offset": 0.0,
            "min_duration_on": 0.0,
            "min_duration_off": 0.6,
            "filter_speech_first": True,
        },
    )

    cfg["diarizer"]["speaker_embeddings"].setdefault("parameters", {})
    # Merge required defaults without clobbering user-provided values
    spk_params = cfg["diarizer"]["speaker_embeddings"]["parameters"]
    spk_params.setdefault("window_length_in_sec", [1.5, 1.0, 0.5])
    spk_params.setdefault("shift_length_in_sec", [0.75, 0.5, 0.25])
    spk_params.setdefault("multiscale_weights", [1, 1, 1])
    spk_params.setdefault("save_embeddings", False)

    cfg["diarizer"].setdefault("clustering", {})
    cfg["diarizer"]["clustering"].setdefault(
        "parameters",
        {
            "oracle_num_speakers": False,
            "max_num_speakers": 8,
            "enhanced_count_thres": 80,
            "max_rp_threshold": 0.25,
            "sparse_search_volume": 30,
            "maj_vote_spk_count": False,
            "chunk_cluster_count": 50,
            "embeddings_per_chunk": 10000,
        },
    )

    # Best-effort speaker count hints
    if args.min_speakers is not None or args.max_speakers is not None:
        cfg["diarizer"]["clustering"].setdefault("parameters", {})
        if args.min_speakers is not None:
            cfg["diarizer"]["clustering"]["parameters"]["min_num_speakers"] = int(args.min_speakers)
        if args.max_speakers is not None:
            cfg["diarizer"]["clustering"]["parameters"]["max_num_speakers"] = int(args.max_speakers)

    cfg = OmegaConf.create(cfg)

    cached_turns: List[Turn] = []
    cached_rttm: Optional[Path] = None
    if cache_enabled and args.reuse_cache:
        cached_turns = _load_cached_turns(cache_turns_path)
    if cached_turns:
        print("⚡ Using cached diarization turns")
        cached_rttm = cache_rttm_path if cache_rttm_path.exists() else None

    if cached_turns:
        turns = cached_turns
        rttm_path = cached_rttm or (diar_out_dir / "pred_rttms" / f"{file_id}.rttm")
    else:
        print("ℹ️  Running NeMo ClusteringDiarizer…")
        sd_model = ClusteringDiarizer(cfg=cfg).to(args.device)
        sd_model.diarize()

        pred_rttm_dir = diar_out_dir / "pred_rttms"
        rttm_candidates = list(pred_rttm_dir.glob("*.rttm"))
        if not rttm_candidates:
            rttm_candidates = list(diar_out_dir.rglob("*.rttm"))
        if not rttm_candidates:
            print("❌ Could not find NeMo RTTM output.")
            return 2
        rttm_path = rttm_candidates[0]
        turns = _parse_rttm(rttm_path)
        if cache_enabled:
            cache_rttm_path.write_text(Path(rttm_path).read_text(encoding="utf-8"), encoding="utf-8")
            _store_cached_turns(cache_turns_path, turns)

    per = _summarize_turns(turns)

    speakers_sorted = sorted(per.items(), key=lambda kv: kv[1]["duration"], reverse=True)
    srt_path = _find_diarized_srt(inp)
    srt_segments = _parse_srt_segments(srt_path) if srt_path else []
    srt_label_map = _map_srt_to_diar_speakers(srt_segments, turns)
    srt_distribution = _analyze_srt_distribution(srt_segments, srt_label_map)
    if not srt_label_map:
        unique_speakers = sorted({t.speaker for t in turns})
        for idx, spk in enumerate(unique_speakers, start=1):
            srt_label_map.setdefault(spk, f"Speaker {idx}")

    turns_with_labels = [
        {
            **t.__dict__,
            "srt_label": srt_label_map.get(t.speaker),
        }
        for t in turns
    ]

    speakers_report = []
    for spk, stats in speakers_sorted:
        speakers_report.append(
            {
                "nemo_id": spk,
                "srt_label": srt_label_map.get(spk),
                "duration": stats["duration"],
                "turns": stats["turns"],
            }
        )

    report = {
        "wav_16k_mono": str(wav_path),
        "nemo_rttm": str(rttm_path),
        "num_speakers": len(per),
        "speakers": speakers_report,
        "gender_heuristic": {},
        "turns": turns_with_labels,
        "srt_distribution": srt_distribution,
        "srt_label_map": srt_label_map,
    }

    if not args.no_gender:
        report["gender_heuristic"] = _gender_estimate_pitch(wav_path, turns)

    out_path = Path(args.out) if args.out else (workdir / "nemo_diarization_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Speakers: {report['num_speakers']}")
    for entry in speakers_report:
        extra = ""
        g = report.get("gender_heuristic", {}).get(entry["nemo_id"])
        if isinstance(g, dict) and ("median_f0_hz" in g) and ("gender_guess" in g):
            extra = f" | pitch≈{float(g['median_f0_hz']):.0f}Hz -> {g['gender_guess']}"
        print(
            f"- {entry.get('srt_label') or entry['nemo_id']}: {entry['duration']:.1f}s across {int(entry['turns'])} turns"
            + (f" (NeMo {entry['nemo_id']})" if entry.get("nemo_id") else "")
            + extra
        )
    print(f"✅ Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
