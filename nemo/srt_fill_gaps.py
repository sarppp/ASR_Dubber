"""
srt_fill_gaps.py — Fill gaps in SRT files using Whisper

Detects gaps between subtitle segments, transcribes missing audio with Whisper,
and inserts new segments with speaker attribution using speaker embeddings.

Usage:
  cd nemo && uv run python srt_fill_gaps.py video.mp4 input.srt output.srt --min-gap 2.0
  cd nemo && uv run python srt_fill_gaps.py video.mp4 input.srt output.srt --whisper-model base
  cd nemo && uv run python srt_fill_gaps.py video.mp4 input.srt output.srt --no-embeddings  # Use old proximity method
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

_WHISPER_PY = os.environ.get("WHISPER_PY", "")
_WHISPER_HELPER = Path(__file__).parent / "whisper_transcribe_helper.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("srt_fill_gaps")


def parse_srt_timestamp(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.strip().replace(',', '.')
    parts = ts.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {ts}")
    h, m, s = parts
    return float(h) * 3600 + float(m) * 60 + float(s)


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')


def parse_srt(content: str) -> List[dict]:
    """Parse SRT content into list of segments."""
    segments = []
    entries = re.split(r'\n\n+', content.strip())
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0])
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,\.]\d{3})', lines[1])
            if not time_match:
                continue
            
            start = parse_srt_timestamp(time_match.group(1))
            end = parse_srt_timestamp(time_match.group(2))
            text = '\n'.join(lines[2:])
            
            # Extract speaker tag if present
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
            else:
                speaker = None
            
            segments.append({
                'index': index,
                'start': start,
                'end': end,
                'text': text,
                'speaker': speaker
            })
        except (ValueError, IndexError):
            continue
    
    return segments


def find_gaps(segments: List[dict], min_gap: float = 2.0) -> List[Tuple[float, float, Optional[str]]]:
    """Find gaps between segments longer than min_gap seconds.
    
    Returns list of (start, end, speaker) tuples where speaker is inferred
    from surrounding segments (fallback when embeddings unavailable).
    """
    gaps = []
    
    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']
        gap_duration = next_start - current_end
        
        if gap_duration >= min_gap:
            # Infer speaker from surrounding segments (fallback)
            speaker = segments[i].get('speaker')
            if not speaker:
                speaker = segments[i + 1].get('speaker')
            
            gaps.append((current_end, next_start, speaker))
    
    return gaps


# ── Speaker Embedding Functions ───────────────────────────────────────────────

def _import_nemo_asr():
    """Import nemo.collections.asr, handling the case where local nemo.py exists."""
    import importlib
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    
    def _is_script_dir(entry: str) -> bool:
        try:
            return entry and Path(entry).resolve() == script_dir
        except OSError:
            return False
    
    try:
        sys.path = [e for e in original_path if not _is_script_dir(e)]
        return importlib.import_module("nemo.collections.asr")
    finally:
        sys.path = original_path


def load_speaker_model():
    """Load TitaNet speaker embedding model."""
    nemo_asr = _import_nemo_asr()
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def extract_speaker_embedding(audio_path: Path, model) -> Optional['numpy.ndarray']:
    """Extract speaker embedding from audio file using TitaNet."""
    try:
        import torch
        with torch.no_grad():
            emb = model.get_embedding(str(audio_path))
            # Normalize embedding for cosine similarity
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten()
    except Exception as e:
        log.warning(f"Failed to extract embedding: {e}")
        return None


def compute_speaker_embeddings(
    video_path: Path,
    segments: List[dict],
    model,
    tmpdir: Path,
    max_segments_per_speaker: int = 5
) -> Dict[str, 'numpy.ndarray']:
    """Compute average speaker embeddings from known segments.
    
    For each speaker, extract audio from a few segments and average their embeddings.
    """
    import numpy as np
    
    # Group segments by speaker
    speaker_segments: Dict[str, List[dict]] = {}
    for seg in segments:
        speaker = seg.get('speaker')
        if speaker:
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
    
    speaker_embeddings = {}
    
    for speaker, segs in speaker_segments.items():
        # Take up to max_segments_per_speaker longest segments
        segs = sorted(segs, key=lambda s: s['end'] - s['start'], reverse=True)[:max_segments_per_speaker]
        
        embeddings = []
        for seg in segs:
            # Extract audio for this segment
            audio_path = tmpdir / f"spk_{speaker}_{seg['start']:.0f}.wav"
            if extract_audio_segment(video_path, seg['start'], seg['end'], audio_path):
                emb = extract_speaker_embedding(audio_path, model)
                if emb is not None:
                    embeddings.append(emb)
        
        if embeddings:
            # Average embeddings for this speaker
            avg_emb = np.mean(embeddings, axis=0)
            # Re-normalize
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            speaker_embeddings[speaker] = avg_emb
            log.info(f"  Computed embedding for speaker '{speaker}' from {len(embeddings)} segments")
    
    return speaker_embeddings


def match_speaker_by_embedding(
    audio_path: Path,
    speaker_embeddings: Dict[str, 'numpy.ndarray'],
    model,
    threshold: float = 0.5
) -> Optional[str]:
    """Match gap audio to closest speaker using embeddings.
    
    Returns speaker name if match above threshold, else None.
    """
    import numpy as np
    
    if not speaker_embeddings:
        return None
    
    gap_emb = extract_speaker_embedding(audio_path, model)
    if gap_emb is None:
        return None
    
    # Compute cosine similarity with each speaker
    best_speaker = None
    best_score = -1.0
    
    for speaker, emb in speaker_embeddings.items():
        score = float(np.dot(gap_emb, emb))  # Cosine similarity (embeddings normalized)
        if score > best_score:
            best_score = score
            best_speaker = speaker
    
    if best_score >= threshold:
        log.info(f"    Speaker match: '{best_speaker}' (similarity={best_score:.3f})")
        return best_speaker
    else:
        log.info(f"    No speaker match (best={best_score:.3f} < threshold={threshold})")
        return None


def extract_audio_segment(video_path: Path, start: float, end: float, output_path: Path) -> bool:
    """Extract audio segment from video file."""
    duration = end - start
    cmd = [
        'ffmpeg', '-ss', str(start), '-t', str(duration),
        '-i', str(video_path),
        '-vn', '-ac', '1', '-ar', '16000',
        '-y', str(output_path), '-loglevel', 'error'
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path.exists()
    except subprocess.CalledProcessError:
        return False


def check_audio_energy(audio_path: Path, threshold: float = 500.0) -> Tuple[bool, float]:
    """Check if audio has enough energy to contain speech.
    
    Returns (has_speech, rms_energy).
    RMS < threshold suggests silence/background noise only.
    """
    import wave
    import numpy as np
    
    try:
        with wave.open(str(audio_path), 'rb') as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(float)
        
        if len(audio) == 0:
            return False, 0.0
        
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Check percentage of samples above threshold
        active_ratio = np.sum(np.abs(audio) > threshold) / len(audio)
        
        # If less than 10% of audio is above threshold, likely silence
        has_speech = active_ratio > 0.10
        
        return has_speech, rms
    except Exception:
        return True, 0.0  # Assume speech on error


def transcribe_with_whisper(audio_path: Path, model_name: str = 'base', model=None) -> List[dict]:
    """Transcribe audio via whisper venv subprocess, returning segments with timestamps."""
    import json
    if not _WHISPER_PY:
        log.error("WHISPER_PY env var not set — cannot run whisper")
        return []
    cmd = [_WHISPER_PY, str(_WHISPER_HELPER), str(audio_path), model_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"Whisper subprocess failed: {e.stderr}")
        return []
    except Exception as e:
        log.error(f"Whisper subprocess error: {e}")
        return []


def insert_segments(original: List[dict], gap_start: float, gap_end: float, 
                    new_segments: List[dict], speaker: Optional[str]) -> List[dict]:
    """Insert new segments into the original list, adjusting timestamps and indices."""
    result = []
    insert_index = 0
    
    # Find where to insert
    for i, seg in enumerate(original):
        if seg['end'] <= gap_start:
            result.append(seg)
            insert_index = i + 1
        elif seg['start'] >= gap_end:
            # This segment is after the gap
            break
    
    # Add new segments with adjusted timestamps (relative to gap_start)
    for new_seg in new_segments:
        text = new_seg['text']
        if speaker:
            text = f"[{speaker}] {text}"
        
        result.append({
            'index': 0,  # Will be renumbered
            'start': gap_start + new_seg['start'],
            'end': gap_start + new_seg['end'],
            'text': text,
            'speaker': speaker
        })
    
    # Add remaining original segments
    for seg in original[insert_index:]:
        result.append(seg)
    
    # Renumber all segments
    for i, seg in enumerate(result):
        seg['index'] = i + 1
    
    return result


def write_srt(segments: List[dict], output_path: Path) -> None:
    """Write segments to SRT file."""
    lines = []
    for seg in segments:
        start_ts = format_srt_timestamp(seg['start'])
        end_ts = format_srt_timestamp(seg['end'])
        lines.append(f"{seg['index']}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg['text'])
        lines.append("")
    
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Fill gaps in SRT files using Whisper with speaker embedding matching")
    parser.add_argument("video", help="Video file to extract audio from")
    parser.add_argument("input_srt", help="Input SRT file with gaps")
    parser.add_argument("output_srt", help="Output SRT file with gaps filled")
    parser.add_argument("--min-gap", type=float, default=2.0, 
                        help="Minimum gap duration to fill (seconds, default: 2.0)")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model size (default: base)")
    parser.add_argument("--max-gap", type=float, default=60.0,
                        help="Maximum gap duration to fill (seconds, default: 60)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Disable speaker embedding matching, use proximity-based fallback")
    parser.add_argument("--embedding-threshold", type=float, default=0.5,
                        help="Minimum cosine similarity for speaker match (default: 0.5)")
    args = parser.parse_args()
    
    video_path = Path(args.video).resolve()
    input_srt = Path(args.input_srt).resolve()
    output_srt = Path(args.output_srt).resolve()
    
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not input_srt.exists():
        log.error(f"SRT not found: {input_srt}")
        sys.exit(1)
    
    import json as _json

    checkpoint_path = output_srt.with_suffix('.checkpoint.json')

    # ── Per-run log file ──────────────────────────────────────────────────────
    log_path = output_srt.with_suffix('.log')
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s │ %(levelname)-8s │ %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(file_handler)
    log.info(f"Log: {log_path}")

    # Parse input SRT
    log.info(f"Reading {input_srt}...")
    content = input_srt.read_text(encoding='utf-8')
    segments = parse_srt(content)
    log.info(f"  Found {len(segments)} segments")

    # Find gaps (always from original SRT so the gap list is stable across runs)
    gaps = find_gaps(segments, min_gap=args.min_gap)
    log.info(f"  Found {len(gaps)} gaps >= {args.min_gap}s")

    if not gaps:
        log.info("No gaps to fill.")
        output_srt.write_text(content, encoding='utf-8')
        return

    # ── Resume from checkpoint if available ──────────────────────────────────
    done_gaps: set = set()
    if checkpoint_path.exists():
        try:
            cp = _json.loads(checkpoint_path.read_text(encoding='utf-8'))
            segments = cp['segments']
            done_gaps = set(cp['done_gaps'])
            log.info(f"  Resuming from checkpoint: {len(done_gaps)}/{len(gaps)} gaps already done")
        except Exception as e:
            log.warning(f"  Checkpoint unreadable ({e}) — starting from scratch")
            checkpoint_path.unlink(missing_ok=True)

    # Load speaker embedding model if enabled
    speaker_model = None
    speaker_embeddings = None
    use_embeddings = not args.no_embeddings

    if use_embeddings:
        try:
            log.info("Loading TitaNet speaker embedding model...")
            speaker_model = load_speaker_model()
        except Exception as e:
            log.warning(f"Failed to load speaker model, falling back to proximity: {e}")
            use_embeddings = False

    # Process each gap
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Compute speaker embeddings from known segments (if enabled)
        if use_embeddings and speaker_model:
            log.info("Computing speaker embeddings from known segments...")
            speaker_embeddings = compute_speaker_embeddings(video_path, parse_srt(content), speaker_model, tmpdir)
            if not speaker_embeddings:
                log.warning("No speaker embeddings computed, falling back to proximity")
                use_embeddings = False

        # Verify whisper subprocess is available
        if not _WHISPER_PY:
            log.error("WHISPER_PY env var not set. Pass the whisper venv Python path via WHISPER_PY.")
            sys.exit(1)
        log.info(f"Whisper model: {args.whisper_model} (via subprocess: {_WHISPER_PY})")
        whisper_model = None

        for gap_start, gap_end, fallback_speaker in gaps:
            gap_duration = gap_end - gap_start

            if gap_duration > args.max_gap:
                log.info(f"  Skipping large gap: {gap_start:.1f}s - {gap_end:.1f}s ({gap_duration:.1f}s > max {args.max_gap}s)")
                done_gaps.add(gap_start)
                continue

            if gap_start in done_gaps:
                log.info(f"  ✓ Gap {gap_start:.1f}s already done — skipping")
                continue

            log.info(f"  Processing gap: {gap_start:.1f}s - {gap_end:.1f}s ({gap_duration:.1f}s)")

            # Extract audio
            audio_path = tmpdir / f"gap_{gap_start:.0f}.wav"
            if not extract_audio_segment(video_path, gap_start, gap_end, audio_path):
                log.warning(f"    Failed to extract audio for gap")
                done_gaps.add(gap_start)
                continue

            # Check if audio has speech-like energy
            has_speech, rms = check_audio_energy(audio_path)
            log.info(f"    Audio energy: RMS={rms:.0f}, has_speech={has_speech}")

            if not has_speech:
                log.info(f"    Skipping: likely natural pause (low audio energy)")
                done_gaps.add(gap_start)
                continue

            # Match speaker using embeddings (if available)
            speaker = fallback_speaker
            if use_embeddings and speaker_model and speaker_embeddings:
                matched = match_speaker_by_embedding(
                    audio_path, speaker_embeddings, speaker_model,
                    threshold=args.embedding_threshold
                )
                if matched:
                    speaker = matched
                else:
                    log.info(f"    Using fallback speaker: {fallback_speaker}")

            if speaker:
                log.info(f"    Speaker: {speaker}")

            # Transcribe with Whisper
            log.info(f"    Transcribing with Whisper ({args.whisper_model})...")
            new_segments = transcribe_with_whisper(audio_path, args.whisper_model)

            if not new_segments:
                log.warning(f"    No transcription produced")
                done_gaps.add(gap_start)
                continue

            log.info(f"    Found {len(new_segments)} new segments")
            for ns in new_segments:
                log.info(f"      [{ns['start']:.2f}s → {ns['end']:.2f}s] {ns['text'] or '(empty)'}")

            segments = insert_segments(segments, gap_start, gap_end, new_segments, speaker)
            done_gaps.add(gap_start)

            # Save checkpoint after every gap
            checkpoint_path.write_text(
                _json.dumps({'segments': segments, 'done_gaps': list(done_gaps)}, indent=2),
                encoding='utf-8',
            )

    # Write output and clean up checkpoint
    log.info(f"Writing {output_srt}...")
    write_srt(segments, output_srt)
    checkpoint_path.unlink(missing_ok=True)
    log.info(f"  {len(segments)} total segments")
    log.info("Done!")


if __name__ == "__main__":
    main()
