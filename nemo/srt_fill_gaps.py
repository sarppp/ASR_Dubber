"""
srt_fill_gaps.py — Fill gaps in SRT files using Whisper

Detects gaps between subtitle segments, transcribes missing audio with Whisper,
and inserts new segments with speaker attribution from surrounding context.

Usage:
  cd nemo && uv run python srt_fill_gaps.py video.mp4 input.srt output.srt --min-gap 2.0
  cd nemo && uv run python srt_fill_gaps.py video.mp4 input.srt output.srt --whisper-model base
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


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
    from surrounding segments.
    """
    gaps = []
    
    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']
        gap_duration = next_start - current_end
        
        if gap_duration >= min_gap:
            # Infer speaker from surrounding segments
            # Use the speaker of the segment before the gap
            speaker = segments[i].get('speaker')
            
            # If no speaker before, try the one after
            if not speaker:
                speaker = segments[i + 1].get('speaker')
            
            gaps.append((current_end, next_start, speaker))
    
    return gaps


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


def transcribe_with_whisper(audio_path: Path, model_name: str = 'base') -> List[dict]:
    """Transcribe audio with Whisper, returning segments with timestamps."""
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed. Run: uv pip install openai-whisper")
        return []
    
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language='en')
    
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip()
        })
    
    return segments


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
    parser = argparse.ArgumentParser(description="Fill gaps in SRT files using Whisper")
    parser.add_argument("video", help="Video file to extract audio from")
    parser.add_argument("input_srt", help="Input SRT file with gaps")
    parser.add_argument("output_srt", help="Output SRT file with gaps filled")
    parser.add_argument("--min-gap", type=float, default=2.0, 
                        help="Minimum gap duration to fill (seconds, default: 2.0)")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model size (default: base)")
    parser.add_argument("--max-gap", type=float, default=60.0,
                        help="Maximum gap duration to fill (seconds, default: 60)")
    args = parser.parse_args()
    
    video_path = Path(args.video).resolve()
    input_srt = Path(args.input_srt).resolve()
    output_srt = Path(args.output_srt).resolve()
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    if not input_srt.exists():
        print(f"Error: SRT not found: {input_srt}")
        sys.exit(1)
    
    # Parse input SRT
    print(f"Reading {input_srt}...")
    content = input_srt.read_text(encoding='utf-8')
    segments = parse_srt(content)
    print(f"  Found {len(segments)} segments")
    
    # Find gaps
    gaps = find_gaps(segments, min_gap=args.min_gap)
    print(f"  Found {len(gaps)} gaps >= {args.min_gap}s")
    
    if not gaps:
        print("No gaps to fill.")
        output_srt.write_text(content, encoding='utf-8')
        return
    
    # Process each gap
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for gap_start, gap_end, speaker in gaps:
            gap_duration = gap_end - gap_start
            
            if gap_duration > args.max_gap:
                print(f"  Skipping large gap: {gap_start:.1f}s - {gap_end:.1f}s ({gap_duration:.1f}s > max {args.max_gap}s)")
                continue
            
            print(f"  Processing gap: {gap_start:.1f}s - {gap_end:.1f}s ({gap_duration:.1f}s)")
            if speaker:
                print(f"    Speaker: {speaker}")
            
            # Extract audio
            audio_path = tmpdir / f"gap_{gap_start:.0f}.wav"
            if not extract_audio_segment(video_path, gap_start, gap_end, audio_path):
                print(f"    Warning: Failed to extract audio for gap")
                continue
            
            # Check if audio has speech-like energy
            has_speech, rms = check_audio_energy(audio_path)
            print(f"    Audio energy: RMS={rms:.0f}, has_speech={has_speech}")
            
            if not has_speech:
                print(f"    Skipping: likely natural pause (low audio energy)")
                continue
            
            # Transcribe with Whisper
            print(f"    Transcribing with Whisper ({args.whisper_model})...")
            new_segments = transcribe_with_whisper(audio_path, args.whisper_model)
            
            if not new_segments:
                print(f"    Warning: No transcription produced")
                continue
            
            print(f"    Found {len(new_segments)} new segments")
            
            # Insert into segments list
            segments = insert_segments(segments, gap_start, gap_end, new_segments, speaker)
    
    # Write output
    print(f"\nWriting {output_srt}...")
    write_srt(segments, output_srt)
    print(f"  {len(segments)} total segments")
    print("Done!")


if __name__ == "__main__":
    main()
