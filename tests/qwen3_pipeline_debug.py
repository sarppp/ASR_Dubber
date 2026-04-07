#!/usr/bin/env python3
"""
qwen3_pipeline_debug.py - Debug and fix ASR/TTS pipeline issues

This script helps diagnose and fix common issues in the Qwen3 dubbing pipeline:
1. Word alignment problems (apostrophes, special characters)
2. Silent gaps and TTS timeouts
3. Speaker diarization errors
4. Voice assignment mismatches

Usage:
    python qwen3_pipeline_debug.py input.srt --output debug_output --verbose
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

log = logging.getLogger(__name__)

def load_srt_file(srt_path: Path) -> List[Dict]:
    """Load SRT file and parse segments"""
    segments = []
    current_segment = {}
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            # Segment number
            current_segment['index'] = int(line)
            i += 1
            
            # Timestamp
            if i < len(lines):
                timestamp_line = lines[i].strip()
                start_str, end_str = timestamp_line.split(' --> ')
                current_segment['start'] = _parse_srt_time(start_str)
                current_segment['end'] = _parse_srt_time(end_str)
                i += 1
            
            # Text (may span multiple lines)
            text_lines = []
            while i < len(lines) and lines[i].strip() != '':
                text_lines.append(lines[i].strip())
                i += 1
            current_segment['text'] = ' '.join(text_lines)
            
            segments.append(current_segment)
            current_segment = {}
        else:
            i += 1
    
    return segments

def _parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp HH:MM:SS,mmm to seconds"""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def diagnose_word_alignment(segments: List[Dict]) -> List[Dict]:
    """Diagnose word alignment issues"""
    issues = []
    
    for seg in segments:
        text = seg.get('text', '')
        
        # Check for apostrophe issues (especially French)
        if "'" in text:
            # Specific French apostrophe patterns that cause issues
            french_apostrophes = ["l'", "d'", "qu'", "t'", "j'", "m'", "n'", "s'"]
            for pattern in french_apostrophes:
                if pattern in text.lower():
                    issues.append({
                        'type': 'french_apostrophe',
                        'segment': seg['index'],
                        'text': text,
                        'start': seg['start'],
                        'end': seg['end'],
                        'pattern': pattern,
                        'issue': f'French apostrophe "{pattern}" may cause TTS alignment issues',
                        'fix': f'Consider replacing "{pattern}" with expanded form or use TTS preprocessing',
                        'timestamp': _format_timestamp(seg['start'])
                    })
        
        # Check for multiple apostrophes in one segment
        apostrophe_count = text.count("'")
        if apostrophe_count > 1:
            issues.append({
                'type': 'multiple_apostrophes',
                'segment': seg['index'],
                'text': text,
                'start': seg['start'],
                'end': seg['end'],
                'count': apostrophe_count,
                'issue': f'Multiple apostrophes ({apostrophe_count}) may cause alignment problems',
                'fix': 'Consider splitting segment or preprocessing apostrophes',
                'timestamp': _format_timestamp(seg['start'])
            })
        
        # Check for very short segments that might be missed
        duration = seg['end'] - seg['start']
        if duration < 0.5 and len(text.strip()) > 0:
            issues.append({
                'type': 'short_segment',
                'segment': seg['index'],
                'text': text,
                'start': seg['start'],
                'end': seg['end'],
                'duration': duration,
                'issue': 'Very short segment may be skipped by TTS',
                'fix': 'Consider merging with adjacent segments or adjusting timing',
                'timestamp': _format_timestamp(seg['start'])
            })
        
        # Check for very long segments that might timeout
        if duration > 15.0:
            issues.append({
                'type': 'long_segment',
                'segment': seg['index'],
                'text': text[:100] + '...' if len(text) > 100 else text,
                'start': seg['start'],
                'end': seg['end'],
                'duration': duration,
                'issue': 'Very long segment may cause TTS timeout',
                'fix': 'Consider splitting into smaller segments',
                'timestamp': _format_timestamp(seg['start'])
            })
    
    return issues

def diagnose_silent_gaps(segments: List[Dict]) -> List[Dict]:
    """Find suspicious gaps where TTS might fail"""
    issues = []
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        gap = next_seg['start'] - current['end']
        
        # Look for medium gaps (2-30 seconds) that might have missed content
        if 2.0 < gap < 30.0:
            issues.append({
                'type': 'suspicious_gap',
                'between_segments': (current['index'], next_seg['index']),
                'gap_duration': gap,
                'current_end': current['end'],
                'next_start': next_seg['start'],
                'current_text': current['text'][:50] + '...' if len(current['text']) > 50 else current['text'],
                'next_text': next_seg['text'][:50] + '...' if len(next_seg['text']) > 50 else next_seg['text'],
                'issue': f'Gap of {gap:.1f}s might contain missed content or TTS failure',
                'fix': 'Check original audio for content in this gap - TTS may have failed',
                'start_timestamp': _format_timestamp(current['end']),
                'end_timestamp': _format_timestamp(next_seg['start'])
            })
    
    return issues

def diagnose_speaker_issues(segments: List[Dict]) -> List[Dict]:
    """Diagnose speaker assignment issues"""
    issues = []
    speakers = set()
    
    # Collect all speakers
    for seg in segments:
        if 'speaker' in seg:
            speakers.add(seg['speaker'])
    
    # Check for speaker inconsistencies
    speaker_texts = {}
    speaker_durations = {}
    
    for seg in segments:
        speaker = seg.get('speaker', 'unknown')
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
            speaker_durations[speaker] = []
        
        speaker_texts[speaker].append(seg['text'])
        speaker_durations[speaker].append(seg['end'] - seg['start'])
    
    # Look for patterns that might indicate wrong speaker assignment
    for speaker, texts in speaker_texts.items():
        # Check for very short utterances (might be misassigned)
        short_utterances = [t for t in texts if len(t.strip()) < 5]
        total_utterances = len(texts)
        
        if total_utterances > 0:
            short_ratio = len(short_utterances) / total_utterances
            if short_ratio > 0.2:  # More than 20% short utterances
                avg_duration = sum(speaker_durations[speaker]) / len(speaker_durations[speaker])
                
                issues.append({
                    'type': 'speaker_suspicious_pattern',
                    'speaker': speaker,
                    'short_utterance_ratio': short_ratio,
                    'total_utterances': total_utterances,
                    'short_utterances': short_utterances[:5],  # Show first 5
                    'avg_duration': avg_duration,
                    'issue': f'Speaker {speaker} has {short_ratio:.1%} short utterances (avg {avg_duration:.1f}s), possible misassignment',
                    'fix': 'Review diarization parameters - short utterances like "et" may be misassigned'
                })
        
        # Check for single-word utterances that might be errors
        single_words = [t for t in texts if len(t.strip().split()) == 1 and len(t.strip()) < 10]
        if len(single_words) > 3:  # More than 3 single-word utterances
            issues.append({
                'type': 'single_word_pattern',
                'speaker': speaker,
                'single_words': single_words,
                'count': len(single_words),
                'issue': f'Speaker {speaker} has {len(single_words)} single-word utterances, likely diarization errors',
                'fix': 'Consider manual correction or diarization parameter tuning'
            })
    
    return issues

def fix_french_apostrophes(text: str) -> str:
    """Fix French apostrophes for better TTS processing"""
    replacements = {
        "l'": "le ",      # l'heure -> le heure
        "d'": "de ",      # d'abord -> de abord  
        "qu'": "que ",    # qu'il -> que il
        "t'": "te ",      # t'aime -> te aime
        "j'": "je ",      # j'ai -> je ai
        "m'": "me ",      # m'appelle -> me appelle
        "n'": "ne ",      # n'est -> ne est
        "s'": "se ",      # s'est -> se est
    }
    
    result = text
    for pattern, replacement in replacements.items():
        # Case-insensitive replacement
        pattern_regex = re.compile(re.escape(pattern), re.IGNORECASE)
        result = pattern_regex.sub(replacement, result)
    
    return result

def generate_fixed_srt(segments: List[Dict], output_path: Path, fix_apostrophes: bool = False) -> None:
    """Generate a fixed SRT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            
            start_time = _format_srt_timestamp(seg['start'])
            end_time = _format_srt_timestamp(seg['end'])
            f.write(f"{start_time} --> {end_time}\n")
            
            text = seg['text']
            if fix_apostrophes:
                text = fix_french_apostrophes(text)
            
            f.write(f"{text}\n\n")

def _format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_fixes(issues: List[Dict], output_dir: Path, segments: List[Dict]) -> None:
    """Generate fix scripts and fixed files"""
    output_dir.mkdir(exist_ok=True)
    
    # Group issues by type
    by_type = {}
    for issue in issues:
        issue_type = issue['type']
        if issue_type not in by_type:
            by_type[issue_type] = []
        by_type[issue_type].append(issue)
    
    # Generate fixed SRT with apostrophe preprocessing
    if 'french_apostrophe' in by_type or 'multiple_apostrophes' in by_type:
        fixed_srt = output_dir / "fixed_apostrophes.srt"
        generate_fixed_srt(segments, fixed_srt, fix_apostrophes=True)
        log.info(f"Generated SRT with fixed apostrophes: {fixed_srt}")
    
    # Generate segment merge suggestions
    if 'short_segment' in by_type:
        merge_suggestions = output_dir / "segment_merges.txt"
        with open(merge_suggestions, 'w') as f:
            f.write("Suggested Segment Merges:\n")
            f.write("=" * 40 + "\n\n")
            
            short_segments = by_type['short_segment']
            for seg_issue in short_segments:
                seg_idx = seg_issue['segment']
                f.write(f"Segment {seg_idx} (duration: {seg_issue['duration']:.2f}s):\n")
                f.write(f"  Text: '{seg_issue['text']}'\n")
                f.write(f"  Suggestion: Merge with adjacent segment\n\n")
        
        log.info(f"Generated merge suggestions: {merge_suggestions}")
    
    # Generate gap analysis
    if 'suspicious_gap' in by_type:
        gap_analysis = output_dir / "gap_analysis.txt"
        with open(gap_analysis, 'w') as f:
            f.write("Silent Gap Analysis:\n")
            f.write("=" * 40 + "\n\n")
            
            for gap_issue in by_type['suspicious_gap']:
                f.write(f"Gap between segments {gap_issue['between_segments'][0]} and {gap_issue['between_segments'][1]}:\n")
                f.write(f"  Duration: {gap_issue['gap_duration']:.1f}s\n")
                f.write(f"  Time: {gap_issue['start_timestamp']} -> {gap_issue['end_timestamp']}\n")
                f.write(f"  Context: '{gap_issue['current_text']}' -> '{gap_issue['next_text']}'\n")
                f.write(f"  Issue: {gap_issue['issue']}\n")
                f.write(f"  Fix: {gap_issue['fix']}\n\n")
        
        log.info(f"Generated gap analysis: {gap_analysis}")
    
    # Generate speaker correction suggestions
    if 'speaker_suspicious_pattern' in by_type or 'single_word_pattern' in by_type:
        speaker_fixes = output_dir / "speaker_corrections.txt"
        with open(speaker_fixes, 'w') as f:
            f.write("Speaker Assignment Corrections:\n")
            f.write("=" * 40 + "\n\n")
            
            for speaker_issue in by_type.get('speaker_suspicious_pattern', []):
                f.write(f"Speaker {speaker_issue['speaker']}:\n")
                f.write(f"  Issue: {speaker_issue['issue']}\n")
                f.write(f"  Short utterances ({len(speaker_issue['short_utterances'])}): {speaker_issue['short_utterances']}\n")
                f.write(f"  Fix: {speaker_issue['fix']}\n\n")
            
            for single_word_issue in by_type.get('single_word_pattern', []):
                f.write(f"Speaker {single_word_issue['speaker']} - Single Words:\n")
                f.write(f"  Issue: {single_word_issue['issue']}\n")
                f.write(f"  Single words: {single_word_issue['single_words']}\n")
                f.write(f"  Fix: {single_word_issue['fix']}\n\n")
        
        log.info(f"Generated speaker corrections: {speaker_fixes}")

def main():
    parser = argparse.ArgumentParser(description="Diagnose Qwen3 ASR/TTS pipeline issues")
    parser.add_argument("srt_file", help="SRT file to analyze")
    parser.add_argument("--output", "-o", default="debug_output", help="Output directory for fixes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix-apostrophes", action="store_true", help="Generate SRT with fixed apostrophes")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    srt_path = Path(args.srt_file)
    if not srt_path.exists():
        log.error(f"SRT file not found: {srt_path}")
        return 1
    
    output_dir = Path(args.output)
    
    # Load and analyze SRT
    log.info(f"Loading SRT file: {srt_path}")
    segments = load_srt_file(srt_path)
    log.info(f"Loaded {len(segments)} segments")
    
    # Run diagnostics
    log.info("🔍 Running word alignment diagnostics...")
    alignment_issues = diagnose_word_alignment(segments)
    
    log.info("🔍 Analyzing silent gaps...")
    gap_issues = diagnose_silent_gaps(segments)
    
    log.info("🔍 Analyzing speaker assignments...")
    speaker_issues = diagnose_speaker_issues(segments)
    
    all_issues = alignment_issues + gap_issues + speaker_issues
    
    # Report findings
    log.info(f"\n📊 Found {len(all_issues)} total issues:")
    log.info(f"  - Word alignment: {len(alignment_issues)}")
    log.info(f"  - Silent gaps: {len(gap_issues)}")
    log.info(f"  - Speaker issues: {len(speaker_issues)}")
    
    # Print specific issues you mentioned
    print("\n🎯 Specific Issues Analysis:")
    
    # Check for "L'horloge" type issues
    french_apostrophe_issues = [i for i in alignment_issues if i['type'] == 'french_apostrophe']
    if french_apostrophe_issues:
        print(f"\n📝 French Apostrophe Issues ({len(french_apostrophe_issues)}):")
        for issue in french_apostrophe_issues[:3]:  # Show first 3
            print(f"  Segment {issue['segment']} at {issue['timestamp']}: '{issue['text']}'")
            print(f"    Pattern: {issue['pattern']}")
            print(f"    Fix: {issue['fix']}")
    
    # Check for gaps around 39:02-39:26
    gap_39 = [i for i in gap_issues if 39*60 <= i['current_end'] <= 39.5*60]
    if gap_39:
        print(f"\n⏱️  Gap around 39:00-39:30:")
        for gap in gap_39:
            print(f"  Gap of {gap['gap_duration']:.1f}s from {gap['start_timestamp']} to {gap['end_timestamp']}")
            print(f"  Context: '{gap['current_text']}' -> '{gap['next_text']}'")
    
    # Check for speaker assignment issues
    if speaker_issues:
        print(f"\n👥 Speaker Assignment Issues ({len(speaker_issues)}):")
        for issue in speaker_issues[:3]:  # Show first 3
            print(f"  Speaker {issue['speaker']}: {issue['issue']}")
            if 'short_utterances' in issue:
                print(f"    Examples: {issue['short_utterances'][:3]}")
    
    # Generate fixes
    if all_issues:
        log.info(f"\n🔧 Generating fixes in: {output_dir}")
        generate_fixes(all_issues, output_dir, segments)
        
        # Save detailed report
        report_file = output_dir / "debug_report.json"
        with open(report_file, 'w') as f:
            json.dump(all_issues, f, indent=2, ensure_ascii=False)
        log.info(f"📄 Detailed report saved: {report_file}")
        
        print(f"\n✅ Debug complete! Check {output_dir}/ for:")
        print(f"  - debug_report.json (detailed issues)")
        print(f"  - fixed_apostrophes.srt (if apostrophe issues found)")
        print(f"  - gap_analysis.txt (silent gap analysis)")
        print(f"  - speaker_corrections.txt (speaker assignment fixes)")
        print(f"  - segment_merges.txt (short segment suggestions)")
    else:
        log.info("🎉 No issues found!")
    
    return 0

if __name__ == "__main__":
    exit(main())
