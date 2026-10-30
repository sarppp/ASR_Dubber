"""Publish this test's SRTs and copy-video German dub after all model stages pass."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'subtitle_divider'))
from bilingual_subtitles import Cue, write_subtitles, validate_srt_layout
from batch_dubbing.phrase_trial import sha256, write_json


def publish(trial):
    trial = Path(trial)
    started = time.monotonic()
    data = json.loads((trial / 'translated.json').read_text())
    dub = json.loads((trial / 'dub.json').read_text())
    manifest = json.loads((trial / 'input/manifest.json').read_text())
    if data['status'] != 'complete' or dub['status'] != 'complete':
        raise ValueError('Do not publish an incomplete test')
    if data['input_sha256'] != sha256(trial / 'asr.json') or dub['input_sha256'] != sha256(trial / 'translated.json'):
        raise ValueError('Model outputs do not match their source artifacts')
    original = Path(manifest['source'])
    if sha256(original) != manifest['source_sha256']:
        raise ValueError('Original media changed')
    if manifest.get('clip_start_seconds') != 0 or manifest.get('clip_duration_seconds') != 600 or data['duration_seconds'] != 600:
        raise ValueError('Publishing is limited to the first ten minutes')
    target = trial / 'output'
    target.mkdir(exist_ok=False)
    rows = data['phrases']
    stats = []
    for languages in [('en',), ('de',), ('fr',), ('de', 'fr'), ('en', 'de', 'fr')]:
        cues = [Cue(r['start_ms'], r['end_ms'], '\n'.join(r[l] for l in languages)) for r in rows]
        validate_srt_layout(cues, font_size=28, max_height_percent=20)
        filename = target / ('CRISPR-test.' + '-'.join(languages) + '.srt')
        write_subtitles(filename, cues, font_size=28, max_height_percent=20, ass_preview=True)
        stats.append({'languages': list(languages), 'cues': len(cues), 'srt': str(filename)})
    video = target / 'CRISPR-test.german-dub.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(original),
                    '-i', str(trial / 'german-dub.m4a'), '-map', '0:v:0', '-map', '1:a:0',
                    '-t', '600',
                    '-c:v', 'copy', '-c:a', 'copy', '-metadata:s:a:0', 'language=deu',
                    '-movflags', '+faststart', str(video)], check=True, timeout=180)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                      'format=duration:stream=codec_type', '-of', 'json', str(video)], timeout=20))
    if abs(float(info['format']['duration']) - data['duration_seconds']) > 1:
        raise ValueError('Dubbed media duration mismatch')
    if not {'video', 'audio'} <= {s['codec_type'] for s in info['streams']}:
        raise ValueError('Incomplete video streams')
    report = {'status': 'complete', 'subtitles': stats, 'video': str(video),
              'video_sha256': sha256(video), 'packaging_seconds': time.monotonic()-started,
              'short_cues_under_1_5_seconds': [r['id'] for r in rows if r['end_ms']-r['start_ms'] < 1500],
              'semantic_quality_verified': False, 'listening_review_complete': False,
              'inference': 'All model inference on Modal; assembly local'}
    write_json(target / 'validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial', type=Path)
    print(json.dumps(publish(parser.parse_args().trial), indent=2))
