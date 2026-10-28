"""Package already completed speech results; never overwrite the original."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batch_subtitles import srt_stats


def srt(rows, duration):
    def timestamp(seconds):
        value = round(seconds * 1000)
        return f'{value//3600000:02}:{value//60000%60:02}:{value//1000%60:02},{value%1000:03}'
    blocks = []
    for row in sorted(rows, key=lambda row: (row['start'], row['end'])):
        start, end, text = float(row['start']), float(row['end']), str(row['text']).strip()
        if not text or start < 0 or end <= start or end > duration + 0.1:
            raise ValueError('Invalid subtitle cue')
        blocks.append(f'{len(blocks)+1}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n')
    result = '\n'.join(blocks).encode('utf-8')
    srt_stats(result, duration)
    return result


def package(index):
    work = ROOT / 'modal_control/batch_inputs'
    meta = json.loads((work / f'video_{index:02d}.json').read_text())
    speech = json.loads((work / f'video_{index:02d}.gpu_full.json').read_text())
    variant = 'cpu_v2_full' if (work / f'video_{index:02d}.cpu_v2_full.json').exists() else 'cpu_full'
    dub = json.loads((work / f'video_{index:02d}.{variant}.json').read_text())
    original = ROOT / meta['source']
    audio = work / f'video_{index:02d}.{variant}.m4a'
    if speech['status'] != 'complete' or dub['status'] != 'complete':
        raise ValueError('Both model stages must have completed')
    digest = hashlib.sha256()
    with original.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != meta['source_sha256']:
        raise ValueError('Original video changed after input extraction')
    duration = speech['duration_seconds']
    destinations = {lang: original.with_name(original.stem + f'.{lang}.srt') for lang in ('en', 'de', 'fr')}
    video = original.with_name(original.stem + '.german-dub.mp4')
    contents = {'en': srt(speech['english'], duration),
                'de': srt(speech['translations']['de'], duration),
                'fr': srt(speech['translations']['fr'], duration)}
    if video.exists() or any(path.exists() and path.read_bytes() != contents[lang] for lang, path in destinations.items()):
        raise FileExistsError('A conflicting output file already exists')
    with tempfile.TemporaryDirectory(prefix='dub-package-', dir=original.parent) as tmp:
        tmp = Path(tmp)
        staged = tmp / 'output.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(original), '-i', str(audio),
                        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'copy',
                        '-metadata:s:a:0', 'language=deu', '-movflags', '+faststart', str(staged)], check=True)
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration:stream=codec_type', '-of', 'json', str(staged)]))
        if not {'audio', 'video'} <= {s['codec_type'] for s in info['streams']}:
            raise ValueError('Dubbed video lacks an audio or video stream')
        if abs(float(info['format']['duration']) - meta['duration_seconds']) > 1.0:
            raise ValueError('Dubbed video duration differs from original')
        # Hard-link exclusively; never replace a pre-existing destination.
        os.link(staged, video)
        for lang, destination in destinations.items():
            if not destination.exists():
                with destination.open('xb') as out:
                    out.write(contents[lang])
    return {'video': str(video), 'subtitles': {k: str(v) for k, v in destinations.items()}}


if __name__ == '__main__':
    print(json.dumps(package(int(sys.argv[1])), ensure_ascii=False, indent=2))
