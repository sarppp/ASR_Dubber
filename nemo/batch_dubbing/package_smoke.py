"""Package the explicitly requested 60-second smoke-test outputs."""
import json
import subprocess
from pathlib import Path
from package_video import srt

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'modal_control/batch_inputs'


def main():
    source = ROOT / json.loads((WORK / 'smoke.json').read_text())['source']
    speech = json.loads((WORK / 'gpu_smoke.json').read_text())
    audio_report = json.loads((WORK / 'cpu_v2_smoke.json').read_text())
    if speech['status'] != 'complete' or audio_report['status'] != 'complete':
        raise ValueError('Both smoke stages must pass before packaging')
    duration = speech['duration_seconds']
    outputs = {lang: WORK / f'smoke.{lang}.srt' for lang in ('en', 'de', 'fr')}
    video = WORK / 'smoke.german-dub.mp4'
    if any(p.exists() for p in [video, *outputs.values()]):
        raise FileExistsError('Smoke output exists; will not overwrite')
    captions = {'en': srt(speech['english'], duration),
                'de': srt(speech['translations']['de'], duration),
                'fr': srt(speech['translations']['fr'], duration)}
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(source),
                    '-i', str(WORK / 'cpu_v2_smoke.wav'), '-map', '0:v:0', '-map', '1:a:0',
                    '-t', str(duration), '-c:v', 'copy', '-c:a', 'aac',
                    '-metadata:s:a:0', 'language=deu', str(video)], check=True)
    for lang, path in outputs.items():
        with path.open('xb') as f:
            f.write(captions[lang])
    print(json.dumps({'video': str(video), 'subtitles': {k: str(v) for k, v in outputs.items()}}))


if __name__ == '__main__':
    main()
