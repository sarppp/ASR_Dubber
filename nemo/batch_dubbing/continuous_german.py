"""One narration request for the original ten German display cues."""
import hashlib
import json
import re
from pathlib import Path

MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
REVISION = 'fd4b254389122332181a7c3db7f27e918eec64e3'
MAX_TOKENS = 600


def passage(path):
    blocks = re.split(r'\n\s*\n', Path(path).read_text(encoding='utf-8-sig').strip())
    texts = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].isdigit() or ' --> ' not in lines[1]:
            raise ValueError('Invalid SRT cue')
        texts.append(' '.join(lines[2:]))
    if len(texts) != 10:
        raise ValueError('Exactly ten original cues required')
    text = ' '.join(texts)
    if not 100 < len(text) < 1600:
        raise ValueError('Passage outside bounded scope')
    return text


def synthesize(model, script, reference):
    # SRT boundaries never become synthesis boundaries. No added punctuation.
    return model.generate_voice_clone(text=passage(script), language='German',
        ref_audio=str(reference), xvec_only=True, max_new_tokens=MAX_TOKENS)


def save_result(result, destination):
    import io
    import wave
    destination = Path(destination)
    report = result['report']
    data = result['audio']
    if (report.get('status') != 'complete' or report.get('synthesis_calls') != 1
            or len(data) > 3_000_000 or hashlib.sha256(data).hexdigest() != report['audio_sha256']):
        raise ValueError('Invalid continuous take result')
    with wave.open(io.BytesIO(data)) as audio:
        duration = audio.getnframes() / audio.getframerate()
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
            raise ValueError('Unexpected WAV format')
        if not 10 < duration < 49 or abs(duration - report['audio_seconds']) > .001:
            raise ValueError('Truncated, capped or overlong take')
        if not any(audio.readframes(audio.getnframes())):
            raise ValueError('Silent take')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'CONTINUOUS_GERMAN.wav').write_bytes(data)
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


def package(source, audio, subtitles, destination):
    """Mux the whole native take; hold the final picture only if it runs longer."""
    import subprocess
    import wave
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    video_seconds = float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration','-of','default=nw=1:nk=1',str(source)], text=True, timeout=15))
    with wave.open(str(audio)) as wav:
        audio_seconds = wav.getnframes()/wav.getframerate()
    duration = max(video_seconds,audio_seconds)
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-i',str(source),
        '-i',str(audio),'-i',str(subtitles),'-map','0:v:0','-map','1:a:0','-map','2:0',
        '-vf',f'tpad=stop_mode=clone:stop_duration={max(0,audio_seconds-video_seconds)+.1}',
        '-c:v','libx264','-preset','veryfast','-crf','20','-c:a','aac','-b:a','192k',
        '-c:s','mov_text','-metadata:s:a:0','language=deu','-metadata:s:s:0','language=deu',
        '-t',str(duration),'-movflags','+faststart',str(destination)]
    subprocess.run(command,check=True,timeout=90)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(destination),'-f','null','-'],check=True,timeout=60)
    return dict(source_seconds=video_seconds,audio_seconds=audio_seconds,
        picture_hold_seconds=max(0,audio_seconds-video_seconds),
        subtitle_timing='Original input.de.srt display times; speech alignment not verified',
        audio_transform='AAC encoding only; no speech cuts, speed changes or inserted pauses')
