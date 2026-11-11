"""Stream native speech at absolute source positions; never shorten the picture."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import wave

RATE=24000


def plan_clips(clips,duration):
    """Validate every clip before writing. Durations are exact PCM sample counts."""
    if not math.isfinite(duration) or not 0<duration<=24*3600:
        raise ValueError('Invalid video duration')
    total=round(duration*RATE);plan=[];previous_end=0
    if not clips:raise ValueError('No speech supplied')
    for i,clip in enumerate(clips):
        start,end,frames=clip['start'],clip['end'],clip['frames']
        if (not all(math.isfinite(x) for x in (start,end)) or not 0<=start<end<=duration+.001
                or type(frames) is not int or frames<=0):
            raise ValueError('Invalid speech interval or sample count')
        position=round(start*RATE)
        next_start=clips[i+1]['start'] if i+1<len(clips) else duration
        limit=min(total,round(min(end+.25,next_start)*RATE))
        if position<previous_end or position+frames>limit:
            raise ValueError(f'Speech overlaps or exceeds its source deadline at {start:.3f}s; nothing shifted or cut')
        plan.append(dict(**clip,start_frame=position,end_frame=position+frames,gap_frames=position-previous_end))
        previous_end=position+frames
    return dict(sample_rate=RATE,total_frames=total,clips=plan,trailing_frames=total-previous_end)


def video_duration(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],text=True,timeout=30))
    video=next(s for s in data['streams'] if s['codec_type']=='video')
    return float(video.get('duration',data['format']['duration']))


def packet_hash(path):
    return subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'],text=True,timeout=600).strip()


def render(final_files,video_path,output_dir,temp_dir,background=None,trim_to=None,pad_audio=True):
    duration=video_duration(video_path)
    if trim_to is not None:
        raise ValueError('Implicit output video trimming is forbidden; an explicitly selected excerpt must be supplied as the input video')
    clips=[]
    for path,start,end in final_files:
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(f'Missing speech: {path}; refusing partial output')
        with wave.open(str(path)) as wav:
            if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,RATE):
                raise ValueError(f'Unexpected PCM format: {path}')
            clips.append(dict(file=str(path.resolve()),start=start,end=end,frames=wav.getnframes()))
    plan=plan_clips(clips,duration)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_dir,prefix='anchored-') as temporary:
        work=Path(temporary);track=work/'dub_track.wav'
        with wave.open(str(track),'wb') as target:
            target.setparams((1,2,RATE,0,'NONE','not compressed'))
            def silence(frames):
                while frames:
                    n=min(frames,RATE);target.writeframesraw(bytes(n*2));frames-=n
            for clip in plan['clips']:
                silence(clip['gap_frames'])
                with wave.open(clip['file']) as source:
                    while block:=source.readframes(RATE):target.writeframesraw(block)
            if pad_audio:silence(plan['trailing_frames'])
        movie=work/'final_dub.mp4'
        command=['ffmpeg','-nostdin','-v','error','-n','-i',str(video_path),'-i',str(track)]
        if background:
            if not Path(background).is_file():raise FileNotFoundError(background)
            command+=['-i',str(background),'-filter_complex','[1:a]volume=1.5[v];[2:a]volume=0.4[b];[v][b]amix=inputs=2:duration=first[out]',
                      '-map','0:v','-map','[out]']
        else:command+=['-map','0:v','-map','1:a:0']
        command+=['-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(movie)]
        subprocess.run(command,check=True,timeout=600)
        if packet_hash(movie)!=packet_hash(video_path):raise ValueError('Video packet identity changed')
        if abs(video_duration(movie)-duration)>.05:raise ValueError('Video duration changed')
        actual=[(r['start_frame']/RATE,r['end_frame']/RATE,r['start'],r['end']) for r in plan['clips']]
        report=dict(status='complete',duration_seconds=duration,total_frames=plan['total_frames'],
                    video_packets_unchanged=True,tempo=1.,placements=actual,clip_count=len(clips),audio_padded_to_video=pad_audio,
                    note='Native speech only, exact absolute starts, full source video preserved; this is not a semantic-quality check')
        track.replace(output_dir/'dub_track.wav');movie.replace(output_dir/'final_dub.mp4')
        (output_dir/'timeline_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return output_dir/'final_dub.mp4',actual


def validate_end_timing(sidecar,start,end,max_gap=.4):
    """Do not hide short speech behind a successful maximum-duration fit."""
    edges=sidecar['edges'];tempo=sidecar['tempo']
    if tempo!=1.:raise ValueError('Source-locked mode requires native speech; no tempo compensation')
    if 'sample_rate' not in edges or 'last_active_end_sample' not in edges:
        raise ValueError('Missing measured speech-edge evidence')
    audible_end=(edges['last_active_end_sample']-edges['start_sample'])/edges['sample_rate']
    gap=end-start-audible_end
    if gap>max_gap:
        raise ValueError(f'Speech ends {gap:.3f}s before its source passage; needs delivery/content review, not slowing or padding')
    return gap
