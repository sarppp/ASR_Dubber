"""Review prepared dialogue and verified native WAV results without re-encoding."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from batch_dubbing.dialogue_trial import verify_sources,jobs
from batch_dubbing.native_excerpt_probe import digest,input_digest,decode_wave


def write_review(input_file,output_file,*,root,report_file=None):
    root=Path(root).resolve();output=Path(output_file).resolve()
    data=verify_sources(json.loads(Path(input_file).read_text()),root)
    records={};report_path=None
    if report_file:
        report_path=Path(report_file).resolve();report=json.loads(report_path.read_text())
        if (report.get('status')!='complete' or report.get('stage')!='workload' or report.get('input')!=data
                or report.get('input_sha256')!=input_digest(data) or report.get('playback_rate')!=1.
                or report.get('synthesis_speed')!=1. or report.get('speech_trimmed') is not False
                or report.get('waveform_time_stretched') is not False):
            raise ValueError('Complete matching native trial report required')
        expected=jobs(data,lambda text,size:[text])
        if len(report.get('samples',[]))!=len(expected):raise ValueError('Missing or extra trial samples')
        for want,row in zip(expected,report['samples']):
            if any(row.get(k)!=v for k,v in want.items()) or row.get('file')!=want['id']+'.wav':
                raise ValueError('Trial sample identity changed')
            wav=(report_path.parent/row['file']).read_bytes();payload=decode_wave(wav)
            if digest(wav)!=row.get('sha256') or digest(payload)!=row.get('samples_sha256') or len(payload)//4!=row.get('native_frames'):
                raise ValueError('Trial WAV integrity check failed')
            records[row['id']]=row
    esc=html.escape
    def url(path):return esc(quote(os.path.relpath(path,output.parent),safe='/'))
    sections=[]
    continuous=data.get('review_mode')=='continuous_before_after'
    for pair in data['pairs']:
        start,end=pair['start'],pair['end'];players=[]
        for arm,label,key in [('old','Old wording, newly spoken at 1.0×','old_text'),('adapted','Meaning-first wording, newly spoken at 1.0×','adapted_text')]:
            row=records.get(pair['id']+'_'+arm)
            status='<p>Audio not generated yet.</p>'
            if row:
                duration=row['native_frames']/44100
                status=f'<audio controls preload="none" src="{url(report_path.parent/row["file"])}"></audio><p>Complete take: {duration:.3f}s; difference from caption span: {duration-(end-start):+.3f}s.</p>'
            players.append(f'<article><h3>{label}</h3><p lang="de">{esc(pair[key])}</p>{status}</article>')
        if continuous:
            sections.append(f'<section><h2>Opening passage · {start:.2f}–{end:.2f}s</h2>'
                '<p>The passage ends at the sentence boundary after 15 seconds. Play each version from the beginning to hear the flow.</p>'
                '<div class="grid"><article><h3>Before — existing German dub</h3>'
                f'<video controls preload="none" src="{url(root/pair["older_dub"])}#t={start},{end}" data-start="{start}" data-end="{end}"></video></article>'
                +players[1]+'</div><details><summary>Original English, wording and experimental control</summary>'
                f'<video controls preload="none" src="{url(root/pair["original_video"])}#t={start},{end}" data-start="{start}" data-end="{end}"></video>'
                f'<p>{esc(pair["source_text"])}</p>'+players[0]+
                f'<p>{esc(pair["adaptation_note"])}</p></details></section>')
            continue
        sections.append(f'<section><h2>{esc(pair["id"].replace("_"," "))}</h2><p>{start:.3f}–{end:.3f}s · source caption span {end-start:.3f}s · preset {esc(pair["voice"])}</p>'
            f'<p><strong>English source:</strong> {esc(pair["source_text"])}</p>'
            f'<details><summary>Play original passage</summary><video controls preload="none" src="{url(root/pair["original_video"])}#t={start},{end}" data-start="{start}" data-end="{end}"></video></details>'
            '<div class="grid">'+''.join(players)+f'</div><p><strong>What changed:</strong> {esc(pair["adaptation_note"])}</p></section>')
    status='Native audio available for listening; quality and source speech/pause matching remain unverified.' if records else 'Prepared dialogue only. Synthesis has not been released or run.'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meaning-first dubbing comparison</title><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4f5f7;color:#19242e}
section{background:white;padding:22px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
video,audio{width:100%}video{max-height:400px}article{border-top:2px solid #b7c7da}h1{font-size:1.8rem}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>Meaning-first dubbing comparison</h1><p><strong>'''+esc(status)+'''</strong></p>
<p>Source-linked dialogue. The newly generated old and revised German use the same model, preset voice and seed within each pair, at synthesis/playback 1.0×.
Full takes are retained without time stretching, trimming, or padding. Compare meaning, articulation, rhythm and sentence endings.</p>
<p>Caption spans provide reference time budgets, not acoustic speech measurements. Raw duration includes native silence.
This experiment tests dialogue wording; it does not transfer the original performance, clone the speakers, match every pause, or produce a synchronized film.</p>'''+''.join(sections)+'''
<script>const players=[...document.querySelectorAll('video,audio')];for(const p of players){p.playbackRate=1;
p.addEventListener('ratechange',()=>{if(p.playbackRate!==1)p.playbackRate=1});
p.addEventListener('play',()=>{for(const q of players)if(q!==p)q.pause();if(p.dataset.end&&(p.currentTime<+p.dataset.start||p.currentTime>=+p.dataset.end))p.currentTime=+p.dataset.start});
p.addEventListener('timeupdate',()=>{if(p.dataset.end&&p.currentTime>=+p.dataset.end)p.pause()});}</script></html>'''
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:f.write(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();write_review(args.input,args.output,root=args.root,report_file=args.report)
