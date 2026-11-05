"""Prepare a listening regression set from existing evidence; no model or media processing.

Caption density and gaps select cases only. They are never acoustic ground truth.
Historical dub tempo is processing evidence, not the original speaker's pace.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def choose_cases(review, crispr):
    videos = {v['id']: v for v in review['videos']}
    bill, mind = videos['bill_ackman'], videos['mind_control']
    cases = []

    def add(ident, title, video, start, end, reason, listen_for):
        start, end = max(0, start), min(video['review_end'], end)
        if not 0 < end - start <= 45:
            raise ValueError('Each listening excerpt must be at most 45 seconds')
        cases.append(dict(id=ident, title=title, video_id=video['id'],
            original_path=video['original_path'], old_dub_path=video['dub_path'],
            start=round(start, 3), end=round(end, 3), selection_reason=reason,
            listen_for=listen_for, corrected_take_status='not_tested',
            acoustic_timing_status='not_measured',
            original_pace_classification='pending_listening',
            english_context=[dict(start=c['start'], end=c['end'], text=c['text'])
                for c in video['english_cues'] if c['end'] > start and c['start'] < end],
            historical_tempo=[dict(index=s['index'], start=s['start'], rate=s['rate'])
                for s in video.get('merged_segments', []) if s['end'] > start and s['start'] < end]))

    add('calm_control', 'Steady delivery: CRISPR control', crispr, 4.32, 22.32,
        'The user reports mostly steady delivery; retain a control alongside difficult cases.',
        'Does the corrected take keep the easy passage natural without extra pauses?')
    short = max((s for s in bill['merged_segments'] if len(s['text'].split()) <= 6), key=lambda s:s['rate'])
    add('short_reply', 'Short reply under timing pressure', bill, short['start']-7, short['end']+7,
        f"Historical segment {short['index']} was fitted at {short['rate']:.3f}x. This does not establish fast original speech.",
        'Is the reply complete, clearly articulated, and placed at the correct conversational moment?')
    left, right = max(zip(mind['merged_segments'], mind['merged_segments'][1:]),
                     key=lambda pair:abs(pair[1]['rate']-pair[0]['rate']))
    add('tempo_transition', 'Sudden change in the old dub', mind, left['start']-4, right['end']+4,
        f"Adjacent historical fit factors change from {left['rate']:.3f}x to {right['rate']:.3f}x.",
        'Does the new voice follow the original rhythm, or still suddenly sprint and then drag?')
    cues = bill['english_cues']
    a,b = max(zip(cues,cues[1:]), key=lambda pair:pair[1]['start']-pair[0]['end'])
    add('pause', 'Long gap and restart', bill, a['start']-4, b['end']+4,
        f"Caption gap of {b['start']-a['end']:.3f}s; listen to determine whether this is silence, music, or an edit.",
        'Are the source pause and restart preserved, with no slow drawl or speech pushed into the gap?')
    a,b = next((a,b) for a,b in zip(cues,cues[1:]) if a.get('speaker') and b.get('speaker')
               and a['speaker'] != b['speaker'] and 0 <= b['start']-a['end'] <= .8)
    add('speaker_change', 'Quick speaker handover', bill, a['start']-5, b['end']+7,
        'Two different recorded speaker labels with a caption gap no longer than 0.8s.',
        'Do the voices remain distinct and the reply start correctly without cutting the previous speaker?')
    # Rank complete caption groups, not isolated tiny cue windows or fitted dub rates.
    windows = []
    for i,c in enumerate(mind['english_cues']):
        group = [r for r in mind['english_cues'][i:] if r['end'] <= c['start']+20]
        if group and group[-1]['end']-c['start'] >= 15:
            duration = group[-1]['end']-c['start']
            windows.append((sum(len(r['text'].split()) for r in group)/duration,c['start'],group[-1]['end']))
    density,start,end = max(windows)
    add('dense_source', 'Dense source passage: verify by listening', mind, start, end,
        f"Highest 15–20s caption-density window: {density:.2f} English words/s. Selection proxy only; not verified articulation speed.",
        'If the original genuinely speeds up, does the dub retain that energy while staying intelligible?')
    return cases


def make_page(manifest, directory, root):
    esc = html.escape
    cards = []
    for case in manifest['cases']:
        players = []
        for key,label in [('original_path','Original English'),('old_dub_path','Historical German dub')]:
            url = quote(os.path.relpath(root / case[key], directory), safe='/')
            players.append(f'<div><h3>{label}</h3><video controls preload="none" '
                f'src="{esc(url)}#t={case["start"]},{case["end"]}" data-start="{case["start"]}" '
                f'data-end="{case["end"]}"></video></div>')
        options = '<option value="pending">Not evaluated</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="uncertain">Uncertain</option>'
        checks = ''.join(f'<label>{label} <select data-check="{key}">{options}</select></label>' for key,label in [
            ('meaning','Meaning and complete words'),('pace','Natural pace relative to original'),
            ('pauses','Speech and pause placement'),('performance','Emphasis and delivery'),('voice','Voice consistency')])
        context = ' '.join(c['text'] for c in case['english_context'])
        cards.append(f'<section data-case="{case["id"]}"><h2>{esc(case["title"])}</h2>'
            f'<p>{case["start"]:.2f}–{case["end"]:.2f}s · {esc(case["video_id"])}</p>'
            f'<p>{esc(case["selection_reason"])}</p><p><strong>Listen for:</strong> {esc(case["listen_for"])}</p>'
            '<div class="players">'+''.join(players)+'</div>'
            '<h3 class="candidate-title">Corrected take: not supplied</h3><p>Select an excerpt audio file that starts at this excerpt’s beginning. '
            'The file stays on this computer. Attaching it does not verify timing or processing.</p>'
            '<input type="file" accept="audio/*"><audio controls preload="none"></audio>'
            '<fieldset disabled><legend>Assessment of the corrected take</legend>'+checks+
            '<label>Notes <textarea placeholder="Describe the exact problem and timestamp"></textarea></label></fieldset>'
            f'<details><summary>Existing English transcript — not independently verified</summary><p>{esc(context)}</p></details></section>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dubbing stress review</title><style>
body{max-width:1100px;margin:auto;padding:24px;font:17px/1.5 system-ui;background:#f3f4f6;color:#17212b}
section{background:white;padding:24px;margin:24px 0;border-radius:12px}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}
video,audio{width:100%}label{display:block;margin:10px 0}textarea{display:block;width:95%}button,select,input{font:inherit}
@media(max-width:700px){.players{grid-template-columns:1fr}}button{padding:12px}h1{font-size:1.8rem}
</style><h1>Is the dubbing actually fixed?</h1>
<p>Six contrasting excerpts at 1.0× playback. Preserve genuine changes in the original speaker’s pace;
listen for unnatural changes introduced by the dub. The historical German files are baselines, not new results.</p>
<p><strong>Current verdict: not established.</strong> No corrected takes or acoustic speech/pause measurements
have been added to this set. Caption density and gaps only help select passages; they do not prove how the speaker sounds.</p>
<p>Start either reference to replay its selected range. Only one player runs at a time. Notes are kept in this page
until it closes; use Export to save them. Timing, meaning and listening quality must be reviewed separately.</p>
''' + ''.join(cards) + '''<button id="export">Export listening notes</button>
<script>
const players=[...document.querySelectorAll('audio,video')];
for(const p of players){p.playbackRate=1;
p.addEventListener('ratechange',()=>{if(p.playbackRate!==1)p.playbackRate=1});
p.addEventListener('play',()=>{for(const q of players)if(q!==p)q.pause();
if(p.dataset.end&&(p.currentTime<+p.dataset.start||p.currentTime>=+p.dataset.end))p.currentTime=+p.dataset.start;});
p.addEventListener('timeupdate',()=>{if(p.dataset.end&&p.currentTime>=+p.dataset.end)p.pause()});}
for(const s of document.querySelectorAll('section')){const input=s.querySelector('input'),audio=s.querySelector('audio');
input.addEventListener('change',()=>{audio.pause();if(s.dataset.url)URL.revokeObjectURL(s.dataset.url);
const file=input.files[0];for(const select of s.querySelectorAll('select'))select.value='pending';s.querySelector('textarea').value='';
s.querySelector('fieldset').disabled=!file;
if(!file){audio.removeAttribute('src');audio.load();s.querySelector('.candidate-title').textContent='Corrected take: not supplied';return;}
s.dataset.url=URL.createObjectURL(file);audio.src=s.dataset.url;
s.querySelector('.candidate-title').textContent='Corrected take: '+file.name;});}
document.querySelector('#export').onclick=()=>{
const cases=[...document.querySelectorAll('section')].map(s=>({id:s.dataset.case,
candidate_file:s.querySelector('input').files[0]?.name||null,
candidate_bytes:s.querySelector('input').files[0]?.size||null,
checks:Object.fromEntries([...s.querySelectorAll('select')].map(x=>[x.dataset.check,x.value])),
notes:s.querySelector('textarea').value}));
const result={schema:1,created_at:new Date().toISOString(),type:'subjective_listening_only',
acoustic_timing_verified:false,processing_verified:false,cases};
const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
const a=document.createElement('a');a.href=url;a.download='dubbing-listening-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></html>'''


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    review_path = root/'dubbing_naturalness_2026-10-28/review_data.json'
    crispr_path = root/'modal_control/batch_inputs/video_01.gpu_full.json'
    source_path = root/'dubbing_naturalness_2026-10-28/crispr_native/inputs.json'
    review = json.loads(review_path.read_text())
    source = json.loads(source_path.read_text())['source_metadata']
    data = json.loads(crispr_path.read_text())
    crispr = dict(id='crispr',review_end=data['duration_seconds'],english_cues=data['english'],
                  original_path=source['original_video'],dub_path=source['older_dub'])
    cases = choose_cases(review, crispr)
    for case in cases:
        for key in ('original_path','old_dub_path'):
            path = (root/case[key]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Missing or external reference: {path}')
    manifest = dict(schema=1,status='prepared_no_new_synthesis',cases=cases,
        total_excerpt_seconds=round(sum(c['end']-c['start'] for c in cases),3),
        input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (review_path,crispr_path,source_path)},
        limitations=['Existing media paths are verified, not newly content-hashed.',
            'Historical segment boundaries were reconstructed; they are not fresh acoustic alignment.',
            'Fast original delivery remains unverified until listening; dub tempo cannot prove it.',
            'Excerpt edges provide listening context, not approved synthesis boundaries.',
            'No corrected takes, measured acoustic timelines, or naturalness verdict exist for this set.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'review.html').write_text(make_page(manifest,output,root))
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build(args.root,args.output)
    print(json.dumps(dict(cases=len(result['cases']),seconds=result['total_excerpt_seconds'],status=result['status'])))
