"""Export existing benchmark evidence across ALL hardware. Never submits work."""
import csv,html,json
from pathlib import Path
from modal_control.core import Store,utc
ROOT=Path(__file__).resolve().parents[1]/'speech_model_benchmark_2026-10-27'
OUT=ROOT/'05_evaluation';STORE=Store(ROOT.parent/'modal_control/.state')
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def link(p):return '/benchmark/'+str(p.relative_to(ROOT))
models=read(ROOT/'01_protocol/models.json');records=[]
for category in ['runs','attempts','smoke_runs']:
 bases=set()
 for pattern in ['**/model.json','**/hardware.json','**/failure.json','**/metrics.json','**/segments.json']:
  for p in (ROOT/category).glob(pattern):
   bases.add(p.parent.parent if p.parent.name in ['asr','translation','tts','speech_translation'] else p.parent)
 for base in sorted(bases):
  spec=read(base/'model.json');key=spec.get('key') or base.relative_to(ROOT/category).parts[0]
  model=next((m for m in models if m['key']==key),None)
  if not model:continue
  hardware=read(base/'hardware.json');gpu=hardware.get('gpu');hw=gpu.splitlines()[1].split(',')[0] if gpu and len(gpu.splitlines())>1 else ('CPU' if model['hardware']=='cpu' else 'not observed')
  for field in model['fields']:
   folder=base/field;metrics=read(folder/'metrics.json');segments=read(folder/'segments.json') or [];failure=read(folder/'failure.json') or read(base/'failure.json')
   if not folder.exists() and not failure:continue
   scope='smoke' if spec.get('smoke') or category=='smoke_runs' else 'full'
   if metrics and metrics.get('input_audio_seconds',metrics.get('source_excerpt_seconds',600))<600:scope='smoke'
   status=metrics.get('status') or ('failed' if failure else 'partial')
   validation=[]
   if field in ['asr','translation'] and segments and any(not x.get('text','').strip() for x in segments):validation.append('One or more empty text segments; not a valid complete output')
   if scope=='full' and metrics and len(segments)<24:validation.append('Fewer than 24 source windows present')
   if validation and status=='complete':status='invalid_or_incomplete_output'
   quality=read(folder/'quality_proxy.json')
   records.append({'model':key,'field':field,'hardware':hw,'scope':scope,'status':status,'seconds':metrics.get('inference_seconds'),'speed_x':metrics.get('realtime_multiple'),'source_seconds':metrics.get('input_audio_seconds',metrics.get('source_excerpt_seconds')),'generated_audio_seconds':metrics.get('generated_audio_seconds'),'segments_recorded':len(segments),'tts_wer_proxy_percent':100*quality['wer'] if 'wer' in quality else None,'artifact_path':str(base.relative_to(ROOT)),'model_spec':spec,'hardware_raw':hardware,'metrics_raw':metrics,'load_raw':read(base/'load.json'),'download_raw':read(base/'download.json'),'client_timing_raw':read(base/'client_timing.json'),'quality_raw':quality,'failure_raw':failure,'validation_notes':validation,'files':[{'name':p.name,'url':link(p)} for p in sorted(folder.iterdir()) if p.is_file()] if folder.exists() else []})
write('ALL_RECORDED_RESULTS.json',{'generated':utc(),'policy':'All hardware and attempts included. No new tests. Smoke, partial and invalid results are labeled. Repeated attempts are not independent models.','records':records})
columns=['model','field','hardware','scope','status','seconds','speed_x','source_seconds','generated_audio_seconds','segments_recorded','tts_wer_proxy_percent','artifact_path']
with (OUT/'ALL_RECORDED_RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r[k] for k in columns} for r in records)
with STORE.db() as c:
 runs=[dict(x) for x in c.execute('SELECT * FROM runs ORDER BY created')];alerts=[dict(x) for x in c.execute('SELECT * FROM alerts ORDER BY id')]
 with (OUT/'ALL_CONTROLLER_EVENTS.jsonl').open('w') as f:
  count=0
  for row in c.execute('SELECT * FROM events ORDER BY seq'):
   d=dict(row);d['payload']=json.loads(d['payload']);f.write(json.dumps(d,ensure_ascii=False)+'\n');count+=1
for r in runs:r['manifest']=json.loads(r['manifest'])
write('ALL_CONTROLLER_RUNS.json',runs);write('ALL_ALERTS.json',alerts)
snapshots=[{'file':p.name,'data':read(p)} for p in sorted((STORE.root/'snapshots').glob('*.json'))];write('ALL_CLOUD_SNAPSHOTS.json',snapshots)
inventory=[]
for top in ['00_inputs','01_protocol','02_asr','03_translation','04_dubbing','05_evaluation','06_logs','code','runs','attempts','smoke_runs']:
 for p in sorted((ROOT/top).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('ALL_'):inventory.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'url':link(p)})
for p in sorted((STORE.root/'runs').rglob('*')):
 if p.is_file():inventory.append({'path':'controller/'+str(p.relative_to(STORE.root/'runs')),'bytes':p.stat().st_size,'url':'/files/'+str(p.relative_to(STORE.root/'runs'))})
write('ALL_ARTIFACTS.json',inventory)
page=['<!doctype html><meta charset="utf-8"><title>All recorded benchmark evidence</title><style>body{font:14px system-ui;background:#101827;color:#e8eef8;margin:28px}a{color:#9bceff}table{border-collapse:collapse;width:100%;margin-bottom:25px}td,th{padding:9px;border:1px solid #35445c;text-align:left}summary{cursor:pointer;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}input{padding:8px;width:400px}h1{font-size:26px}</style><h1>All recorded results — every hardware type and attempt</h1><p>No new testing was performed. T4, L4, A10G and CPU data are all included. Smoke tests and invalid/partial output are not comparable to completed ten-minute tests. Missing accuracy remains unmeasured. Model loading, warm-up, transfer and image builds are separate from inference timings. Current Canary includes alignment; Flash runs may omit it.</p>']
page.append('<p>'+ ' · '.join(f'<a href="{name}">{label}</a>' for name,label in [('ALL_RECORDED_RESULTS.csv','All measurements CSV'),('ALL_RECORDED_RESULTS.json','Complete raw result data'),('ALL_CONTROLLER_RUNS.json','Every controller attempt'),('ALL_CONTROLLER_EVENTS.jsonl','All saved events and printed output'),('ALL_ALERTS.json','All alerts'),('ALL_CLOUD_SNAPSHOTS.json','All cloud snapshots'),('ALL_ARTIFACTS.json','Complete file inventory')])+'</p>')
page.append(f'<p>{len(models)} model configurations · {len(records)} result/partial records · {len(runs)} controller/history records · {count} events · {len(inventory)} indexed files. Historical imports and repeated attempts are not additional models.</p>')
for scope in ['full','smoke']:
 page.append('<h2>'+scope.title()+' attempts</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in ['Model','Field','Hardware','Status','Seconds','Speed ×','Audio seconds','TTS WER proxy %','Evidence'])+'</tr>')
 for r in records:
  if r['scope']!=scope:continue
  vals=[r['model'],r['field'],r['hardware'],r['status'],r['seconds'],r['speed_x'],r['generated_audio_seconds'],r['tts_wer_proxy_percent']]
  fmt=lambda v:'—' if v is None else str(round(v,2)) if isinstance(v,float) else str(v)
  links=' · '.join(f'<a href="{html.escape(f["url"])}">{html.escape(f["name"])}</a>' for f in r['files'] if f['name'] in ['metrics.json','failure.json','segments.json','transcript.txt','quality_proxy.json'])
  page.append('<tr>'+''.join('<td>'+html.escape(fmt(v))+'</td>' for v in vals)+'<td>'+links+'<br><small>'+html.escape(r['artifact_path'])+'</small></td></tr>')
 page.append('</table>')
page.append('<h2>Every model, including those without completed output</h2><table><tr><th>Model</th><th>Recorded result entries</th><th>Latest controller state</th></tr>')
for m in models:
 latest=next((r for r in reversed(runs) if r['job']==m['key']),None)
 page.append('<tr><td>'+m['key']+'</td><td>'+str(sum(r['model']==m['key'] for r in records))+'</td><td>'+html.escape(latest['status'] if latest else 'Historical data only / no supervised run')+'</td></tr>')
page.append('</table><details><summary>All files: transcripts, audio, videos, logs, model cards, environment records, source snapshots and hashes</summary><ul>')
for f in inventory:page.append(f'<li><a href="{html.escape(f["url"])}">{html.escape(f["path"])}</a> ({f["bytes"]} bytes)</li>')
page.append('</ul></details>');(OUT/'ALL_RESULTS.html').write_text(''.join(page))
print(json.dumps({'result_records':len(records),'controller_records':len(runs),'events':count,'files':len(inventory),'full_complete':sum(r['scope']=='full' and r['status']=='complete' for r in records)}))
