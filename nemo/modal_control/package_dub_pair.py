"""Package existing generated waveforms into the two requested videos. No model calls."""
from pathlib import Path
import json,subprocess,tempfile,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]/'speech_model_benchmark_2026-10-27'
PAIR=ROOT/'07_german_dub_comparison';Q=PAIR/'qwen_17b_t4'
def ff(args):subprocess.run(['ffmpeg','-nostdin','-v','error','-y',*map(str,args)],check=True)
def stamp(t):
 m=round(t*1000);return f'{m//3600000:02}:{m//60000%60:02}:{m//1000%60:02},{m%1000:03}'
def main():
 metrics=json.loads((Q/'metrics.json').read_text());rows=json.loads((Q/'segments.json').read_text())
 if metrics.get('status')!='complete' or len(rows)!=24:raise ValueError('Full 24-segment generation is required')
 with tempfile.TemporaryDirectory(prefix='qwen17-dub-') as tmp:
  tmp=Path(tmp);raw=[];fitted=[]
  for row in rows:
   i=row['index'];src=Q/row['path'];r=tmp/f'raw{i:03}.wav';f=tmp/f'fit{i:03}.wav';ff(['-i',src,'-ar','24000','-ac','1',r]);raw.append(r)
   window=row['source_end']-row['source_start'];ratio=max(1,row['audio_seconds']/window);filters=[]
   while ratio>2:filters.append('atempo=2');ratio/=2
   if ratio>1.001:filters.append(f'atempo={ratio:.8f}')
   filters+=['apad',f'atrim=duration={window}'];ff(['-i',r,'-af',','.join(filters),f]);fitted.append(f)
  for name,files in [('raw_narration.wav',raw),('timeline_600s.wav',fitted)]:
   listing=tmp/(name+'.txt');listing.write_text(''.join("file '"+str(p)+"'\n" for p in files));ff(['-f','concat','-safe','0','-i',listing,'-c','copy',Q/name])
  ff(['-i',ROOT/'00_inputs/source_10min.mp4','-i',Q/'timeline_600s.wav','-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-metadata:s:a:0','language=deu','-t','600','-movflags','+faststart',Q/'dubbed_10min.mp4'])
 srt='\n\n'.join(f"{i+1}\n{stamp(r['source_start'])} --> {stamp(r['source_end'])}\n{r['text']}" for i,r in enumerate(rows))+'\n'
 for name in ['qwen_17b_t4','supertonic_cpu']:(PAIR/name/'german_chunk_timing.srt').write_text(srt)
 for idx in [0,8,16]:shutil.copy2(Q/f'{idx:03}.wav',Q/f'review_{idx*25:03}s.wav')
 checks={}
 for name in ['qwen_17b_t4','supertonic_cpu']:
  video=PAIR/name/'dubbed_10min.mp4';info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(video)],text=True))
  if not any(s['codec_type']=='video' for s in info['streams']) or not any(s['codec_type']=='audio' for s in info['streams']):raise ValueError('Missing video/audio stream')
  if abs(float(info['format']['duration'])-600)>1:raise ValueError('Wrong video duration')
  checks[name]={'file':str(video),'bytes':video.stat().st_size,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'media':info}
 (PAIR/'delivery_manifest.json').write_text(json.dumps(checks,indent=2))
 page=(PAIR/'index.html').read_text().replace('<p id="qwen">Generation in progress. Reload this page after completion.</p>','<video controls preload="metadata" src="qwen_17b_t4/dubbed_10min.mp4"></video><p><a href="qwen_17b_t4/dubbed_10min.mp4" download>Download Qwen 1.7B video</a> · <a href="qwen_17b_t4/raw_narration.wav">Raw narration</a> · <a href="qwen_17b_t4/german_chunk_timing.srt">German subtitles</a></p>')
 (PAIR/'index.html').write_text(page);print('Both ten-minute videos verified:',json.dumps({k:v['bytes'] for k,v in checks.items()}))
if __name__=='__main__':main()
