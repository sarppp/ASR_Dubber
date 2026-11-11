"""Explicitly user-authorized local GPU execution of the continuous German trial."""
import hashlib
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT/'new_guys/crispr/deepseek_t4_10cues_2026-11-05'


def main():
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    from batch_dubbing.continuous_german import MODEL, REVISION, passage, synthesize, save_result
    destination = TRIAL/'continuous_local'
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing local take')
    scope = json.loads((TRIAL/'scope.json').read_text())
    for name in ('input.de.srt','voice_reference.wav','source_10cues.mp4'):
        if hashlib.sha256((TRIAL/name).read_bytes()).hexdigest() != scope['input_sha256'][name]:
            raise ValueError('Input identity changed: '+name)
    import torch
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from faster_qwen3_tts import FasterQwen3TTS
    if not torch.cuda.is_available():
        raise RuntimeError('Local CUDA GPU required; no CPU or cloud fallback')
    free, total = torch.cuda.mem_get_info()
    gpu = torch.cuda.get_device_name()
    print(json.dumps(dict(stage='preflight',gpu=gpu,free_bytes=free,total_bytes=total)),flush=True)
    if free < 4.5*1024**3:
        raise RuntimeError('Less than 4.5 GiB available; refusing to crowd another GPU workload')
    model_path = snapshot_download(MODEL,revision=REVISION,local_files_only=True)
    torch.manual_seed(20261105)
    torch.set_num_threads(4)
    began = time.monotonic()
    print('Loading cached model in float16; max_seq_len=1024',flush=True)
    model = FasterQwen3TTS.from_pretrained(model_path,device='cuda:0',dtype=torch.float16,max_seq_len=1024)
    print('Generating one continuous German take',flush=True)
    wavs, sr = synthesize(model,TRIAL/'input.de.srt',TRIAL/'voice_reference.wav')
    if len(wavs) != 1:
        raise ValueError('Expected one take')
    audio = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else np.asarray(wavs[0])
    if audio.ndim != 1 or sr != 24000 or not np.isfinite(audio).all() or not 10 < audio.size/sr < 49:
        raise ValueError('Invalid or possibly token-capped take; no automatic retry')
    out = io.BytesIO()
    sf.write(out,audio,sr,format='WAV',subtype='PCM_16')
    data = out.getvalue()
    report = dict(status='complete',synthesis_calls=1,execution='local CUDA',
        user_authorization='run it in our gpu',gpu=gpu,precision='float16',
        max_seq_len=1024,model=MODEL,revision=REVISION,text=passage(TRIAL/'input.de.srt'),
        audio_seconds=audio.size/sr,audio_sha256=hashlib.sha256(data).hexdigest(),
        worker_seconds=time.monotonic()-began,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        note='One continuous request, no time stretching or per-cue synthesis. Listening review required.')
    save_result(dict(audio=data,report=report),destination)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
