"""Measure F0 on longer diarized turns; no speech or synthesis models."""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modal_control.speaker_voices import assign_voices


def measure(index):
    work = ROOT / 'modal_control/batch_inputs'
    source = work / f'video_{index:02d}.wav'
    result = json.loads((work / f'video_{index:02d}.gpu_full.json').read_text())
    output = work / f'video_{index:02d}.pitch_review_v2.json'
    if output.exists():
        raise FileExistsError(output)
    pitches, evidence = {}, {}
    with sf.SoundFile(source) as wav:
        for speaker in result['voices']:
            candidates = sorted((t for t in result['turns'] if t['speaker'] == speaker and t['end'] - t['start'] >= 2.0),
                                key=lambda t: -(t['end'] - t['start']))[:20]
            estimates, spans = [], []
            for turn in candidates:
                # Sample from the center, away from speaker boundaries.
                start = (turn['start'] + turn['end']) / 2 - 1
                wav.seek(int(start * wav.samplerate))
                data = wav.read(int(2 * wav.samplerate), dtype='float32')
                f0, voiced, probability = librosa.pyin(data, fmin=60, fmax=350, sr=wav.samplerate)
                samples = f0[voiced & (probability >= 0.35) & np.isfinite(f0)]
                estimates.extend(samples.tolist())
                spans.append({'start': round(start, 3), 'seconds': 2, 'voiced_samples': len(samples)})
            pitches[speaker] = estimates
            evidence[speaker] = spans
    report = {'status': 'complete', 'voices': assign_voices(pitches), 'sample_spans': evidence,
              'method': 'librosa pyin F0 on centered two-second spans of long diarized turns',
              'caution': 'Pitch is an imperfect voice presentation heuristic, not a gender determination.'}
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(measure(int(sys.argv[1])), indent=2))
