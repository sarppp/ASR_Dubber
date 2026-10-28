"""Choose a documented preset for usable but borderline F0 measurements."""
import math


def resolve_borderline_voices(voices):
    """Return copied assignments and mark borderline guesses explicitly.

    Thresholding perceived pitch is a weak voice-matching heuristic and does
    not determine gender. Refuse measurements without enough voiced frames.
    """
    result = {speaker: dict(value) for speaker, value in voices.items()}
    used = {value['voice'] for value in result.values() if value.get('voice')}
    for speaker, value in result.items():
        if value.get('voice'):
            continue
        pitch, count = value.get('median_f0_hz'), value.get('voiced_samples', 0)
        if pitch is None or not math.isfinite(float(pitch)) or count < 10:
            raise ValueError(f'Insufficient voiced pitch evidence for {speaker}')
        prefix = 'M' if float(pitch) < 165 else 'F'
        voice = next((f'{prefix}{i}' for i in range(1, 6) if f'{prefix}{i}' not in used), f'{prefix}1')
        used.add(voice)
        value.update(voice=voice, status='borderline_pitch_estimate')
    return result
