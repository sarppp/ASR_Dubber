"""Group consecutive same-speaker captions into speech-friendly spans."""


def merge_speech_cues(cues, max_seconds=20.0, max_characters=280, max_gap=0.4):
    """Preserve text and speaker order; merge only adjacent, non-overlapping cues."""
    merged = []
    for cue in cues:
        start, end = float(cue['start']), float(cue['end'])
        text, speaker = cue['text'].strip(), cue.get('speaker')
        if start < 0 or end <= start or not text or speaker is None:
            raise ValueError('Invalid or unattributed speech cue')
        if merged and start < merged[-1]['end'] - 0.001:
            raise ValueError('Overlapping speech cues need review')
        if merged:
            prev = merged[-1]
            gap = start - prev['end']
            if (speaker == prev['speaker'] and gap <= max_gap and end - prev['start'] <= max_seconds
                    and len(prev['text']) + 1 + len(text) <= max_characters):
                prev['end'] = end
                prev['text'] += ' ' + text
                continue
        merged.append({'start': start, 'end': end, 'text': text, 'speaker': speaker})
    return merged
