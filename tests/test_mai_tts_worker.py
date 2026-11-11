"""MAI-Voice worker: voice naming, tail cleanup and native-pace retries. No API calls."""
import json

import numpy as np
import pytest

import mai_tts_worker as mai

RATE = 24000


def tone(seconds, db=-14.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * 32767 * 10 ** (db / 20)).astype(np.int16)


def silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.int16)


def test_voice_ids_and_languages():
    assert mai.voice_id("", "de") == "de-DE-Mia:MAI-Voice-2.1-Flash"
    assert mai.voice_id("Harper", "German") == "de-DE-Harper:MAI-Voice-2.1-Flash"
    assert mai.voice_id("Klaus", "de", "microsoft/mai-voice-2-flash") == "de-DE-Klaus:MAI-Voice-2"
    assert mai.voice_id("fr-FR-Marc:MAI-Voice-2.1", "fr") == "fr-FR-Marc:MAI-Voice-2.1"
    assert mai.voice_id(None, "fr") == "fr-FR-Soleil:MAI-Voice-2.1-Flash"
    with pytest.raises(ValueError):
        mai.language_code("ja")


def test_isolated_blips_after_long_silence_are_silenced():
    # The measured MAI pattern: speech, ~450 ms silence, a 200 ms and a 60 ms burst.
    audio = np.concatenate([tone(2.0), silence(.45), tone(.2, -30), silence(.12), tone(.06, -20), silence(.1)])
    cleaned, evidence = mai.clean_tail(audio, RATE)
    assert evidence and abs(evidence['silenced_from_seconds'] - 2.05) < .03
    assert len(evidence['bursts']) == 2
    assert abs(mai.speech_end(cleaned, RATE) - 2.0) < .03
    assert len(cleaned) == len(audio)


@pytest.mark.parametrize('tail', [
    [silence(.15), tone(.2)],            # short pause then a short word: part of speech
    [silence(.45), tone(.4)],            # long trailing run: may be a real word
    [silence(.45)] + [tone(.2), silence(.1)] * 4,   # too much trailing sound in total
])
def test_real_speech_is_never_removed(tail):
    audio = np.concatenate([tone(2.0)] + tail)
    cleaned, evidence = mai.clean_tail(audio, RATE)
    assert evidence is None and np.array_equal(cleaned, audio)


def test_next_speed_targets_aim_and_respects_bounds():
    assert mai.next_speed(1.0, 30.3, 28.94) == pytest.approx(30.3 / (.96 * 28.94))
    assert mai.next_speed(1.0, 60, 28.94) == mai.MAX_SPEED
    assert mai.next_speed(1.0, 10, 28.94) == mai.MIN_SPEED


def fake_api(monkeypatch, durations):
    """Each call returns speech lasting durations[speed] seconds."""
    calls = []
    def synthesize_once(session, key, model, voice, text, speed, tmp):
        calls.append(round(speed, 4))
        seconds = durations(speed)
        return np.concatenate([tone(seconds), silence(.3)]), RATE, dict(speed=round(speed, 4), api_seconds=.1, generation_id='g')
    monkeypatch.setattr(mai, 'synthesize_once', synthesize_once)
    return calls


def test_long_take_is_requested_faster_and_evidence_written(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch, lambda speed: 30.3 / (speed ** .8))   # MAI responds less than 1:1
    out = tmp_path / 'seg_0001.wav'
    evidence = mai.synthesise(None, 'k', mai.DEFAULT_MODEL, dict(text='Hallo Welt', language='de', output=str(out),
                                                                 voice='', target_seconds=28.94))
    assert calls[0] == 1.0 and calls[1] > 1.0
    chosen = evidence['attempts'][evidence['chosen_attempt']]
    assert mai.LOW * 28.94 <= chosen['speech_end_seconds'] <= 28.94 + mai.END_SLACK
    saved = json.loads(out.with_suffix('.mai.json').read_text())
    assert saved['voice'] == 'de-DE-Mia:MAI-Voice-2.1-Flash' and saved['target_seconds'] == 28.94
    rate, samples = mai.read_wav(out)[1], mai.read_wav(out)[0]
    assert rate == RATE and len(samples) > 0


def test_fitting_take_is_used_once_and_no_target_means_native(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch, lambda speed: 27.5)
    mai.synthesise(None, 'k', mai.DEFAULT_MODEL, dict(text='x', language='de', output=str(tmp_path / 'a.wav'), target_seconds=28.94))
    mai.synthesise(None, 'k', mai.DEFAULT_MODEL, dict(text='x', language='de', output=str(tmp_path / 'b.wav')))
    assert calls == [1.0, 1.0]


def test_unreachable_target_stops_at_bound_and_keeps_closest(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch, lambda speed: 45.0 / speed)
    evidence = mai.synthesise(None, 'k', mai.DEFAULT_MODEL, dict(text='x', language='de', output=str(tmp_path / 'a.wav'), target_seconds=28.94))
    assert calls == [1.0, mai.MAX_SPEED]          # second request already at the bound
    assert evidence['attempts'][evidence['chosen_attempt']]['speed'] == mai.MAX_SPEED


def test_runner_accepts_mai_engine():
    from run_pipeline import build_parser
    args = build_parser().parse_args(['--target-lang', 'de', '--tts-engine', 'mai', '--mai-voice', 'Harper'])
    assert (args.tts_engine, args.mai_voice, args.mai_model) == ('mai', 'Harper', 'microsoft/mai-voice-2.1-flash')


def test_take_ending_just_past_target_is_accepted_like_the_finisher(monkeypatch, tmp_path):
    # The replayed real MAI take: speech ends 29.16 s for a 28.94 s target (finisher end 29.04 s + tolerance).
    calls = fake_api(monkeypatch, lambda speed: 29.02)
    mai.synthesise(None, 'k', mai.DEFAULT_MODEL, dict(text='x', language='de', output=str(tmp_path / 'a.wav'), target_seconds=28.94))
    assert calls == [1.0]


def test_explicit_max_end_mirrors_finisher_deadline(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch, lambda speed: 29.16)    # replayed real take end
    evidence = mai.synthesise(None, 'k', mai.DEFAULT_MODEL, dict(text='x', language='de', output=str(tmp_path / 'a.wav'),
                                                                 target_seconds=28.94, max_end_seconds=29.248))
    assert calls == [1.0] and evidence['max_end_seconds'] == 29.248
