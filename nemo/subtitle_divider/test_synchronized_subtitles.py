import json
from pathlib import Path
import re
import subprocess

import pytest

from bilingual_subtitles import Cue, EMPTY_SUBTITLE_ROW, parse_srt, write_srt, write_subtitles
from synchronized_subtitles import block_cues, main, sentence_fragments, synchronize


def test_sentence_fragments_keep_abbreviations_words_and_source_provenance():
    source = [Cue(0, 6000, 'Dr. Doudna spoke. Another sentence follows.')]
    fragments, indices = sentence_fragments(source)
    assert [c.text for c in fragments] == ['Dr. Doudna spoke.', 'Another sentence follows.']
    assert indices == [0, 0]
    assert fragments[0].start_ms == 0 and fragments[-1].end_ms == 6000
    assert fragments[0].end_ms == fragments[1].start_ms


def test_staggered_fragments_become_one_complete_shared_sentence():
    tracks = [[Cue(0, 2000, 'We study'), Cue(2000, 4000, 'genes.')],
              [Cue(200, 3800, 'Wir untersuchen Gene.')],
              [Cue(100, 2100, 'Nous étudions'), Cue(2100, 3900, 'les gènes.')]]
    blocks = synchronize(tracks)
    assert len(blocks) == 1
    assert blocks[0]['text'] == ['We study genes.', 'Wir untersuchen Gene.', 'Nous étudions les gènes.']
    assert (blocks[0]['start_ms'], blocks[0]['end_ms']) == (0, 4000)
    assert blocks[0]['source_cue_indices'] == [[0, 1], [0], [0, 1]]
    assert len(block_cues(blocks)[0].text.splitlines()) == 6


def test_real_pauses_keep_all_languages_off_screen_and_repetitions_survive():
    tracks = [[Cue(0, 2000, 'Yes.'), Cue(4000, 6000, 'Yes.')],
              [Cue(100, 1800, 'Ja.'), Cue(4100, 5900, 'Ja.')]]
    blocks = synchronize(tracks)
    assert [(c['start_ms'], c['end_ms']) for c in blocks] == [(0, 2000), (4000, 6000)]
    assert [c['text'] for c in blocks] == [['Yes.', 'Ja.'], ['Yes.', 'Ja.']]


def test_existing_cue_with_two_sentences_can_follow_shared_sentence_end():
    tracks = [[Cue(0, 10000, 'First sentence. Second sentence.')],
              [Cue(0, 5000, 'Erster Satz.'), Cue(5000, 10000, 'Zweiter Satz.')],
              [Cue(0, 5000, 'Première phrase.'), Cue(5000, 10000, 'Deuxième phrase.')]]
    blocks = synchronize(tracks, max_duration_ms=8000)
    assert [b['text'][0] for b in blocks] == ['First sentence.', 'Second sentence.']
    assert [b['source_cue_indices'][0] for b in blocks] == [[0], [0]]


@pytest.mark.parametrize('tracks', [[], [[Cue(0, 100, 'a')]],
    [[Cue(0, 100, '')], [Cue(0, 100, 'a')]],
    [[Cue(0, 200, 'a'), Cue(100, 300, 'b')], [Cue(0, 300, 'c')]]])
def test_invalid_tracks_fail_without_losing_source_text(tracks):
    with pytest.raises(ValueError):
        synchronize(tracks)


def test_no_path_raises_instead_of_truncating_or_loosening_height():
    tracks = [[Cue(0, 2000, 'W' * 500)], [Cue(0, 2000, 'Hallo.')]]
    with pytest.raises(ValueError, match='No complete'):
        synchronize(tracks)
    with pytest.raises(ValueError, match='does not fit'):
        synchronize([[Cue(0, 2000, 'Hello.')], [Cue(0, 2000, 'Hallo.')]], max_height_percent=1)


def test_cli_exports_matching_formats_with_one_style_and_keeps_existing_trial(tmp_path):
    paths = [tmp_path / f'{i}.srt' for i in range(3)]
    for p, text in zip(paths, ['Hello.', 'Hallo.', 'Bonjour.']):
        write_srt(p, [Cue(0, 3000, text)])
    output = tmp_path / 'trial.srt'
    args = ['--tracks', *map(str, paths), '--output', str(output)]
    assert main(args) == 0
    cues = parse_srt(output, preserve_lines=True)
    assert cues[0].text.replace(EMPTY_SUBTITLE_ROW, '').split() == ['Hello.', 'Hallo.', 'Bonjour.']
    ass = output.with_suffix('.ass').read_text(encoding='utf-8-sig')
    assert 'Style: Default,DejaVu Sans,18,' in ass and r'\fs' not in ass
    assert len(re.findall(r'^Dialogue:', ass, re.M)) == len(cues)
    report = json.loads(output.with_suffix('.alignment.json').read_text())
    assert len(report['inputs']) == 3 and 'not a semantic model' in report['alignment_basis']
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


def test_actual_ass_stays_identical_across_old_individual_boundaries(tmp_path):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required for playback regression')
    tracks = [[Cue(0, 1000, 'Hello'), Cue(1000, 3000, 'world.')],
              [Cue(200, 1600, 'Hallo'), Cue(1600, 2800, 'Welt.')],
              [Cue(100, 2900, 'Bonjour le monde.')]]
    blocks = synchronize(tracks)
    assert len(blocks) == 1
    output = tmp_path / 'trial.srt'
    write_subtitles(output, block_cues(blocks), font_size=18, ass_preview=True)
    rendered = subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720:d=4:r=4',
        '-vf', f'subtitles={output.with_suffix(".ass")}', '-frames:v', '16', '-threads', '1',
        '-pix_fmt', 'gray', '-f', 'rawvideo', '-'], check=True, capture_output=True, timeout=30).stdout
    size = 1280 * 720
    frames = [rendered[i * size:(i + 1) * size] for i in range(16)]
    assert all(f == frames[0] for f in frames[:12])
    ys = [y for y in range(720) if max(frames[0][y * 1280:(y + 1) * 1280]) > 100]
    assert ys and max(ys) - min(ys) + 1 <= 144
    assert all(max(f) < 100 for f in frames[12:])
