from pathlib import Path

import pytest

from bilingual_subtitles import (
    Cue,
    combine_tracks,
    combine_tracks_exact,
    group_sentence_cues,
    main,
    parse_srt,
    split_bilingual_cue,
    split_single_track,
    split_text_naturally,
    write_srt,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_parse_srt_accepts_multiline_text_and_dot_timestamps(tmp_path):
    source = _write(
        tmp_path / "source.srt",
        """
1
00:00:01.250 --> 00:00:03,500
erste Zeile
zweite Zeile
""",
    )

    assert parse_srt(source) == [Cue(1250, 3500, "erste Zeile zweite Zeile")]


def test_exact_mode_uses_timestamp_boundaries_not_cue_numbers():
    top = [Cue(0, 2000, "Hallo"), Cue(2000, 4000, "Welt")]
    bottom = [Cue(0, 1000, "Bonjour"), Cue(1000, 4000, "le monde")]

    assert combine_tracks_exact(top, bottom) == [
        Cue(0, 1000, "Hallo\nBonjour"),
        Cue(1000, 2000, "Hallo\nle monde"),
        Cue(2000, 4000, "Welt\nle monde"),
    ]


def test_combine_tracks_keeps_top_line_above_bottom_line_and_adds_labels():
    result = combine_tracks(
        [Cue(0, 1000, "Deutsch")],
        [Cue(0, 1000, "Français")],
        top_label="DE",
        bottom_label="FR",
    )

    assert result == [Cue(0, 1000, "[DE] Deutsch\n[FR] Français")]


def test_unpaired_intervals_are_kept_by_default_and_can_be_dropped():
    top = [Cue(0, 2000, "Deutsch")]
    bottom = [Cue(1000, 3000, "Français")]

    assert combine_tracks_exact(top, bottom) == [
        Cue(0, 1000, "Deutsch"),
        Cue(1000, 2000, "Deutsch\nFrançais"),
        Cue(2000, 3000, "Français"),
    ]
    assert combine_tracks_exact(top, bottom, keep_unpaired=False) == [
        Cue(1000, 2000, "Deutsch\nFrançais")
    ]


def test_adjacent_intervals_with_unchanged_text_are_coalesced():
    top = [Cue(0, 3000, "Deutsch")]
    bottom = [Cue(0, 1000, "Français"), Cue(1000, 3000, "Français")]

    assert combine_tracks_exact(top, bottom) == [
        Cue(0, 3000, "Deutsch\nFrançais")
    ]


def test_sentence_grouping_rebuilds_fragments_until_terminal_punctuation():
    cues = [
        Cue(0, 1000, "Das ist"),
        Cue(1000, 2000, "ein Satz."),
        Cue(3000, 4000, "Danach."),
    ]

    assert group_sentence_cues(cues) == [
        Cue(0, 2000, "Das ist ein Satz."),
        Cue(3000, 4000, "Danach."),
    ]


def test_cinema_mode_pairs_differently_split_sentence_groups_by_overlap():
    top = [Cue(200, 1000, "Hallo"), Cue(1000, 2200, "Welt.")]
    bottom = [Cue(0, 1800, "Bonjour le monde.")]

    assert combine_tracks(top, bottom) == [
        Cue(0, 2200, "Hallo Welt.\nBonjour le monde.")
    ]


def test_cinema_mode_can_drop_unpaired_groups():
    top = [Cue(0, 1000, "Nur Deutsch.")]
    bottom = [Cue(2000, 3000, "Seulement français.")]

    assert combine_tracks(top, bottom, keep_unpaired=False) == []


def test_natural_split_prefers_punctuation_and_avoids_dangling_article():
    text = "Sie sind sehr klein, und jede Schote wiegt etwa ein Gramm."

    parts = split_text_naturally(text, 2, max_chars=34)

    assert parts == ["Sie sind sehr klein,", "und jede Schote wiegt etwa ein Gramm."]
    assert not parts[0].endswith(" ein")


def test_long_bilingual_cue_is_split_into_equal_short_timed_chunks():
    cue = Cue(
        10_000,
        18_000,
        "Sie sind sechsmal heißer als Jalapeños, und sie sind sehr klein.\n"
        "Ils sont six fois plus forts que les jalapeños, et ils sont minuscules.",
    )

    chunks = split_bilingual_cue(cue, max_chars=38, min_duration_ms=1300)

    assert len(chunks) == 2
    assert chunks[0].start_ms == 10_000
    assert chunks[-1].end_ms == 18_000
    assert all(chunk.end_ms - chunk.start_ms >= 1300 for chunk in chunks)
    assert all(len(chunk.text.splitlines()) == 2 for chunk in chunks)
    assert all(max(map(len, chunk.text.splitlines())) <= 46 for chunk in chunks)


def test_number_and_unit_are_kept_together_when_splitting():
    parts = split_text_naturally(
        "Sie sind nur 1 bis 1,5 Zoll lang und wiegen ein Gramm.",
        2,
        max_chars=30,
    )

    assert all(not part.endswith("1,5") for part in parts[:-1])
    assert any("1,5 Zoll" in part for part in parts)


def test_long_pepper_sentence_becomes_four_two_line_cues():
    cue = Cue(
        118_560,
        126_320,
        "Sie sind etwa sechsmal heißer als ein Jalapeno, und sie sind klein, "
        "nur 1 bis 1,5 Zoll lang und wiegen ein Gramm pro Stück.\n"
        "Ils sont environ six fois plus cuits que les jalapenos, et ils sont "
        "minuscules, seulement 1 à 1,5 pouces de long, et pesent un gramme chaque.",
    )

    chunks = split_bilingual_cue(cue, max_chars=42, min_duration_ms=1300)

    assert len(chunks) == 4
    assert all(len(chunk.text.splitlines()) == 2 for chunk in chunks)
    assert all(max(map(len, chunk.text.splitlines())) <= 42 for chunk in chunks)
    assert any("1,5 Zoll" in chunk.text for chunk in chunks)
    assert chunks[0].start_ms == cue.start_ms
    assert chunks[-1].end_ms == cue.end_ms


def test_single_pepper_sentence_keeps_numeric_range_and_unit_together():
    source = [
        Cue(
            118_560,
            123_457,
            "Sie sind etwa sechsmal heißer als ein Jalapeno, und sie sind klein, "
            "nur 1 bis",
        ),
        Cue(123_457, 126_320, "1,5 Zoll lang und wiegen ein Gramm pro Stück."),
    ]

    chunks = split_single_track(source, max_chars=42, min_duration_ms=1300)

    assert len(chunks) == 4
    assert any("nur 1 bis 1,5 Zoll lang" in chunk.text for chunk in chunks)
    assert all(not chunk.text.endswith((" nur", " bis", " 1,5")) for chunk in chunks)
    assert all(len(chunk.text) <= 48 for chunk in chunks)


def test_single_track_mode_splits_long_cue_and_preserves_timing():
    source = [
        Cue(
            5_000,
            11_000,
            "Das ist ein langer Untertitel, aber er wird natürlich geteilt.",
        )
    ]

    chunks = split_single_track(source, max_chars=34, min_duration_ms=1300)

    assert len(chunks) == 2
    assert chunks[0].start_ms == 5_000
    assert chunks[-1].end_ms == 11_000
    assert all("\n" not in chunk.text for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks) == source[0].text


def test_single_track_cli_creates_default_chunked_output(tmp_path):
    source = _write(
        tmp_path / "movie.de.srt",
        "1\n00:00:00,000 --> 00:00:05,000\n"
        "Das ist ein langer Untertitel, aber er wird natürlich geteilt.",
    )

    assert main(["--input", str(source), "--max-chars", "34"]) == 0

    output = tmp_path / "movie.de.chunked.srt"
    assert output.exists()
    assert len(parse_srt(output)) == 2


def test_parse_srt_rejects_malformed_blocks(tmp_path):
    source = _write(tmp_path / "bad.srt", "1\nnot a timestamp\nText")

    with pytest.raises(ValueError, match="missing timestamp"):
        parse_srt(source)


def test_write_srt_round_trip(tmp_path):
    output = tmp_path / "bilingual.srt"
    expected = [Cue(3723004, 3725500, "Deutsch\nFrançais")]

    write_srt(output, expected)

    assert "01:02:03,004 --> 01:02:05,500" in output.read_text(encoding="utf-8")
    assert parse_srt(output) == [Cue(3723004, 3725500, "Deutsch Français")]


def test_cli_creates_output(tmp_path):
    top = _write(
        tmp_path / "de.srt",
        "1\n00:00:00,000 --> 00:00:01,000\nDeutsch",
    )
    bottom = _write(
        tmp_path / "fr.srt",
        "1\n00:00:00,000 --> 00:00:01,000\nFrançais",
    )
    output = tmp_path / "combined.srt"

    assert main(["--top", str(top), "--bottom", str(bottom), "-o", str(output)]) == 0
    assert output.exists()
    assert "Deutsch\nFrançais" in output.read_text(encoding="utf-8")


def test_input_language_labels_are_removed_without_removing_speaker_notes(tmp_path):
    source = _write(tmp_path / 'labels.srt',
                    '1\n00:00:00,000 --> 00:00:02,000\n'
                    '[EN] Hello\n[de] Hallo\n[FR] Bonjour\n[laughs]')
    assert parse_srt(source)[0].text == 'Hello Hallo Bonjour [laughs]'


def test_three_tracks_preserve_gaps_order_and_endpoints():
    from bilingual_subtitles import combine_multiple_tracks_exact
    tracks = [[Cue(0, 2000, 'Hello')], [Cue(500, 2500, 'Hallo')],
              [Cue(1000, 3000, 'Bonjour')]]
    assert combine_multiple_tracks_exact(tracks) == [
        Cue(0, 500, 'Hello'), Cue(500, 1000, 'Hello\nHallo'),
        Cue(1000, 2000, 'Hello\nHallo\nBonjour'),
        Cue(2000, 2500, 'Hallo\nBonjour'), Cue(2500, 3000, 'Bonjour')]


def test_multitrack_cli_writes_compact_styled_ass(tmp_path):
    paths = [_write(tmp_path / f'{lang}.srt',
                    f'1\n00:00:01,000 --> 00:00:03,000\n[{lang}] {text}')
             for lang, text in [('EN', 'Hello'), ('DE', 'Hallo'), ('FR', 'Bonjour')]]
    output = tmp_path / 'combined.ass'
    assert main(['--tracks', *map(str, paths), '--output', str(output),
                 '--font-size', '36']) == 0
    text = output.read_text()
    assert 'PlayResY: 720' in text
    assert 'Style: Default,DejaVu Sans,36,' in text
    assert r'Hello\NHallo\NBonjour' in text
    assert '[EN]' not in text and '[DE]' not in text and '[FR]' not in text
    assert '0:00:01.00,0:00:03.00' in text


def test_ass_escapes_source_style_overrides(tmp_path):
    from bilingual_subtitles import write_ass
    output = tmp_path / 'escaped.ass'
    write_ass(output, [Cue(0, 1000, r'{\fs100}Hello')])
    assert r'\{\\fs100\}Hello' in output.read_text()


def test_multitrack_cli_rejects_conflicting_inputs(tmp_path):
    with pytest.raises(SystemExit):
        main(['--tracks', 'en.srt', 'de.srt', '--input', 'fr.srt'])


@pytest.mark.parametrize('suffix', ['srt', 'ass'])
def test_subtitle_outputs_have_unicode_marker_and_keep_accents(tmp_path, suffix):
    from bilingual_subtitles import write_subtitles
    output = tmp_path / f'accents.{suffix}'
    text = 'Man weiß: Größe, für, schön.\nFrançais: déjà, où, cœur.'
    write_subtitles(output, [Cue(0, 3000, text)])
    assert output.read_bytes().startswith(b'\xef\xbb\xbf')
    decoded = output.read_bytes().decode('utf-8-sig')
    assert 'weiß' in decoded and 'Größe' in decoded and 'cœur' in decoded
    assert 'Ã' not in decoded
    ass = output.with_suffix('.ass')
    assert ass.exists() == (suffix == 'ass')


def test_ass_uses_one_size_for_short_and_long_captions(tmp_path):
    import re
    from bilingual_subtitles import write_ass, AssMeasurer, _ass_header
    short = 'uncovered something of great import.\nMan weiß, dass es so etwas wie ein viszerales Gefühl ist, dass\nVous savez, c’est un peu un sentiment viscéral que vous avez découvert'
    long = '\n'.join(['Eine sehr lange Übersetzung mit äöü und ß. ' * 5] * 3)
    output = tmp_path / 'capped.ass'
    write_ass(output, [Cue(0, 3000, short), Cue(3000, 9000, long)])
    text = output.read_text(encoding='utf-8-sig')
    size = int(re.search(r'Style: Default,DejaVu Sans,(\d+),', text).group(1))
    assert size < 38
    assert 'Maximum rendered subtitle height: 20%' in text
    events = [line for line in text.splitlines() if line.startswith('Dialogue:')]
    assert all(r'{\fs' not in event for event in events)
    with AssMeasurer(_ass_header(size, 20)) as renderer:
        for content in (short, long):
            left, top, right, bottom = renderer.bounds(content, size)
            assert bottom - top <= 144
            assert 20 <= left < right <= 1260


def test_cli_produces_ass_preview_only_when_requested(tmp_path):
    top = _write(tmp_path / 'de.srt', '1\n00:00:00,000 --> 00:00:03,000\nMan weiß.')
    bottom = _write(tmp_path / 'fr.srt', '1\n00:00:00,000 --> 00:00:03,000\nOn sait.')
    assert main(['--tracks', str(top), str(bottom), '--ass-preview']) == 0
    output = top.with_suffix('.multilingual.ass')
    assert output.read_bytes().startswith(b'\xef\xbb\xbf')
    text = output.read_text(encoding='utf-8-sig')
    assert 'Maximum rendered subtitle height: 20%' in text
    assert r'Man weiß.\NOn sait.' in text


def test_cli_rejects_impossible_height_limit():
    with pytest.raises(SystemExit):
        main(['--tracks', 'en.srt', 'de.srt', '--max-height-percent', '0'])


def test_ass_subcentisecond_transitions_never_overlap_two_caption_blocks(tmp_path):
    from bilingual_subtitles import write_ass
    output = tmp_path / 'short.ass'
    write_ass(output, [Cue(0, 1001, 'First'), Cue(1001, 1004, 'Transition'),
                       Cue(1004, 2000, 'Next')])
    events = [s.split(',',9) for s in output.read_text(encoding='utf-8-sig').splitlines()
              if s.startswith('Dialogue:')]
    assert [(s[1],s[2]) for s in events] == [
        ('0:00:00.00','0:00:01.00'), ('0:00:01.00','0:00:01.01'),
        ('0:00:01.01','0:00:02.00')]


def test_multitrack_default_advances_languages_together():
    from bilingual_subtitles import combine_multiple_tracks
    tracks = [[Cue(0, 2000, 'Hello')], [Cue(500, 2500, 'Hallo')],
              [Cue(1000, 3000, 'Bonjour')], [Cue(600, 2200, 'Merhaba')]]
    assert combine_multiple_tracks(tracks) == [
        Cue(0, 3000, 'Hello\nHallo\nBonjour\nMerhaba')]


def test_multitrack_keeps_real_silent_gaps_and_unpaired_text():
    from bilingual_subtitles import combine_multiple_tracks
    tracks = [[Cue(0, 2000, 'Hello.'), Cue(10000, 12000, 'Alone.')],
              [Cue(100, 2100, 'Hallo.')], [Cue(200, 2200, 'Bonjour.')]]
    assert combine_multiple_tracks(tracks) == [
        Cue(0, 2200, 'Hello.\nHallo.\nBonjour.'), Cue(10000, 12000, 'Alone.')]


def test_multitrack_does_not_chain_sentences_through_small_tail_overlaps():
    from bilingual_subtitles import combine_multiple_tracks
    tracks = [[Cue(0, 4000, 'First.'), Cue(4000, 8000, 'Second.')],
              [Cue(100, 4200, 'Erste.'), Cue(4200, 8200, 'Zweite.')],
              [Cue(0, 3900, 'Première.'), Cue(3900, 8000, 'Deuxième.')]]
    result = combine_multiple_tracks(tracks)
    assert [c.text for c in result] == ['First.\nErste.\nPremière.',
                                      'Second.\nZweite.\nDeuxième.']
    assert result[0].end_ms == result[1].start_ms


@pytest.fixture
def crispr_tracks():
    # Original 23-36 second sequence shown in the user's screenshots.
    return [
        [Cue(23360, 26600, 'When we first published Our research on CRISPR.'),
         Cue(27080, 29880, 'It was clear it was going to be a transformative'),
         Cue(29880, 33800, 'technology for research, and we hoped, of course, for impacting'),
         Cue(33800, 35160, "people's lives in the future.")],
        [Cue(25000, 28927, 'Als wir unsere Forschung zu CRISPR durchführten, war klar, dass es eine'),
         Cue(28927, 32527, 'transformative Technologie für die Forschung sein würde, und wir hofften natürlich,'),
         Cue(32527, 35800, 'dass sie das Leben der Menschen in Zukunft beeinflussen würde.')],
        [Cue(25000, 28307, 'Lorsque nous avons commencé notre recherche sur le CRISPR, il était clair'),
         Cue(28307, 31613, "qu'il allait devenir une technologie transformatrice pour la recherche et nous espérions"),
         Cue(31613, 34920, "bien sûr avoir un impact sur la vie des gens à l'avenir.")],
    ]


def test_crispr_shared_phrases_preserve_every_word_without_flashes(crispr_tracks):
    from bilingual_subtitles import combine_multiple_tracks
    result = combine_multiple_tracks(crispr_tracks)
    assert result[0].start_ms == 23360
    assert result[-1].end_ms == 35800
    assert all(len(c.text.splitlines()) == 3 for c in result)
    assert all(c.end_ms - c.start_ms >= 1300 for c in result)
    assert all(a.end_ms == b.start_ms for a, b in zip(result, result[1:]))
    assert all(len(line) <= 48 for c in result for line in c.text.splitlines())
    for index, track in enumerate(crispr_tracks):
        assert ' '.join(c.text.splitlines()[index] for c in result) == ' '.join(c.text for c in track)


def test_crispr_captions_have_constant_size_and_fit_rendered_frame(tmp_path, crispr_tracks):
    import re
    from bilingual_subtitles import combine_multiple_tracks, write_ass, AssMeasurer, _ass_header
    result = combine_multiple_tracks(crispr_tracks)
    output = tmp_path / 'crispr.ass'
    write_ass(output, result)
    text = output.read_text(encoding='utf-8-sig')
    size = int(re.search(r'Style: Default,DejaVu Sans,(\d+),', text).group(1))
    assert size == 38
    assert r'{\fs' not in text
    with AssMeasurer(_ass_header(size, 20)) as renderer:
        for cue in result:
            left, top, right, bottom = renderer.bounds(cue.text, size)
            assert 20 <= left < right <= 1260
            assert bottom - top <= 144


def test_multitrack_cli_honors_exact_boundaries(tmp_path):
    paths = [_write(tmp_path / f'{index}.srt', text) for index, text in enumerate([
        '1\n00:00:00,000 --> 00:00:02,000\nHello',
        '1\n00:00:00,500 --> 00:00:02,500\nHallo',
        '1\n00:00:01,000 --> 00:00:03,000\nBonjour'])]
    output = tmp_path / 'exact.srt'
    assert main(['--tracks', *map(str, paths), '--exact-boundaries', '-o', str(output)]) == 0
    assert len(parse_srt(output)) == 5
    assert main(['--tracks', *map(str, paths), '--allow-approximate-alignment', '-o', str(output)]) == 0
    assert len(parse_srt(output)) == 1


@pytest.mark.parametrize('option', ['--max-chars', '--min-duration-ms'])
def test_multitrack_rejects_nonpositive_chunk_limits(option):
    with pytest.raises(SystemExit):
        main(['--tracks', 'en.srt', 'de.srt', option, '0'])


def test_multitrack_cli_uses_chunk_settings(tmp_path, crispr_tracks):
    paths = [tmp_path / f'{index}.srt' for index in range(3)]
    for path, track in zip(paths, crispr_tracks):
        write_srt(path, track)
    output = tmp_path / 'combined.srt'
    args = ['--tracks', *map(str, paths), '--allow-approximate-alignment', '-o', str(output)]
    assert main(args + ['--max-chars', '42']) == 0
    short = parse_srt(output)
    assert main(args + ['--max-chars', '60', '--min-duration-ms', '3000']) == 0
    long = parse_srt(output)
    assert len(long) < len(short)
    assert all(c.end_ms - c.start_ms >= 3000 for c in long)


def test_multitrack_preserves_intentional_repetitions():
    from bilingual_subtitles import combine_multiple_tracks
    tracks = [[Cue(0, 2000, 'Yes.'), Cue(2000, 4000, 'Yes.')],
              [Cue(0, 4000, 'Ja. Ja.')], [Cue(0, 4000, 'Oui. Oui.')]]
    result = combine_multiple_tracks(tracks)
    assert result == [Cue(0, 4000, 'Yes. Yes.\nJa. Ja.\nOui. Oui.')]


def test_srt_is_primary_and_strips_per_cue_size_markup(tmp_path):
    from bilingual_subtitles import write_subtitles
    path = tmp_path / 'captions.srt'
    write_subtitles(path, [Cue(0, 2000, '<font size="80">Hello &amp; goodbye</font>\n'
                              r'{\fs90}Hallo & tschüss')])
    assert not path.with_suffix('.ass').exists()
    assert 'Hello & goodbye\nHallo & tschüss' in path.read_text(encoding='utf-8-sig')
    assert '<font' not in path.read_text() and r'\fs' not in path.read_text()


def test_srt_overflow_fails_without_shrinking_an_ass_file_or_replacing_output(tmp_path):
    from bilingual_subtitles import write_subtitles
    path = tmp_path / 'captions.srt'
    path.write_text('existing subtitle')
    with pytest.raises(ValueError, match='SRT cue 1 does not fit'):
        write_subtitles(path, [Cue(0, 2000, 'W' * 160 + '\nHallo\nBonjour')])
    assert path.read_text() == 'existing subtitle'
    assert not path.with_suffix('.ass').exists()


def test_independent_tracks_require_explicit_approximation(tmp_path, crispr_tracks):
    paths = [tmp_path / f'{i}.srt' for i in range(3)]
    for path, cues in zip(paths, crispr_tracks):
        write_srt(path, cues)
    output = tmp_path / 'combined.srt'
    assert main(['--tracks', *map(str, paths), '-o', str(output)]) == 1
    assert not output.exists()


def test_aligned_translation_phrases_are_never_split_independently(tmp_path):
    from bilingual_subtitles import write_subtitles
    tracks = [
        [Cue(29022, 31172, 'a transformative technology for research,'),
         Cue(31172, 32969, 'and we hoped, of course,')],
        [Cue(29022, 31172, 'eine transformative Technologie für die Forschung,'),
         Cue(31172, 32969, 'und wir hofften natürlich,')],
        [Cue(29022, 31172, 'une technologie transformatrice pour la recherche,'),
         Cue(31172, 32969, 'et nous espérions bien sûr,')],
    ]
    paths = [tmp_path / f'{i}.srt' for i in range(3)]
    for path, cues in zip(paths, tracks):
        write_srt(path, cues)
    output = tmp_path / 'combined.srt'
    assert main(['--tracks', *map(str, paths), '--max-chars', '10', '-o', str(output)]) == 0
    blocks = output.read_text(encoding='utf-8-sig').strip().split('\n\n')
    assert len(blocks) == 2
    assert blocks[1].splitlines()[2:] == [track[1].text for track in tracks]
    assert 'transformatrice' not in blocks[1]
    assert 'hofften' not in blocks[0] and 'espérions' not in blocks[0]
