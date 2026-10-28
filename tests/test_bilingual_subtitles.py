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
    from bilingual_subtitles import combine_multiple_tracks
    tracks = [[Cue(0, 2000, 'Hello')], [Cue(500, 2500, 'Hallo')],
              [Cue(1000, 3000, 'Bonjour')]]
    assert combine_multiple_tracks(tracks) == [
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
    assert ass.exists() and ass.read_bytes().startswith(b'\xef\xbb\xbf')


def test_ass_default_keeps_selected_size_and_shrinks_long_captions(tmp_path):
    import re
    from bilingual_subtitles import write_ass, AssMeasurer, _ass_header
    short = 'uncovered something of great import.\nMan weiß, dass es so etwas wie ein viszerales Gefühl ist, dass\nVous savez, c’est un peu un sentiment viscéral que vous avez découvert'
    long = '\n'.join(['Eine sehr lange Übersetzung mit äöü und ß. ' * 5] * 3)
    output = tmp_path / 'capped.ass'
    write_ass(output, [Cue(0, 3000, short), Cue(3000, 9000, long)])
    text = output.read_text(encoding='utf-8-sig')
    assert 'Style: Default,DejaVu Sans,38,' in text
    assert 'Maximum rendered subtitle height: 20%' in text
    events = [line for line in text.splitlines() if line.startswith('Dialogue:')]
    assert r'{\fs' not in events[0]
    reduced = int(re.search(r'\{\\fs(\d+)\}', events[1]).group(1))
    assert reduced < 38
    with AssMeasurer(_ass_header(38, 20)) as renderer:
        for content, size in [(short, 38), (long, reduced)]:
            left, top, right, bottom = renderer.bounds(content, size)
            assert bottom - top <= 144
            assert 20 <= left < right <= 1260


def test_default_cli_produces_twenty_percent_ass_companion(tmp_path):
    top = _write(tmp_path / 'de.srt', '1\n00:00:00,000 --> 00:00:03,000\nMan weiß.')
    bottom = _write(tmp_path / 'fr.srt', '1\n00:00:00,000 --> 00:00:03,000\nOn sait.')
    assert main(['--tracks', str(top), str(bottom)]) == 0
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
