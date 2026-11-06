"""Rebuild an approved continuous-take finish and compare it without inference.

This deliberately tests timing/caption/mux code, not fresh ASR, MT or TTS.
All inputs, including cached alignments and expected outputs, are hash-bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

from batch_dubbing.aligned_finish import finish, passage_spans, read_srt, sha, source_plan, write_srt
from batch_dubbing.anchored_timeline import packet_hash, video_duration


ROLES = ('video', 'source_srt', 'target_srt', 'raw_take', 'source_alignment',
         'target_alignment', 'reference_video', 'reference_wav',
         'reference_source_srt', 'reference_target_srt')


def verified_assets(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('version') != 1:
        raise ValueError('Unsupported reference manifest version')
    assets = {}
    roles=ROLES+(('adjusted_alignment',) if manifest.get('placement_mode')=='continuous_synced' else ())
    for role in roles:
        entry = manifest['assets'][role]
        path = (manifest_path.parent / entry['path']).resolve()
        if sha(path) != entry['sha256']:
            raise ValueError(f'Reference input changed: {role}')
        assets[role] = path
    return manifest, assets


def pcm_identity(path):
    digest = hashlib.sha256()
    with wave.open(str(path)) as wav:
        parameters = [wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()]
        while block := wav.readframes(65536):
            digest.update(block)
    return dict(parameters=parameters, sha256=digest.hexdigest())


def stream_layout(path):
    info = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)], timeout=30))
    return [dict(type=s['codec_type'], codec=s['codec_name'],
                 language=s.get('tags', {}).get('language', 'und'),
                 default=s['disposition']['default']) for s in info['streams']]


def stream_hash(path, selector):
    return subprocess.check_output(
        ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', selector,
         '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-'], text=True, timeout=60).strip()


def compare_artifacts(assets, result, source_language, target_language, *, placement_mode='source_anchored'):
    result = Path(result)
    movie = result / 'final_dub.mp4'
    reference = assets['reference_video']
    layout = stream_layout(movie)
    expected_layout = stream_layout(reference)
    actual_pcm, expected_pcm = pcm_identity(result / 'dub_track.wav'), pcm_identity(assets['reference_wav'])
    checks = dict(audio_pcm_bit_identical=actual_pcm == expected_pcm,
                  video_packets_unchanged=packet_hash(movie) == packet_hash(assets['video']) == packet_hash(reference),
                  full_video_duration_preserved=abs(video_duration(movie) - video_duration(assets['video'])) < .001,
                  encoded_audio_packets_identical=stream_hash(movie, '0:a:0') == stream_hash(reference, '0:a:0'))
    if placement_mode!='continuous':checks['embedded_track_layout_identical']=layout == expected_layout
    captions = {}
    for lang, role in ((source_language, 'reference_source_srt'), (target_language, 'reference_target_srt')):
        actual = result / f'final_dub.{lang}.srt'
        expected = assets[role]
        rows, reference_rows = read_srt(actual), read_srt(expected)
        captions[lang] = dict(cue_count=len(rows), expected_cue_count=len(reference_rows),
                              byte_identical=actual.read_bytes() == expected.read_bytes(),
                              cuts_and_text_identical=[(r['index'], r['text']) for r in rows] == [(r['index'], r['text']) for r in reference_rows],
                              timings_identical=[(r['start'], r['end']) for r in rows] == [(r['start'], r['end']) for r in reference_rows])
        checks[f'{lang}_subtitle_byte_identical'] = captions[lang]['byte_identical']
    if placement_mode!='continuous':
        for i in range(sum(s['type'] == 'subtitle' for s in expected_layout)):
            checks[f'embedded_subtitle_{i}_packets_identical'] = (
                layout == expected_layout and stream_hash(movie, f'0:s:{i}') == stream_hash(reference, f'0:s:{i}'))
    else:
        # The selected continuous MP4 embeds obsolete captions. Validate the
        # new embedded tracks against its selected aligned external SRTs.
        import tempfile
        iso={'en':'eng','de':'deu','fr':'fra','es':'spa','it':'ita'}
        expected_subs=[dict(type='subtitle',codec='mov_text',language=iso.get(lang,lang),default=int(i==0))
                       for i,lang in enumerate((source_language,target_language))]
        checks['embedded_tracks_use_aligned_sidecars']= [s for s in layout if s['type']=='subtitle']==expected_subs
        with tempfile.TemporaryDirectory() as tmp:
            for i,(lang,role) in enumerate(((source_language,'reference_source_srt'),(target_language,'reference_target_srt'))):
                raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(movie),'-map',f'0:s:{i}','-f','srt','-'],timeout=60)
                extracted=Path(tmp)/f'{lang}.srt';extracted.write_bytes(raw)
                checks[f'embedded_{lang}_matches_aligned_reference']=read_srt(extracted)==read_srt(assets[role])
    return dict(status='match' if all(checks.values()) else 'different', checks=checks,
                captions=captions, actual_pcm=actual_pcm, expected_pcm=expected_pcm,
                embedded_tracks=layout, expected_embedded_tracks=expected_layout)


def reproduce(manifest_path, output):
    manifest, assets = verified_assets(manifest_path)
    output = Path(output).resolve()
    # Never overwrite either accepted evidence or a prior test result.
    output.mkdir(parents=True, exist_ok=False)
    src, dst = manifest['source_language'], manifest['target_language']
    source_cues, target_cues = read_srt(assets['source_srt']), read_srt(assets['target_srt'])
    if len(source_cues) != len(target_cues):
        raise ValueError('Reference scripts have different cue coverage')
    spoken = output / f'spoken.{dst}.srt'
    write_srt(spoken, [dict(start=source_cues[a]['start'], end=source_cues[b]['end'],
                           text=' '.join(c['text'] for c in target_cues[a:b+1]))
                       for a, b in passage_spans(source_cues)])
    source_words = json.loads(assets['source_alignment'].read_text())['words']
    timing = output / 'source_timing.json'
    timing.write_text(json.dumps(dict(word_evidence='Cached CPU forced alignment, hash-bound by reference manifest',
                                     words=[dict(word=w.get('word', w.get('text')), start=w['start'], end=w['end']) for w in source_words])))
    plan = source_plan(assets['source_srt'], spoken, {src: assets['source_srt'], dst: assets['target_srt']}, src, dst, source_timing=timing)
    alignment = json.loads(assets['target_alignment'].read_text())
    mode=manifest.get('placement_mode','source_anchored')
    request = dict(video=str(assets['video']), output=str(output / 'result'), plan=plan,placement_mode=mode,
                   takes=[dict(index=1, file=str(assets['raw_take']), text=' '.join(p['text'] for p in plan['passages']),
                               tempo=1., sha256=sha(assets['raw_take']))],
                   bundles=[[p['index'] for p in plan['passages']]])
    (output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    def cached_alignment(audio,text,language):
        digest=sha(audio)
        if digest==sha(assets['raw_take']):return alignment['words']
        if 'adjusted_alignment' in assets:
            adjusted=json.loads(assets['adjusted_alignment'].read_text())
            if digest in adjusted.get('media_sha256',{}).values():return adjusted['words']
        raise ValueError('No hash-bound alignment for the regenerated recording')
    finish(request,cached_alignment)
    comparison = compare_artifacts(assets, output / 'result', src, dst, placement_mode=mode)
    comparison.update(manifest_sha256=sha(manifest_path), synthesis_calls=0, model_calls=0,
                      placement_mode=mode,reference_files={role:str(assets[role]) for role in ROLES if role.startswith('reference_')},
                      scope='Reuses the approved raw voice and cached alignments; does not test fresh ASR, translation or synthesis')
    (output / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing outputs are never overwritten')
    args = parser.parse_args()
    result = reproduce(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] != 'match':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
