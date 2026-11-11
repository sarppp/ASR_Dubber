"""Regression selection must not confuse historical processing with source pace."""
import copy
import json
from pathlib import Path

import pytest

from batch_dubbing.stress_review import build, choose_cases, make_page


def fixtures():
    cues=[dict(start=i*3.,end=i*3.+2.5,text='one two three four five six seven eight nine ten',
               speaker='a' if i%2 else 'b') for i in range(20)]
    segments=[dict(index=i,start=i*5.,end=i*5.+4.,text='A short answer.',rate=1.+(i%3)*.6)
              for i in range(10)]
    videos=[dict(id=ident,review_end=60,english_cues=copy.deepcopy(cues),
                 merged_segments=copy.deepcopy(segments),original_path=f'{ident}.mp4',dub_path=f'{ident}.de.mp4')
            for ident in ('bill_ackman','mind_control')]
    crispr=dict(id='crispr',review_end=60,english_cues=cues,original_path='crispr.mp4',dub_path='crispr.de.mp4')
    return {'videos':videos},crispr


def test_cases_cover_distinct_failure_modes_without_claiming_real_speech_measurement():
    review,crispr=fixtures()
    original=copy.deepcopy(review)
    cases=choose_cases(review,crispr)
    assert {c['id'] for c in cases}=={'calm_control','short_reply','tempo_transition','pause','speaker_change','dense_source'}
    assert all(0<=c['start']<c['end']<=60 and c['end']-c['start']<=45 for c in cases)
    assert all(c['corrected_take_status']=='not_tested' and c['acoustic_timing_status']=='not_measured' for c in cases)
    assert all(c['original_pace_classification']=='pending_listening' for c in cases)
    assert 'not verified articulation speed' in cases[-1]['selection_reason']
    assert review==original
    assert choose_cases(review,crispr)==cases


def test_dense_source_selection_is_independent_of_dub_speed():
    review,crispr=fixtures()
    first=choose_cases(review,crispr)[-1]
    for s in review['videos'][1]['merged_segments']:
        s['rate']=9.
    second=choose_cases(review,crispr)[-1]
    assert (first['start'],first['end'],first['selection_reason'])==(second['start'],second['end'],second['selection_reason'])


def write_fixture_tree(root):
    review,crispr=fixtures()
    files={
        'dubbing_naturalness_2026-10-28/review_data.json':review,
        'modal_control/batch_inputs/video_01.gpu_full.json':dict(duration_seconds=60,english=crispr['english_cues']),
        'dubbing_naturalness_2026-10-28/crispr_native/inputs.json':dict(source_metadata=dict(original_video='crispr.mp4',older_dub='crispr.de.mp4'))}
    for name,data in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
    for v in review['videos']+[crispr]:
        for key in ('original_path','dub_path'):
            (root/v[key]).write_bytes(b'fixture reference only')


def test_build_checks_references_and_does_not_overwrite_existing_review(tmp_path):
    write_fixture_tree(tmp_path)
    output=tmp_path/'review'
    result=build(tmp_path,output)
    assert result['status']=='prepared_no_new_synthesis'
    assert len(result['input_sha256'])==3
    before=(output/'review.html').read_bytes()
    with pytest.raises(FileExistsError):
        build(tmp_path,output)
    assert (output/'review.html').read_bytes()==before


def test_missing_media_fails_before_publishing(tmp_path):
    write_fixture_tree(tmp_path)
    (tmp_path/'crispr.mp4').unlink()
    with pytest.raises(ValueError,match='Missing or external'):
        build(tmp_path,tmp_path/'review')
    assert not (tmp_path/'review').exists()


def test_review_escapes_transcript_and_starts_with_no_approved_candidates(tmp_path):
    review,crispr=fixtures()
    cases=choose_cases(review,crispr)
    cases[0]['english_context'][0]['text']='<script>alert(1)</script>'
    page=make_page({'cases':cases},tmp_path/'out',tmp_path)
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page
    assert page.count('<fieldset disabled>')==6
    assert 'Current verdict: not established.' in page
    assert 'acoustic_timing_verified:false,processing_verified:false' in page
