import json
from pathlib import Path

import pytest

from batch_dubbing.aligned_finish import finish, sha
from batch_dubbing.reference_regression import ROLES, compare_artifacts, reproduce
from test_aligned_finish import align, fixtures


@pytest.mark.parametrize('filename',['crispr_29_reference.json','crispr_29_continuous_reference.json','crispr_29_synced_reference.json'])
def test_approved_29_second_reference_when_local_media_is_available(tmp_path,filename):
    manifest = Path(__file__).resolve().parents[1] / 'batch_dubbing' / filename
    entries = json.loads(manifest.read_text())['assets'].values()
    if not all((manifest.parent / entry['path']).is_file() for entry in entries):
        pytest.skip('Approved media is local evidence and is not stored in Git')
    result = reproduce(manifest, tmp_path / 'reference-result')
    assert result['status'] == 'match', result['checks']
    assert result['model_calls'] == 0


def test_changed_reference_is_rejected_before_output_creation(tmp_path):
    asset = tmp_path / 'asset'
    asset.write_text('approved evidence')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(version=1, assets={
        role: dict(path='asset', sha256=sha(asset)) for role in ROLES})))
    asset.write_text('silently changed evidence')
    with pytest.raises(ValueError, match='Reference input changed: video'):
        reproduce(manifest, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_comparison_catches_caption_drift_even_when_audio_video_match(tmp_path):
    request = fixtures(tmp_path)
    finish(request, align)
    result = tmp_path / 'out'
    expected = tmp_path / 'expected.en.srt'
    expected.write_bytes((result / 'final_dub.en.srt').read_bytes())
    assets = dict(video=tmp_path / 'source.mp4', reference_video=result / 'final_dub.mp4',
                  reference_wav=result / 'dub_track.wav', reference_source_srt=expected,
                  reference_target_srt=result / 'final_dub.de.srt')
    assert compare_artifacts(assets, result, 'en', 'de')['status'] == 'match'
    # A re-cut caption can conserve every word and still break the approved pair.
    target = result / 'final_dub.en.srt'
    target.write_bytes(target.read_bytes().replace(b'One complete sentence.', b'One complete\r\nsentence.'))
    comparison = compare_artifacts(assets, result, 'en', 'de')
    assert comparison['status'] == 'different'
    assert comparison['checks']['audio_pcm_bit_identical']
    assert comparison['checks']['video_packets_unchanged']
    assert not comparison['captions']['en']['cuts_and_text_identical']
    assert comparison['captions']['en']['timings_identical']
