import json
from pathlib import Path

import pytest

from batch_dubbing.replay_asr_checkpoint import prepare, run
from batch_dubbing.aligned_finish import sha


def reference():
    manifest=Path(__file__).resolve().parents[1]/'batch_dubbing/crispr_29_synced_reference.json'
    entries=json.loads(manifest.read_text())['assets'].values()
    if not all((manifest.parent/e['path']).is_file() for e in entries):
        pytest.skip('Local reference media is not stored in Git')
    return manifest


def test_replay_preserves_original_cues_but_does_not_copy_reference_speech(tmp_path):
    output=tmp_path/'replay';request,manifest,assets=prepare(reference(),output)
    stage=output/'input/.subtitle_dub'/request['base']
    assert sha(stage/f'{request["base"]}.nemo.en.srt')==sha(assets['source_srt'])
    assert sha(stage/f'{request["base"]}.nemo.de.srt')==sha(assets['target_srt'])
    assert not list(output.rglob('*.wav'))
    assert not (output/'result').exists()
    command=request['command']
    assert command[2].endswith('/run_pipeline.py')
    assert command[command.index('--dub-timing')+1]=='continuous-synced'
    assert command[command.index('--dub-retries')+1]=='0'
    assert 'original' in request['cached_stages'].lower()
    with pytest.raises(FileExistsError):prepare(reference(),output)


def test_replay_waits_for_busy_gpu_before_creating_any_files(tmp_path,monkeypatch):
    monkeypatch.setattr('batch_dubbing.replay_asr_checkpoint.subprocess.check_output',lambda *a,**k:'12345\n')
    with pytest.raises(RuntimeError,match='GPU occupied'):
        run(tmp_path/'no-manifest-needed',tmp_path/'result')
    assert not (tmp_path/'result').exists()
