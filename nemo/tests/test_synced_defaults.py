"""The ordinary CLI must reach the accepted workflow without opt-in flags."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'qwen3-tts'))
import run_pipeline as runner
import pipeline_subtitle as pipeline


def test_default_cli_dispatches_confirmed_dub_command(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(sys, 'argv', ['run_pipeline.py', '--target-lang', 'de'])
    monkeypatch.setattr(pipeline, 'QWEN_DIR', tmp_path)
    monkeypatch.setattr(pipeline, '_validate_translated_srt', lambda *a: None)
    monkeypatch.setattr(pipeline, '_run', lambda cmd, **kw: commands.append(cmd) or 0)

    def dispatch(args):
        return pipeline._dub(args, tmp_path/'source.mp4', tmp_path/'de.srt',
                             'source', 'de', source_srt=tmp_path/'en.srt',
                             subtitles={'en': tmp_path/'en.srt', 'de': tmp_path/'de.srt'},
                             source_language='en')[1]
    monkeypatch.setattr(pipeline, 'run', dispatch)
    assert runner.main() == 0
    cmd = commands[0]
    for option, value in [('--tts-engine', 'qwen'), ('--qwen-mode', 'clone'),
                          ('--placement-mode', 'continuous_synced'),
                          ('--max-speed', '1.0'), ('--merge-gap', '0')]:
        assert cmd[cmd.index(option)+1] == value
    assert '--no-demucs' in cmd
    assert '--source-script' in cmd and cmd.count('--comparison-script') == 2


@pytest.mark.parametrize('flags,engine,dry', [
    ([], 'qwen', True),
    (['--pipeline', 'legacy-dubbing'], 'supertonic', False),
    (['--tts-engine', 'supertonic', '--preserve-background'], 'supertonic', False),
    (['--pipeline', 'legacy-dubbing', '--tts-engine', 'qwen', '--no-demucs'], 'qwen', True),
])
def test_pipeline_defaults_preserve_explicit_choices(flags, engine, dry):
    args = runner._resolve_dubbing_defaults(runner.build_parser().parse_args(flags))
    assert args.tts_engine == engine
    assert args.no_demucs is dry


@pytest.mark.parametrize('timing', ['continuous-synced', 'continuous'])
def test_failed_synthesis_cannot_rewrite_frozen_captions(tmp_path, monkeypatch, timing):
    video = tmp_path/'source.mp4'; video.touch()
    cues = {}
    for lang, text in [('en', 'A sentence.'), ('de', 'Ein Satz.')]:
        path = tmp_path/f'source.nemo.{lang}.srt'
        path.write_text(f'1\n00:00:00,000 --> 00:00:02,000\n{text}\n')
        cues[lang] = path
    args = runner._resolve_dubbing_defaults(runner.build_parser().parse_args([
        '--input-file', str(video), '--output-dir', str(tmp_path/'out'),
        '--language', 'en', '--target-lang', 'de', '--dub-timing', timing]))
    monkeypatch.setattr(pipeline, 'QWEN_DIR', tmp_path/'qwen')
    monkeypatch.setattr(pipeline, '_produce_subtitles', lambda *a, **kw: cues)
    monkeypatch.setattr(pipeline, '_calibrated_rate', lambda *a, **kw: 20.)
    translator = Mock()
    monkeypatch.setattr(pipeline, 'OllamaTranslator', lambda: translator)
    repair = Mock(side_effect=AssertionError('Frozen captions must not be rewritten'))
    monkeypatch.setattr(pipeline, '_adapt_from_report', repair)
    monkeypatch.setattr(pipeline, '_make_compressor', repair)
    work = tmp_path/'failed'; work.mkdir()
    (work/'fit_report.json').write_text('{"status":"failed"}')
    dub = Mock(return_value=(work, 1)); monkeypatch.setattr(pipeline, '_dub', dub)
    with pytest.raises(SystemExit) as error:
        pipeline.run(args)
    assert error.value.code == 1
    dub.assert_called_once()
    repair.assert_not_called()
    translator.translate_sentence.assert_not_called()
    assert 'Ein Satz.' in cues['de'].read_text()


@pytest.mark.parametrize('override', [None, '/explicit/model'])
def test_continuous_clone_loads_pinned_model_without_inference(monkeypatch, override):
    import qwen_tts_worker as worker
    torch = SimpleNamespace(manual_seed=Mock(), float16='fp16')
    loader = Mock(return_value=SimpleNamespace())
    download = Mock(return_value='/pinned/model')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'faster_qwen3_tts', SimpleNamespace(
        FasterQwen3TTS=SimpleNamespace(from_pretrained=loader)))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    monkeypatch.setenv('QWEN_TTS_CONTINUOUS', '1')
    monkeypatch.delenv('FORCE_SLOW_TTS', raising=False)
    monkeypatch.delenv('QWEN_TTS_MODEL_PATH', raising=False)
    if override:
        monkeypatch.setenv('QWEN_TTS_MODEL_PATH', override)
    worker._load_model('clone', 'cuda:0')
    if override:
        download.assert_not_called()
    else:
        download.assert_called_once_with('Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                                         revision='fd4b254389122332181a7c3db7f27e918eec64e3')
    loader.assert_called_once_with(override or '/pinned/model', device='cuda:0',
                                   dtype='fp16', max_seq_len=1024)
    torch.manual_seed.assert_called_once_with(20261105)
