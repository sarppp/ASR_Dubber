import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

# We need to import the module to test it
from dub import main

@pytest.fixture
def mock_args(mocker):
    # Mock sys.argv to control arguments passed to argparse
    return mocker.patch.object(sys, 'argv', ['dub.py'])

@pytest.fixture
def mock_deps(mocker, tmp_path):
    # Setup mock dependencies for dub.py's main execution flow
    deps = MagicMock()
    
    # Mock dub_srt functions
    deps.parse_srt = mocker.patch('dub.parse_srt')
    deps.merge_segments = mocker.patch('dub.merge_segments')
    deps.build_voice_map = mocker.patch('dub.build_voice_map')
    
    # Mock dub_audio functions
    deps.extract_audio = mocker.patch('dub.extract_audio')
    deps.separate_audio = mocker.patch('dub.separate_audio')
    deps.extract_clone_refs = mocker.patch('dub.extract_clone_refs')
    deps._qwen_python = mocker.patch('dub._qwen_python', return_value="python3")
    deps._qwen_worker = mocker.patch('dub._qwen_worker', return_value="worker.py")
    deps.PersistentTTSWorker = mocker.patch('dub.PersistentTTSWorker')
    deps.speed_fit = mocker.patch('dub.speed_fit')
    deps.stitch_and_mix = mocker.patch('dub.stitch_and_mix')
    deps._save_checkpoint = mocker.patch('dub._save_checkpoint')
    deps._load_checkpoint = mocker.patch('dub._load_checkpoint', return_value=[])
    
    # Setup default return values corresponding to success
    deps.parse_srt.return_value = [
        {"index": 1, "speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 2.0}
    ]
    deps.merge_segments.return_value = deps.parse_srt.return_value
    deps.build_voice_map.return_value = {"SPEAKER_00": "female_voice_1"}
    
    # Mock the returned audio paths for separation
    deps.separate_audio.return_value = (tmp_path / "vocals.wav", tmp_path / "background.wav")
    
    # Mock clone refs extraction
    deps.extract_clone_refs.return_value = {"SPEAKER_00": tmp_path / "clone_ref.wav"}
    
    # Mock worker behavior
    mock_worker_instance = MagicMock()
    mock_worker_instance.generate_clone.return_value = True
    mock_worker_instance.generate_custom.return_value = True
    deps.PersistentTTSWorker.return_value = mock_worker_instance
    # Mock path check for generating raw out so loop continues, but don't break pathlib
    original_exists = Path.exists
    def exists_side_effect(self, *args, **kwargs):
        if "seg_" in str(self) and ".wav" in str(self):
            return True
        return original_exists(self, *args, **kwargs)
    mocker.patch('pathlib.Path.exists', exists_side_effect)

    original_stat = Path.stat
    def stat_side_effect(self, *args, **kwargs):
        if "seg_" in str(self) and ".wav" in str(self):
            return MagicMock(st_size=0)
        return original_stat(self, *args, **kwargs)
    mocker.patch('pathlib.Path.stat', stat_side_effect)
    
    # Mock speed fit return
    deps.speed_fit.return_value = tmp_path / "fitted.wav"
    
    # Mock stitch return
    deps.stitch_and_mix.return_value = tmp_path / "final.mp4"
    
    return deps

@pytest.fixture
def mock_fs(mocker, tmp_path):
    """Mocks file system interactions finding inputs."""
    # Create the virtual input files
    video_file = tmp_path / "test_video.mp4"
    srt_file = tmp_path / "test_video.nemo.en.diarize_fr.srt"
    
    # We patch Path.exists but we need to ensure we don't break pathlib
    original_exists = Path.exists
    def side_effect(self, *args, **kwargs):
        if str(self) == str(video_file) or str(self) == str(srt_file):
            return True
        return original_exists(self, *args, **kwargs)
        
    mocker.patch('pathlib.Path.exists', side_effect)
    
    return {"video": video_file, "srt": srt_file, "search_dir": tmp_path}


def test_main_missing_inputs_no_discovery(mock_args, mocker, tmp_path):
    """Test behavior when no inputs are provided and discovery fails (empty dir)."""
    # Empty search dir
    mocker.patch('sys.argv', ['dub.py', '--search-dir', str(tmp_path)])
    
    # Should exit 1 because no SRTs found
    assert main() == 1


def test_main_auto_discovery_success(mock_args, mock_deps, tmp_path, mocker):
    """Test that video and SRT are correctly auto-discovered in search dir."""
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    
    # Create fake files to discover
    (search_dir / "my_video.mp4").touch()
    (search_dir / "my_video.nemo.de.diarize_fr.srt").touch()
    
    mocker.patch('sys.argv', ['dub.py', '--search-dir', str(search_dir)])
    
    # We need to unpatch Path.exists from the general mock if any
    
    result = main()
    assert result == 0
    # Verify srt was parsed
    assert mock_deps.parse_srt.call_count == 1
    passed_srt = mock_deps.parse_srt.call_args[0][0]
    assert passed_srt.name == "my_video.nemo.de.diarize_fr.srt"


def test_main_explicit_inputs(mock_deps, mock_fs, mocker, tmp_path):
    """Test successful run with explicit inputs and demucs enabled."""
    mocker.patch('sys.argv', [
        'dub.py', str(mock_fs['video']), str(mock_fs['srt']),
        '--workdir', str(tmp_path / "work")
    ])
    
    assert main() == 0
    
    # Verify demucs was called (default behavior)
    mock_deps.separate_audio.assert_called_once()
    mock_deps.extract_audio.assert_not_called()
    
    # Verify stitch and mix received the background audio
    stitch_kwargs = mock_deps.stitch_and_mix.call_args[1]
    assert stitch_kwargs.get('background') is not None


def test_main_no_demucs_flag(mock_deps, mock_fs, mocker, tmp_path):
    """Test --no-demucs skips separation and skips clone ref extraction if not cloning."""
    mocker.patch('sys.argv', [
        'dub.py', str(mock_fs['video']), str(mock_fs['srt']),
        '--no-demucs', '--qwen-mode', 'custom',
        '--workdir', str(tmp_path / "work")
    ])
    
    assert main() == 0
    
    # Verify extract_audio was not called because we are in custom mode
    mock_deps.extract_audio.assert_not_called()
    mock_deps.separate_audio.assert_not_called()
    mock_deps.extract_clone_refs.assert_not_called()
    
    # Verify stitch and mix background was None
    stitch_kwargs = mock_deps.stitch_and_mix.call_args[1]
    assert stitch_kwargs.get('background') is None


def test_main_no_demucs_with_clone(mock_deps, mock_fs, mocker, tmp_path):
    """Test --no-demucs WITH --qwen-mode clone uses extract_audio instead of separate."""
    mocker.patch('sys.argv', [
        'dub.py', str(mock_fs['video']), str(mock_fs['srt']),
        '--no-demucs', '--qwen-mode', 'clone',
        '--workdir', str(tmp_path / "work")
    ])
    
    assert main() == 0
    
    # Verify extract_audio called for clone refs, but separate_audio skipped
    mock_deps.extract_audio.assert_called_once()
    mock_deps.separate_audio.assert_not_called()
    mock_deps.extract_clone_refs.assert_called_once()


def test_main_clone_fallback_to_custom(mock_deps, mock_fs, mocker, tmp_path):
    """Test that if clone generation fails, it falls back to custom voice generation."""
    mocker.patch('sys.argv', [
        'dub.py', str(mock_fs['video']), str(mock_fs['srt']),
        '--qwen-mode', 'clone',
        '--workdir', str(tmp_path / "work")
    ])
    
    # Needs a mock clone ref to attempt generation
    mock_deps.extract_clone_refs.return_value = {"SPEAKER_00": tmp_path / "clone.wav"}
    (tmp_path / "clone.wav").touch()
    
    # Make clone generation return False (fail)
    mock_worker_instance = mock_deps.PersistentTTSWorker.return_value
    mock_worker_instance.generate_clone.return_value = False
    
    assert main() == 0
    
    # Verify both were attempted
    mock_worker_instance.generate_clone.assert_called_once()
    mock_worker_instance.generate_custom.assert_called_once()


def test_main_no_segments_parsed(mock_deps, mock_fs, mocker):
    """Test that the script exits with error if SRT parsing yields no segments."""
    mocker.patch('sys.argv', ['dub.py', str(mock_fs['video']), str(mock_fs['srt'])])
    
    # Empty segments list
    mock_deps.parse_srt.return_value = []
    
    # Should exit with code 1
    assert main() == 1
    
    # Separation shouldn't be reached
    mock_deps.separate_audio.assert_not_called()


def test_main_all_tts_failed(mock_deps, mock_fs, mocker, tmp_path):
    """Test that the script exits with error if no outputs were generated for any segments."""
    mocker.patch('sys.argv', [
        'dub.py', str(mock_fs['video']), str(mock_fs['srt']),
        '--workdir', str(tmp_path / "work")
    ])
    
    # Make all generations fail
    mock_worker_instance = mock_deps.PersistentTTSWorker.return_value
    mock_worker_instance.generate_clone.return_value = False
    mock_worker_instance.generate_custom.return_value = False
    
    assert main() == 1
    
    # Stitching shouldn't be reached since final_files is empty
    mock_deps.stitch_and_mix.assert_not_called()
