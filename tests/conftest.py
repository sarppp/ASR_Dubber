"""
tests/conftest.py — pytest configuration for all ASR pipeline tests.
Sets up sys.path and a dummy SRT env so translate_diarize.py can import.
"""
import os
import sys
import tempfile
import types
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                            # pipeline_utils, pipeline_paths
sys.path.insert(0, str(ROOT / "translate-gemma"))        # translate_diarize, translate_utils
sys.path.insert(0, str(ROOT / "nemo"))                   # nemo_audio, nemo_diarize, nemo_model
sys.path.insert(0, str(ROOT / "qwen3-tts"))              # dub_audio, dub_srt, dub

# ── Torch stub ────────────────────────────────────────────────────────────────
# nemo_audio.py imports torch at module level. Provide a minimal stub so the
# nemo unit tests run without CUDA or the full torch install.
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _cuda = mock.MagicMock()
    _cuda.is_available = lambda: False
    _cuda.mem_get_info = lambda: (0, 0)
    _torch.cuda = _cuda
    _torch.inference_mode = mock.MagicMock(return_value=mock.MagicMock(
        __enter__=lambda s, *a: None, __exit__=lambda s, *a: None))
    sys.modules["torch"] = _torch

# ── omegaconf stub ────────────────────────────────────────────────────────────
if "omegaconf" not in sys.modules:
    _omegaconf = types.ModuleType("omegaconf")
    _omegaconf.OmegaConf = mock.MagicMock()
    sys.modules["omegaconf"] = _omegaconf

# translate_diarize.py has module-level code that calls sys.exit(1)
# when INPUT_DIR has no .srt files. Create a minimal one so import works.
_tmp = tempfile.mkdtemp()
_dummy = os.path.join(_tmp, "dummy.nemo.en.diarize.srt")
with open(_dummy, "w") as _f:
    _f.write("1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n")

os.environ.setdefault("INPUT_DIR", _tmp)
os.environ.setdefault("TARGET_LANG_CODE", "fr")
os.environ.setdefault("SOURCE_LANG_CODE", "en")
