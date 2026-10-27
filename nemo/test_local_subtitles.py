from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from local_subtitles import infer_chunk


def hypothesis(text, timestamps=None):
    return SimpleNamespace(text=text, timestamp=timestamps or {})


@pytest.fixture(autouse=True)
def no_gpu_needed(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())


def test_source_speech_without_alignment_uses_ctc_word_times():
    model = Mock()
    model.transcribe.return_value = [hypothesis("Speech plus an unalignable scream")]
    model.timestamps_asr_model.transcribe.return_value = [hypothesis("Real speech.", {
        "word": [{"word": "Real", "start": 0.4, "end": 0.8},
                 {"word": "speech.", "start": 0.9, "end": 1.5}],
    })]
    segments = infer_chunk(model, Path("chunk.wav"), "en", "en", 20, 30)
    assert segments == [{"start": 20.4, "end": 21.5, "text": "Real speech.",
                         "transcription_fallback": "ctc"}]
    assert model.timestamps_asr_model.transcribe.call_args.kwargs["timestamps"] is True


def test_punctuation_on_silence_does_not_create_subtitles():
    model = Mock()
    model.transcribe.return_value = [hypothesis("... !")]
    assert infer_chunk(model, Path("chunk.wav"), "en", "en", 0, 25) == []
    model.timestamps_asr_model.transcribe.assert_not_called()


def test_empty_ctc_result_does_not_invent_speech_or_timestamps():
    model = Mock()
    model.transcribe.return_value = [hypothesis("A hallucination")]
    model.timestamps_asr_model.transcribe.return_value = [hypothesis("")]
    assert infer_chunk(model, Path("chunk.wav"), "en", "en", 0, 25) == []


def test_translation_cannot_fall_back_to_source_language_text():
    model = Mock()
    model.transcribe.return_value = [hypothesis("Traduction sans horodatage")]
    with pytest.raises(RuntimeError, match="without timestamps"):
        infer_chunk(model, Path("chunk.wav"), "en", "fr", 0, 25)
    model.timestamps_asr_model.transcribe.assert_not_called()
