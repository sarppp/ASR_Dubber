"""Offline tests for pipeline_subtitle (no models, no TTS)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline_subtitle as ps  # noqa: E402


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        input_file=None, input_dir=None, output_dir=None,
        target_lang="de", language="en", trim=0, whisper_model="base",
        subtitle_languages=None, asr_model=None, translator=None,
        skip_dub=False, run_mode="full", fit_target=1.0, max_sentences=None,
        tts_engine="supertonic", supertonic_voice="M1", qwen_mode="clone",
        no_demucs=False, max_speed=1.15, min_speed=1.0,
        merge_gap=1.0, merge_max_dur=10.0,
        tts_workers=1, tts_devices="0",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_languages_default_to_source_plus_target():
    assert ps._subtitle_languages(_args(), "en", "de") == ["en", "de"]
    assert ps._subtitle_languages(_args(), "de", "en") == ["de", "en"]
    assert ps._subtitle_languages(_args(subtitle_languages=["fr"]), "en", "de") == ["fr", "de", "en"]
    assert ps._subtitle_languages(_args(subtitle_languages=["de", "fr"]), "en", "de") == ["de", "fr", "en"]


def test_run_produces_subtitles_and_dubs_target(tmp_path, monkeypatch):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"not a real video")
    nemo_dir = tmp_path / "nemo"
    nemo_dir.mkdir()

    monkeypatch.setattr(ps, "QWEN_DIR", tmp_path / "qwen3-tts")
    calls = []

    def fake_run(cmd, cwd=None, label=None, env=None, check=True):
        calls.append([str(c) for c in cmd])
        if "batch_subtitles.py" in cmd[1]:
            work = Path(cmd[cmd.index("--input-dir") + 1])
            stem = "talk"
            langs = [cmd[cmd.index("--languages") + 1 + i]
                     for i in range(len(cmd) - cmd.index("--languages") - 1)]
            for lang in langs:
                (work / f"{stem}.nemo.{lang}.srt").write_text(
                    f"1\n00:00:00,000 --> 00:00:01,000\n[{lang}]\n", encoding="utf-8")
            for suffix in (".multi.srt", ".multi.ass"):
                (work / f"{stem}{suffix}").write_text("multi", encoding="utf-8")
        return 0

    class _Translator:
        def translate_sentence(self, text, source, language, max_chars=None, max_words=None):
            return " ".join(text.split()[:max(1, max_chars // 6)])

    monkeypatch.setattr(ps, "_make_compressor", lambda *a: None)
    monkeypatch.setattr(ps, "_calibrated_rate", lambda *a, **k: 20.0)
    monkeypatch.setattr(ps, "_run", fake_run)
    monkeypatch.setattr(ps, "OllamaTranslator", _Translator)
    monkeypatch.setattr(ps, "_detect_source_language", lambda *a, **k: "en")
    monkeypatch.setattr(ps, "_find_video", lambda **k: video)
    monkeypatch.setattr(ps, "_validate_translated_srt", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_finalize_outputs", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_derive_run_label", lambda *a, **k: "talk__de")

    rc = ps.run(_args(input_file=str(video), input_dir=str(nemo_dir), output_dir=str(tmp_path / "out")))
    assert rc == 0
    assert len(calls) == 2, calls

    dub = calls[1]
    assert "dub.py" in dub[1]
    assert "--source-script" in dub and "--comparison-script" in dub
    assert dub[dub.index("--placement-mode")+1] == "continuous_synced"
    assert "--strict-timing" not in dub
    assert "--language" in dub and dub[dub.index("--language") + 1] == "de"
    assert dub[dub.index("--language") - 1].endswith("talk.nemo.de.dub.srt")

    # with --input-file the video folder wins, same as the legacy runner
    out = video.parent
    assert (out / "talk.nemo.en.srt").exists()
    assert (out / "talk.nemo.de.srt").exists()
    assert (out / "talk.multi.srt").exists()
    assert (out / "talk.multi.ass").exists()


def test_subtitle_command_uses_backend_local_and_selected_models(tmp_path, monkeypatch):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    nemo_dir = tmp_path / "nemo"
    nemo_dir.mkdir()
    seen = {}

    class _Translator:
        def translate_sentence(self, text, source, language, max_chars=None, max_words=None):
            return text

    monkeypatch.setattr(ps, "OllamaTranslator", _Translator)

    def fake_run(cmd, cwd=None, label=None, env=None, check=True):
        if "batch_subtitles.py" in cmd[1]:
            seen["cmd"] = [str(c) for c in cmd]
            work = Path(cmd[cmd.index("--input-dir") + 1])
            for lang in ("en", "de"):
                (work / f"talk.nemo.{lang}.srt").write_text(
                    "1\n00:00:00,000 --> 00:00:02,000\n"
                    + ("Hallo Welt" if lang == "de" else "Hello world") + "\n",
                    encoding="utf-8")
            (work / "talk.multi.srt").write_text("m", encoding="utf-8")
        return 0

    class _Translator:
        def translate_sentence(self, text, source, language, max_chars=None, max_words=None):
            return " ".join(text.split()[:max(1, max_chars // 6)])

    monkeypatch.setattr(ps, "_make_compressor", lambda *a: None)
    monkeypatch.setattr(ps, "_calibrated_rate", lambda *a, **k: 20.0)
    monkeypatch.setattr(ps, "_run", fake_run)
    monkeypatch.setattr(ps, "OllamaTranslator", _Translator)
    monkeypatch.setattr(ps, "_detect_source_language", lambda *a, **k: "en")
    monkeypatch.setattr(ps, "_find_video", lambda **k: video)
    monkeypatch.setattr(ps, "_validate_translated_srt", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_finalize_outputs", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_derive_run_label", lambda *a, **k: "run")

    ps.run(_args(input_file=str(video), input_dir=str(nemo_dir), output_dir=str(tmp_path / "out"),
                 asr_model="parakeet-ultra", translator="tower", skip_dub=True))
    cmd = seen["cmd"]
    assert "--backend" in cmd and cmd[cmd.index("--backend") + 1] == "local"
    assert cmd[cmd.index("--asr-model") + 1] == "parakeet-ultra"
    assert cmd[cmd.index("--translator") + 1] == "tower"


if __name__ == "__main__":
    import tempfile

    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, value in reversed(self._undo):
                setattr(obj, name, value)

    for test in (test_languages_default_to_source_plus_target,
                 test_run_produces_subtitles_and_dubs_target,
                 test_subtitle_command_uses_backend_local_and_selected_models):
        with tempfile.TemporaryDirectory() as tmp:
            mp = _MP()
            try:
                if test.__code__.co_argcount > 1:
                    test(Path(tmp), mp)
                else:
                    test()
            finally:
                mp.undo()
        print("ok", test.__name__)
