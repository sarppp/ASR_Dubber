"""
canary_patch.py — NeMo Canary inference patches for uv/NeMo >= 2.1

Three things this module fixes at runtime so Canary can do inference:

1. EOS assert  — canary2.py:209 asserts answer_ids ends with EOS, which fails
                  during inference because there's no ground-truth answer text.
                  We patch the assert out since we don't need it for decoding.

2. Manifest lang — NeMo writes source_lang/target_lang as 'en' into the temp
                   JSONL manifest regardless of what you ask for. We intercept
                   builtins.open for /tmp/*.json writes and force-overwrite those
                   fields so lhotse sees the correct language pair.

3. TranscriptionConfig — MultiTaskTranscriptionConfig schema changed in NeMo 2.1+.
                          We introspect fields at runtime and build the right kwargs.
"""

import builtins
import importlib
import inspect
import json as _json
import logging

log = logging.getLogger("nemo_local")

# ── 1. EOS assert patch ──────────────────────────────────────────────────────

def patch_canary2_eos_assert() -> None:
    """
    Remove the EOS assertion from canary2's prompt formatter.

    During inference, lhotse treats audio as a supervised sample and tries
    to verify that tokenised answer_ids end with EOS — but we have no
    ground-truth text, so it always fails.  The assert is irrelevant for
    decoding (the model generates its own output), so we patch it to True.
    """
    try:
        import nemo.collections.common.prompts.canary2 as _c2
        src = inspect.getsource(_c2.canary2)
        if "answer_ids" not in src:
            return  # already patched or different build
        patched = src.replace(
            'encoded["answer_ids"][-1].item() == formatter.tokenizer.eos',
            'True  # EOS assert disabled for inference',
        ).replace(
            'AssertionError: Expected the last token in answer_ids to be EOS',
            'AssertionError: (patched out)',
        )
        globs = _c2.canary2.__globals__
        exec(compile(patched, inspect.getfile(_c2), "exec"), globs)
        if "canary2" in globs:
            _c2.canary2 = globs["canary2"]
            log.info("canary2 EOS assert patched for inference mode")
    except Exception as exc:
        log.warning(f"canary2 EOS patch failed ({exc}); attempting anyway")

# ── 2. Manifest language injection ──────────────────────────────────────────

class _ManifestWriter:
    """Wraps a file handle; rewrites lang fields in every JSONL line."""
    def __init__(self, inner, src_lang: str, tgt_lang: str):
        self._inner = inner
        self._src = src_lang
        self._tgt = tgt_lang

    def write(self, s: str):
        s = s.strip()
        if s:
            try:
                obj = _json.loads(s)
                if isinstance(obj, dict) and "audio_filepath" in obj:
                    obj["source_lang"] = self._src
                    obj["target_lang"] = self._tgt
                    obj["taskname"] = "asr"
                    obj["pnc"] = "yes"
                    s = _json.dumps(obj)
                    log.info(f"Manifest lang injected → source={self._src!r} target={self._tgt!r}")
            except Exception:
                pass
        return self._inner.write(s + "\n" if s and not s.endswith("\n") else s)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._inner.__exit__(*args)

def patch_manifest_lang(src_lang: str, tgt_lang: str) -> None:
    """
    Intercept builtins.open so NeMo's temp JSONL manifest gets correct lang fields.

    NeMo hardcodes source_lang='en'/target_lang='en' when writing the internal
    manifest. lhotse reads lang_field='target_lang' from that manifest to decide
    what language the model should output — so we must overwrite those fields
    before lhotse reads the file.

    Only installs once per process; re-calling updates the lang values in-place
    so chunk loops with different languages work correctly.
    """
    _orig_open = getattr(builtins, "_nemo_orig_open", None) or builtins.open

    def _patched_open(file, mode="r", *args, **kwargs):
        fh = _orig_open(file, mode, *args, **kwargs)
        if "w" in str(mode) and isinstance(file, str) and "/tmp/" in file and file.endswith(".json"):
            log.info(f"Intercepted manifest write: {file}")
            return _ManifestWriter(fh, src_lang, tgt_lang)
        return fh

    if not getattr(builtins, "_nemo_orig_open", None):
        builtins._nemo_orig_open = builtins.open
    builtins.open = _patched_open
    log.info(f"Manifest lang patch active: source={src_lang!r} target={tgt_lang!r}")

# ── 3. MultiTaskTranscriptionConfig builder ──────────────────────────────────

def _find_config_cls():
    for mod_path in [
        "nemo.collections.asr.models.aed_multitask_models",
        "nemo.collections.asr.parts.utils.transcribe_utils",
    ]:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, "MultiTaskTranscriptionConfig", None)
            if cls is not None:
                return cls
        except (ImportError, ModuleNotFoundError):
            continue
    return None

def _config_fields(cls) -> set:
    import dataclasses
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}

def _wrap_lang(code: str) -> str:
    code = code.strip().lower()
    return code if code.startswith("<|") else f"<|{code}|>"

def build_transcription_config(src_lang: str, tgt_lang: str):
    """
    Build a MultiTaskTranscriptionConfig that works with the installed NeMo build.

    NeMo 2.0:  flat fields (source_lang, target_lang, use_lhotse, …)
    NeMo 2.1+: prompt dict + timestamps bool; lang comes from the manifest file
               (lang_field='target_lang'), not from the config prompt slots.
    """
    cls = _find_config_cls()
    if cls is None:
        log.warning("MultiTaskTranscriptionConfig not found — will call transcribe() without override")
        return None

    valid = _config_fields(cls)
    log.info(f"MultiTaskTranscriptionConfig fields: {sorted(valid)}")

    kwargs: dict = {"batch_size": 1, "num_workers": 0}

    if "lang_field" in valid:
        kwargs["lang_field"] = "target_lang"

    if "prompt" in valid:
        # NeMo 2.1+ — lang goes via manifest (patched above); prompt slots
        # are included for belt-and-suspenders but lang_field is authoritative.
        try:
            from nemo.collections.common.prompts import canary2  # noqa
            kwargs["prompt"] = {
                "slots": {
                    "source_lang": _wrap_lang(src_lang),
                    "target_lang": _wrap_lang(tgt_lang),
                    "pnc": "<|pnc|>",
                }
            }
        except Exception:
            pass
        if "timestamps" in valid:
            kwargs["timestamps"] = False
        if "verbose" in valid:
            kwargs["verbose"] = False
    else:
        # NeMo 2.0 flat fields
        for k, v in {
            "use_lhotse": False, "prompt_format": None, "prompt_defaults": None,
            "source_lang": src_lang, "target_lang": tgt_lang,
            "task": "asr", "pnc": "yes", "channel_selector": None,
        }.items():
            if k in valid:
                kwargs[k] = v

    log.info(f"Building TranscriptionConfig with: {sorted(kwargs.keys())}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        log.warning(f"Config build failed ({exc}); trying empty init")
        try:
            return cls()
        except Exception as exc2:
            log.warning(f"Empty init failed ({exc2}); no override config")
            return None