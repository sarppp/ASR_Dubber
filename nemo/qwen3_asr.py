"""
qwen3_asr.py — Qwen3-ASR model loading and transcription.

Uses the qwen-asr package (pip install qwen-asr), NOT NeMo.
Supports 30+ languages with word-level timestamps via ForcedAligner.

Models:
  qwen3-asr      Qwen/Qwen3-ASR-1.7B   30 langs, ~5GB VRAM, word timestamps
  qwen3-asr-s    Qwen/Qwen3-ASR-0.6B   30 langs, ~2GB VRAM, faster
"""

import logging

log = logging.getLogger("nemo_local")

# ISO 639-1 → Qwen3-ASR language name (None = auto-detect)
QWEN3_LANG_MAP = {
    "en": "English",    "de": "German",     "fr": "French",
    "es": "Spanish",    "it": "Italian",    "nl": "Dutch",
    "pt": "Portuguese", "ru": "Russian",    "zh": "Chinese",
    "ja": "Japanese",   "ko": "Korean",     "ar": "Arabic",
    "tr": "Turkish",    "hi": "Hindi",      "vi": "Vietnamese",
    "th": "Thai",       "pl": "Polish",     "cs": "Czech",
    "sv": "Swedish",    "da": "Danish",     "fi": "Finnish",
    "el": "Greek",      "hu": "Hungarian",  "ro": "Romanian",
    "uk": "Ukrainian",  "id": "Indonesian", "ms": "Malay",
    "fa": "Persian",    "fil": "Filipino",  "mk": "Macedonian",
}


def _is_qwen3_asr(model_name: str) -> bool:
    return "Qwen3-ASR" in model_name or "qwen3-asr" in model_name.lower()


def _load_qwen3_asr(model_name: str, device: str, precision: str):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        raise RuntimeError(
            "qwen-asr package not installed. Run: pip install qwen-asr"
        )
    import torch

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.bfloat16)
    device_map = "cuda:0" if device == "cuda" else "cpu"

    log.info(f"Loading Qwen3-ASR: {model_name} [{dtype}] on {device_map}")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
    )
    log.info("Qwen3-ASR model ready")
    return model


def _transcribe_qwen3_asr(model, audio_path: str, offset: float,
                           src_lang: str) -> tuple[list, list]:
    from nemo_audio import _audio_duration, _strip_special_tokens

    lang_name = QWEN3_LANG_MAP.get(src_lang)  # None = auto-detect

    try:
        results = model.transcribe(
            audio=audio_path,
            language=lang_name,
            return_time_stamps=True,
        )
    except Exception as exc:
        log.error(f"Qwen3-ASR transcribe failed: {exc}"); raise

    if not results:
        return [], []

    result = results[0]
    text = _strip_special_tokens(getattr(result, "text", "") or "")

    all_words = []
    for ts in (getattr(result, "time_stamps", None) or []):
        word = _strip_special_tokens(getattr(ts, "text", "") or "").strip()
        if not word:
            continue
        all_words.append({
            "word": word,
            "start": float(getattr(ts, "start_time", 0.0)) + offset,
            "end":   float(getattr(ts, "end_time",   0.0)) + offset,
        })

    if not all_words and text:
        # ForcedAligner unavailable — fall back to segment-level
        dur = _audio_duration(audio_path)
        seg = {
            "text": text,
            "start": offset,
            "end": offset + (dur if dur > 0 else max(1.0, len(text.split()) * 0.4)),
        }
        return [], [seg]

    return all_words, []
