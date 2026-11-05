"""Local sentence-aware translation for the subtitle pipeline.

Translation follows the flow that was validated against Ling 3.0:

  1. cues are grouped into complete sentences;
  2. each sentence is translated once, whole, by a local Ollama model;
  3. the result is cut back onto the cue boundaries by character weight.

Models that ship without a chat template (Tower-Plus-2B, TranslateGemma) get
their turn markers added here; without them the model continues the prompt
instead of translating it.

Default translator: tower-plus-2b:latest.
Optional:          translategemma:4b (the previous Gemma translator).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "tr": "Turkish",
}

DEFAULT_MT_MODEL = os.getenv("TRANSLATE_MODEL", "tower-plus-2b:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

_GEMMA_TURN = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
_GEMMA_STOP = ["<end_of_turn>", "<eos>"]

# Model families whose GGUF carries no chat template.
MODEL_FAMILIES = {
    "tower": {"template": _GEMMA_TURN, "stop": _GEMMA_STOP},
    "translategemma": {"template": _GEMMA_TURN, "stop": _GEMMA_STOP},
    "gemma": {"template": _GEMMA_TURN, "stop": _GEMMA_STOP},
}

MAX_SENTENCE_CUES = 6
MAX_SENTENCE_SECONDS = 30.0


def model_family(model_name: str) -> dict | None:
    base = (model_name or "").split(":")[0].lower()
    for key, spec in MODEL_FAMILIES.items():
        if key in base:
            return spec
    return None


def apply_chat_template(prompt: str, model_name: str) -> str:
    family = model_family(model_name)
    return prompt if family is None else family["template"].format(prompt=prompt)


def clean_output(text: str) -> str:
    for marker in ("<|endoftext|>", "<end_of_turn>", "<bos>", "<eos>"):
        text = (text or "").replace(marker, "")
    text = re.sub(r"<[^>]{1,60}>", "", text)
    return text.strip()


def _extract_json_text(text: str) -> str:
    """Pull the "text" value out of a JSON answer, tolerating noise.

    The word-budget prompt asks for JSON. Models sometimes wrap it in a code
    fence or add a sentence around it, so fall back to the raw text rather
    than losing the translation.
    """
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict) and value.get("text"):
            return str(value["text"]).strip()
    except (ValueError, TypeError):
        pass
    match = re.search(r'"text"\s*:\s*"(.+?)"', raw, re.DOTALL)
    if match:
        return match.group(1).replace('\\"', '"').strip()
    return raw


def _looks_translated(text: str, source_text: str, source: str, target: str) -> bool:
    """True when `text` looks like it is in the target language, not the source.

    A translation model asked to answer in JSON sometimes echoes the source
    sentence back inside the JSON. The check needs no word list: it compares
    the answer with the source sentence by character bigrams. Two texts in
    different languages share far fewer bigrams than a text and its own copy
    do.

    `source_text` is the sentence that was sent; `source` and `target` are
    the language codes.
    """
    if not text.strip():
        return False
    if source == target:
        return True
    source_grams = _bigrams(source_text)
    answer_grams = _bigrams(text)
    if not answer_grams:
        return False
    shared = len(source_grams & answer_grams) / len(answer_grams)
    # Measured on real output: a copy of the source shares 1.00 of its
    # bigrams, a genuine translation shares about 0.49. The threshold sits
    # well between the two.
    return shared < 0.75


def _bigrams(text: str) -> set[str]:
    """The distinct character pairs of a text, lowercased."""
    clean = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return {clean[i:i + 2] for i in range(len(clean) - 1)}


def split_translation(text: str, weights: list[int]) -> list[str] | None:
    """Cut a sentence translation onto its cues by character weight."""
    words = text.split()
    if not words or len(words) < len(weights):
        return None
    total = sum(weights)
    pieces, prev, cumulative = [], 0, 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        target = round(len(words) * cumulative / total)
        low = prev + 1
        high = len(words) - (len(weights) - index - 1)
        boundary = min(max(target, low), high)
        pieces.append(" ".join(words[prev:boundary]))
        prev = boundary
    return pieces


def group_sentences(cues: list[dict]) -> list[list[dict]]:
    """Group consecutive cues into complete sentences.

    A group closes on terminal punctuation, after MAX_SENTENCE_CUES cues, or
    once it spans MAX_SENTENCE_SECONDS — whichever comes first.
    """
    groups, run = [], []
    for cue in cues:
        run.append(cue)
        text = (cue.get("text") or "").strip()
        span = float(run[-1].get("end", 0.0)) - float(run[0].get("start", 0.0))
        if re.search(r"[.!?]$", text) or len(run) >= MAX_SENTENCE_CUES or span >= MAX_SENTENCE_SECONDS:
            groups.append(run)
            run = []
    if run:
        groups.append(run)
    return groups


@dataclass
class OllamaTranslator:
    model: str = DEFAULT_MT_MODEL
    host: str = OLLAMA_HOST
    timeout: int = 300

    def generate(self, prompt: str, num_predict: int = 400) -> str:
        family = model_family(self.model) or {}
        options = {"temperature": 0.1, "top_p": 0.9, "num_ctx": 2048,
                   "num_predict": num_predict}
        if family.get("stop"):
            options["stop"] = family["stop"]
        body = {
            "model": self.model,
            "prompt": apply_chat_template(prompt, self.model),
            "raw": True,
            "stream": False,
            "keep_alive": "2m",
            "options": options,
        }
        request = urllib.request.Request(
            self.host.rstrip("/") + "/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        text = clean_output(payload.get("response") or "")
        return text.split("\n\n")[0].strip().strip('"').strip()

    def translate_sentence(self, text: str, source: str, target: str,
                           max_chars: int | None = None,
                           max_words: int | None = None) -> str:
        """Translate one complete sentence.

        `max_chars` is a soft guide: models ignore a character budget (they
        return the same text whatever the budget says). `max_words` is the
        reliable control — a model asked for "at most N words" counts far
        more reliably, especially inside a JSON object, because it then has a
        structure to fill rather than a number to hit.

        Measured in tools/probe_control.py: a 143-character budget returned
        190-247 characters (1.4-1.7x over) for every model tried, while a
        14-word budget inside JSON returned 15 words.

        The JSON form has to name the target language FIRST. Asked to answer
        in JSON after the translation instructions, Tower echoes the English
        source back instead of translating it.
        """
        if max_words:
            prompt = (
                f"Translate the {LANG_NAMES.get(source, source)} sentence into "
                f"{LANG_NAMES.get(target, target)} ({target}).\n"
                f"Answer as JSON: {{\"text\": \"<the {target} translation>\"}}\n"
                f"Use at most {max_words} words. Keep every name, number, negation "
                f"and fact; compress the wording, never drop information."
            )
            result = _extract_json_text(
                self.generate(prompt + f"\n\n{text}"))
            # A model that ignored the JSON and echoed the source would be
            # worse than no answer at all, so it is rejected here.
            if not _looks_translated(result, text, source, target):
                result = ""
            return result

        prompt = (
            f"You are a professional subtitle translator from "
            f"{LANG_NAMES.get(source, source)} ({source}) to "
            f"{LANG_NAMES.get(target, target)} ({target}).\n\n"
            f"Translate this complete sentence. Keep names, numbers, negation and "
            f"facts. Do not add commentary. Answer with the translation only, on one line."
        )
        if max_chars:
            prompt += (f"\nThe translation must be spoken aloud in the same time as the "
                       f"original, so use at most about {max_chars} characters. Keep every "
                       f"fact; compress the wording, never drop information.")
        return self.generate(prompt + f"\n\n{text}")

    def adapt_sentence(self, text: str, language: str, max_chars: int) -> str:
        """Compress an already translated line into a hard character budget.

        Cinema-style adaptation: same facts, shorter wording. It must not add
        information and must not drop a verb, number or negation.
        """
        prompt = (
            f"You are a dubbing adapter for {LANG_NAMES.get(language, language)} "
            f"({language}).\n\n"
            f"Rewrite the line below so it can be spoken at a natural pace inside its "
            f"subtitle timing. Use at most {max_chars} characters.\n"
            f"Rules: keep every fact, name, number and negation; never invent content; "
            f"remove filler words and redundant clauses; prefer shorter verbs and "
            f"contractions. Answer with the rewritten line only, one line, no quotes."
        )
        return self.generate(prompt + f"\n\n{text}", num_predict=300)

    def unload(self) -> None:
        body = {"model": self.model, "keep_alive": 0}
        try:
            request = urllib.request.Request(
                self.host.rstrip("/") + "/api/generate",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:
            pass


# Characters the dub voice speaks per second, measured per engine.
# The old single constant (de 11.6) was ~1.9x too low for Qwen and made the
# gate shorten lines that already fitted their slot. See nemo/speech_rate.py
# and nemo/dub_rate_findings.md.
try:
    from speech_rate import DEFAULT_RATES as TTS_CHARS_PER_SECOND
    from speech_rate import character_budget as _character_budget
except ImportError:  # pragma: no cover - direct import without the nemo path
    TTS_CHARS_PER_SECOND = {"en": 15.58, "de": 14.82, "fr": 16.16}
    _character_budget = None

DEFAULT_CHARS_PER_SECOND = 14.0
DUB_MAX_SPEED = 1.15
EDGE_CLEANUP = 0.97


def speech_budget(span: float, language: str, engine: str | None = None) -> int:
    """Characters that fit `span` seconds at the engine's natural rate.

    The dub never speeds the voice up, so this is the whole allowance: the
    measured rate, a little edge cleanup, and a little headroom.
    """
    if _character_budget is not None:
        return _character_budget(span, language, engine)
    cps = float(os.getenv("SUBTITLE_TTS_CPS") or TTS_CHARS_PER_SECOND.get(
        language, DEFAULT_CHARS_PER_SECOND))
    return max(20, int(span * cps * EDGE_CLEANUP * 0.95))


def translate_cues(cues: list[dict], source: str, target: str,
                   translator: OllamaTranslator | None = None,
                   progress=None, fit_budget: bool = True,
                   engine: str | None = None) -> list[dict]:
    """Translate cue-level subtitles sentence-by-sentence. Returns new cues.

    Each sentence is translated whole and then cut back onto its cues, so the
    model never sees a fragment and the cue boundaries stay untouched.
    """
    translator = translator or OllamaTranslator()
    out: list[dict] = []
    groups = group_sentences(cues)
    for number, group in enumerate(groups, 1):
        sentence = " ".join((c.get("text") or "").strip() for c in group).strip()
        if not sentence:
            out.extend(group)
            continue
        budget = None
        if fit_budget:
            span = float(group[-1].get("end", 0.0)) - float(group[0].get("start", 0.0))
            budget = speech_budget(span, target, engine)
        translated = translator.translate_sentence(sentence, source, target, max_chars=budget)
        weights = [max(1, len((c.get("text") or "").strip())) for c in group]
        pieces = split_translation(translated, weights) if translated else None
        if pieces is None:
            pieces = [(c.get("text") or "") for c in group]
            if translated:
                pieces[0] = translated
        for cue, piece in zip(group, pieces):
            out.append({**cue, "text": piece})
        if progress:
            progress(number, len(groups), translated[:70])
    return out
