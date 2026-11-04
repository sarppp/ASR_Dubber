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

    def translate_sentence(self, text: str, source: str, target: str) -> str:
        prompt = (
            f"You are a professional subtitle translator from "
            f"{LANG_NAMES.get(source, source)} ({source}) to "
            f"{LANG_NAMES.get(target, target)} ({target}).\n\n"
            f"Translate this complete sentence. Keep names, numbers, negation and "
            f"facts. Do not add commentary. Answer with the translation only, on one line.\n\n"
            f"{text}"
        )
        return self.generate(prompt)

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


def translate_cues(cues: list[dict], source: str, target: str,
                   translator: OllamaTranslator | None = None,
                   progress=None) -> list[dict]:
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
        translated = translator.translate_sentence(sentence, source, target)
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
