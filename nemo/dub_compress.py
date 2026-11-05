"""Shortening a translated line so it fits its slot, automatically.

The natural-pace dub must never speed the voice up, so a line that does not
fit has to be made shorter. The translation model is asked to do that, but
no single prompt form works on every model:

* Tower (the default) will not shorten a German line at all - it returns the
  same text whatever the budget says. Asked to rewrite, it answers in
  English.
* hymt2 will shorten, but sometimes drops a fact or returns nothing.
* eurollm answers in English even when told to keep the language.

So the adapter tries the available compressors in turn and keeps the first
answer that is genuinely shorter, in the RIGHT language, and still carries
the content. Every candidate goes through three checks:

* it must be shorter than the original;
* it must be in the target language (a candidate in another language is
  rejected - otherwise the dub would speak the wrong language);
* it must carry the content (the same meaning guard the rest of the
  pipeline uses: a number, a negation and the amount of substance survive).

Nothing here is hand-tuned per language: the compressors are asked for the
target language by name and every answer is checked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# The rewrite prompt. The model is told to keep the language explicitly,
# because a translation model asked to "rewrite" a German line will happily
# answer in English.
REWRITE_PROMPT = (
    "Rewrite this {language} subtitle line so it is shorter, in {language}. "
    "Keep every fact, name, number and negation. Compress the wording only; "
    "never drop information and never translate it. Use at most {words} words. "
    'Answer as JSON: {{"text": "<the shortened {language} line>"}}'
)

# A second form: ask for the shorter line without the JSON wrapper. Some
# models answer the plain form better.
REWRITE_PLAIN = (
    "Rewrite this {language} subtitle line so it is shorter, in {language}. "
    "Keep every fact, name, number and negation. Compress the wording only; "
    "never drop information and never translate it. Use at most {words} words. "
    "Answer with the shortened line only, one line, no explanation."
)


@dataclass
class Candidate:
    text: str
    engine: str
    form: str
    words: int
    chars: int


class Compressor:
    """Shortens a translated line using the translation models.

    `engines` is the list of translators to try, in order. The first one is
    the default translator; the others are fallbacks. The adapter keeps the
    first candidate that is shorter than the original, in the target
    language, and passes the meaning guard, so a model that cannot shorten
    never breaks the run.
    """

    def __init__(self, engines: list, language: str = "de",
                 language_name: str = "German"):
        self.engines = engines
        self.language = language
        self.language_name = language_name

    def candidates(self, text: str, max_words: int) -> list[Candidate]:
        """Every shortening the engines offer, shortest first."""
        found: list[Candidate] = []
        for engine in self.engines:
            for form, prompt in (("rewrite", REWRITE_PROMPT),
                                 ("plain", REWRITE_PLAIN)):
                try:
                    raw = engine.generate(
                        prompt.format(language=self.language_name,
                                      words=max_words) + f"\n\n{text}",
                        num_predict=500)
                except Exception:
                    continue
                candidate = _clean(raw, form)
                if not candidate:
                    continue
                found.append(Candidate(
                    text=candidate,
                    engine=getattr(engine, "model", "?"),
                    form=form,
                    words=len(candidate.split()),
                    chars=len(candidate)))
        found.sort(key=lambda c: c.chars)
        return found

    def shorten(self, text: str, max_words: int, guard=None,
                source_text: str | None = None,
                fits=None, max_rounds: int = 4) -> tuple[str, dict]:
        """The shortest valid shortening of `text`.

        `guard(before, after)` returns a reason when the candidate loses
        something it must not lose, or None when it is acceptable.
        `source_text` is the original sentence in the source language; it is
        used to tell a same-language answer from a translation.
        `fits(candidate_text)` says whether the candidate is short enough;
        when it is given, the compressor keeps shortening until the result
        fits, instead of stopping at the first valid shortening.

        Returns the chosen text and a report. When nothing valid is found the
        original is returned, so the meaning is never silently replaced.
        """
        report = {"attempted": 0, "rejected": [], "engine": None, "form": None,
                  "rounds": 0}
        current = text
        best: tuple[str, dict] | None = None

        for round_index in range(max_rounds):
            report["rounds"] = round_index + 1
            round_report = {"attempted": 0, "rejected": [], "engine": None,
                            "form": None}
            for candidate in self.candidates(current, max_words):
                report["attempted"] += 1
                round_report["attempted"] += 1
                if len(candidate.text) >= len(current):
                    round_report["rejected"].append({
                        "engine": candidate.engine, "form": candidate.form,
                        "reason": "not shorter"})
                    continue
                if not same_language(current, candidate.text, source_text):
                    round_report["rejected"].append({
                        "engine": candidate.engine, "form": candidate.form,
                        "reason": "answered in another language"})
                    continue
                if guard is not None:
                    problem = guard(current, candidate.text)
                    if problem:
                        round_report["rejected"].append({
                            "engine": candidate.engine, "form": candidate.form,
                            "reason": problem})
                        continue
                round_report["engine"] = candidate.engine
                round_report["form"] = candidate.form
                round_report["chars"] = len(candidate.text)

                # Keep the best candidate seen so far.
                if best is None or len(candidate.text) < len(best[0]):
                    best = (candidate.text, dict(round_report))
                if fits is None or fits(candidate.text):
                    report.update({k: v for k, v in round_report.items()
                                   if k != "rejected"})
                    report["rejected"].extend(round_report["rejected"])
                    return candidate.text, report
                # Short enough to be valid but not short enough to fit: use
                # it as the input for the next round.
                current = candidate.text
                max_words = max(3, int(max_words * 0.85))
                break
            else:
                # No candidate in this round; stop.
                break
            report["rejected"].extend(round_report["rejected"])

        if best is not None:
            report.update({k: v for k, v in best[1].items() if k != "rejected"})
            return best[0], report
        report["kept_original"] = True
        return text, report


def same_language(original: str, candidate: str,
                  source_text: str | None = None) -> bool:
    """True when the candidate is in the same language as the original.

    The check needs no word list and no model. Two texts in the same
    language share a lot of their character bigrams.

    German and English are the hard case: "Technologie" and "technology"
    share enough that a naive overlap score cannot separate them. The
    reliable signal is the SOURCE sentence: a candidate that matches the
    original language more than the source language is in the original
    language. Measured on real output: a German candidate scores 1.00
    against the German original and 0.47 against the English source; an
    English candidate scores 0.59 and 0.85.
    """
    from local_mt import _bigrams

    candidate_grams = _bigrams(candidate)
    if not candidate_grams:
        return False
    original_grams = _bigrams(original)
    if not original_grams:
        return False

    def overlap(text_grams: set[str]) -> float:
        return len(text_grams & candidate_grams) / len(candidate_grams)

    against_original = overlap(original_grams)
    if source_text:
        source_grams = _bigrams(source_text)
        if source_grams:
            against_source = overlap(source_grams)
            # The candidate belongs to the original language when it matches
            # the original more than the source.
            return against_original >= against_source
    # Without a source to compare against, require a clear overlap.
    return against_original >= 0.6


def _clean(raw: str, form: str) -> str:
    """The shortened line out of whatever the model returned."""
    text = (raw or "").strip()
    if form == "rewrite":
        from local_mt import _extract_json_text
        text = _extract_json_text(text)
    else:
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        # Some models prefix the answer with a label.
        text = re.sub(r"^(?:hier ist|answer|antwort|translated|rewritten)\s*[:\-]\s*",
                      "", text, flags=re.IGNORECASE)
    text = text.split("\n")[0].strip()
    return text.strip().strip('"').strip()
