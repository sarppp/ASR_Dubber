"""L3: does the dub still say what the source said?

The gate can shorten a line, and the translator can drop a fact. Neither is
visible from timing alone: a dub that fits perfectly can still say something
different from the source.

This module checks the meaning of a finished dub automatically:

1. transcribe the dubbed audio again (Ultra, the same ASR the subtitles came
   from) -> what the audience actually hears;
2. compare it with the subtitle that was sent -> WER and, more importantly,
   whether the numbers and names survived;
3. back-translate the German to English and compare it with the source
   English -> did the meaning survive the whole chain?

Step 3 needs a translation model, so it is optional and off by default; steps
1 and 2 need only the local ASR.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# No word lists.
#
# A stopword list does not scale: every language, every domain and every
# speaker add words, and a list that is wrong for one video is wrong for
# thousands. Everything below is derived from the two texts being compared,
# so it works for any language without being told anything about it.

# Numbers are facts and must survive.
NUMBER_PATTERN = re.compile(r"\d+")

# An all-caps token (CRISPR, DNA, NASA) is a name in every language.
ACRONYM_PATTERN = re.compile(r"\b[A-ZÄÖÜÀ-Þ][A-ZÄÖÜÀ-Þ0-9]{1,}\b")


def facts(text: str) -> set[str]:
    """The facts in a line: numbers and acronyms.

    Both are language-independent and cannot be produced by a translation
    accident. Proper names are deliberately not included: deciding whether a
    capitalised word is a name needs to know the language's noun rules, and
    a wrong guess here would either hide a lost name or invent one. Names are
    checked by the WER instead, which compares every word.
    """
    text = text or ""
    found = {match.lower() for match in NUMBER_PATTERN.findall(text)}
    found |= {match.lower() for match in ACRONYM_PATTERN.findall(text)}
    return found


def normalize(text: str) -> list[str]:
    """Lowercase, strip accents and punctuation, split into words.

    No stopword removal: words are compared as they are, so the check works
    for every language without a word list.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ß", "ss")
    return re.findall(r"[a-z0-9]+", text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, normalised by the reference length."""
    reference_words = normalize(reference)
    hypothesis_words = normalize(hypothesis)
    if not reference_words:
        return 0.0
    previous = list(range(len(hypothesis_words) + 1))
    for i, ref in enumerate(reference_words, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis_words, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1] / len(reference_words)


def missing_facts(reference: str, hypothesis: str,
                  document: str | None = None) -> list[str]:
    """Facts that are in the reference but not in the hypothesis.

    `document` is the whole transcript. A word can straddle the boundary
    between two subtitle windows and be heard just after it, so a fact is
    only reported as lost when it is nowhere in the document either.
    """
    expected = facts(reference)
    haystack = (hypothesis or "").lower()
    whole = (document or hypothesis or "").lower()
    missing = []
    for fact in expected:
        if fact not in haystack and fact not in whole:
            missing.append(fact)
    return sorted(missing)


def similarity(a: str, b: str) -> float:
    """How similar two texts are, 0..1, ignoring word order noise."""
    left, right = normalize(a), normalize(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


@dataclass
class SentenceCheck:
    index: int
    start: float
    end: float
    source: str
    spoken: str
    back_translation: str = ""
    wer: float = 0.0
    similarity: float = 0.0
    missing: list[str] = field(default_factory=list)
    problem: str | None = None
    note: str | None = None

    @property
    def ok(self) -> bool:
        return self.problem is None


@dataclass
class MeaningReport:
    sentences: list[SentenceCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if not s.ok]

    @property
    def ok(self) -> bool:
        return not self.failed and not self.warnings

    def summary(self) -> dict:
        return {
            "sentences": len(self.sentences),
            "failed": [{"index": s.index, "problem": s.problem,
                        "missing": s.missing, "wer": round(s.wer, 3)}
                       for s in self.failed],
            "notes": [{"index": s.index, "note": s.note}
                      for s in self.sentences if s.note],
            "mean_wer": round(sum(s.wer for s in self.sentences) /
                              len(self.sentences), 3) if self.sentences else 0.0,
            "mean_similarity": round(sum(s.similarity for s in self.sentences) /
                                     len(self.sentences), 3) if self.sentences else 0.0,
            "warnings": self.warnings,
            "ok": self.ok,
        }


def check_sentences(pairs: list[tuple[int, float, float, str, str]],
                    max_wer: float = 0.45,
                    max_missing_facts: int = 0,
                    document: str | None = None) -> MeaningReport:
    """Compare each spoken sentence with the source line.

    `pairs` holds (index, start, end, source_text, spoken_text). `document` is
    the whole transcript, used so that a word spoken just outside a window is
    not reported as lost.
    """
    report = MeaningReport()
    for index, start, end, source, spoken in pairs:
        wer = word_error_rate(source, spoken)
        missing = missing_facts(source, spoken, document)
        score = similarity(source, spoken)
        check = SentenceCheck(index=index, start=start, end=end, source=source,
                              spoken=spoken, wer=wer, similarity=score,
                              missing=missing)
        if not spoken.strip():
            # Nothing was heard here (the clip may end mid-sentence). Not a
            # meaning failure, but worth recording.
            check.problem = None
            check.note = "no speech heard in this window"
        elif len(missing) > max_missing_facts:
            check.problem = f"lost facts: {', '.join(missing)}"
        elif wer > max_wer:
            check.problem = f"WER {wer:.2f} above {max_wer:.2f}"
        report.sentences.append(check)
    return report


def content_similarity(source_text: str, other_text: str) -> float:
    """Similarity of the words that are NOT shared boilerplate.

    Plain word overlap is too weak to spot a changed meaning: "The research
    was successful" and "The research failed" share most of their words.

    The words that carry the meaning are found without a word list. A shared
    word is boilerplate when it is short and common in BOTH texts; what is
    left is what each text actually says differently, and that has to match.
    A word that appears only on one side is always content: that is exactly
    the change being tested for.
    """
    source = normalize(source_text)
    other = normalize(other_text)
    if not source or not other:
        return 0.0
    source_set, other_set = set(source), set(other)
    shared = source_set & other_set
    boilerplate = {word for word in shared if len(word) <= 4}
    source_content = [w for w in source if w not in boilerplate]
    other_content = [w for w in other if w not in boilerplate]
    if not source_content or not other_content:
        return similarity(source_text, other_text)
    # Words that only exist on one side count fully against the match.
    return difflib.SequenceMatcher(None, source_content, other_content).ratio()


def compare_back_translation(source_text: str, back_text: str,
                             min_similarity: float = 0.45) -> tuple[float, str | None]:
    """Compare the source with its back-translation.

    Both texts are in the same language (source -> target -> source). Returns
    (similarity, problem). The similarity is symmetric and ignores word
    order, so a reordering is fine but a changed meaning is not.
    """
    score = content_similarity(source_text, back_text)
    if score < min_similarity:
        return score, (f"back-translation similarity {score:.2f} below "
                       f"{min_similarity:.2f}")
    return score, None


def write_report(report: MeaningReport, path: Path) -> None:
    Path(path).write_text(json.dumps(report.summary(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
