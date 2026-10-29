"""Prepare a native-speed excerpt from saved caption records without models.

Grouping preserves wording and caption provenance. A punctuation boundary is
not a claim of grammatical completeness, and captions are not acoustic timing.
"""
from copy import deepcopy
import math
import re

from batch_dubbing.cheap_native_probe import MODEL, REVISION, IMAGE_ID, PACKAGE


def group_caption_blocks(cues, *, start, end, gap_seconds=1.0):
    """Split only at a long gap after punctuation; reject partial edge cues."""
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for x in (start, end, gap_seconds)) or not 0 <= start < end or gap_seconds <= 0:
        raise ValueError("Invalid excerpt or grouping bounds")
    if not isinstance(cues, list) or len(cues) > 20000:
        raise ValueError("Expected bounded caption records")
    result, previous_end = [], -1.0
    for index, cue in enumerate(cues):
        if not isinstance(cue, dict):
            raise ValueError("Invalid caption")
        left, right, text = cue.get("start"), cue.get("end"), cue.get("text")
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (left, right))
                or left < 0 or right <= left or left < previous_end
                or not isinstance(text, str) or not text.strip()):
            raise ValueError("Invalid or overlapping caption")
        previous_end = right
        if right <= start or left >= end:
            continue
        if left < start or right > end:
            raise ValueError("Excerpt cuts through a caption; select complete boundaries")
        text = " ".join(text.split())
        terminal = bool(result and re.search(r'[.!?…][\"\'»›”’\)\]]*$', result[-1]["text"]))
        if result and not (left - result[-1]["srt_end"] >= gap_seconds and terminal):
            result[-1]["text"] += " " + text
            result[-1]["srt_end"] = right
            result[-1]["source_indices"].append(index)
        else:
            result.append(dict(srt_start=left, srt_end=right, text=text, source_indices=[index]))
    if not result or not re.search(r'[.!?…][\"\'»›”’\)\]]*$', result[-1]["text"]):
        raise ValueError("Excerpt must end at a punctuated caption boundary")
    if any(len(row["text"]) > 5000 for row in result):
        raise ValueError("A complete block exceeds the text bound; select a shorter excerpt")
    return result


def build_input(payload, *, start, end, voice, source_metadata):
    """Use a chosen test preset; do not infer speaker identity from captions."""
    if not isinstance(payload, dict) or not re.fullmatch(r"[FM][1-5]", voice):
        raise ValueError("A supported explicit test voice is required")
    if not isinstance(source_metadata, dict):
        raise ValueError("Source provenance is required")
    samples = []
    for language, cues in (("en", payload.get("english")),
                           ("de", payload.get("translations", {}).get("de"))):
        for number, group in enumerate(group_caption_blocks(cues, start=start, end=end)):
            samples.append(dict(id=f"{language}_{number:03d}", language=language,
                                voice=voice, speaker=f"comparison_voice_{voice}",
                                seed=20261029 + len(samples), **group))
    return dict(schema=1, model=MODEL, revision=REVISION, package=PACKAGE,
                image_id=IMAGE_ID, speed=1.0, steps=8, max_chunk_length=300,
                silence_duration=.3, excerpt_start=float(start), excerpt_end=float(end),
                source_metadata=deepcopy(source_metadata), samples=samples)
