"""Pure input checks and job construction for the bounded native-voice trial."""
import math

from batch_dubbing.natural_timing import group_speech_cues
from modal_control.dub_segments import merge_speech_cues

MODEL = "Supertone/supertonic-3"
REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"
PACKAGE = "supertonic==1.3.1"
MAX_WORKLOAD_CALLS = 31  # one additional call belongs to controller smoke


def build_jobs(data):
    """Validate two frozen excerpts and return sequential synthesis requests.

    Source time is at most 60 seconds, total workload at most 31 calls. Each
    arm receives exactly the same cue words and voice assignments. Grouping
    changes only; no translation, audio processing, or model is invoked here.
    """
    if data.get("schema") != 1 or data.get("model") != MODEL or data.get("revision") != REVISION:
        raise ValueError("Unexpected experiment schema/model/revision")
    excerpts = data.get("excerpts", [])
    if len(excerpts) != 2 or {e.get("id") for e in excerpts} != {"bill_ackman", "mind_control"}:
        raise ValueError("Exactly the two reviewed excerpts are required")
    jobs, source_seconds = [], 0.0
    for excerpt in excerpts:
        cues = excerpt["german_cues"]
        if not cues:
            raise ValueError("Empty excerpt")
        # This validates finite ordered non-overlapping cue times and speakers.
        candidate = group_speech_cues(cues, max_seconds=20, max_characters=500, meaningful_gap=.25)
        baseline = merge_speech_cues(cues)
        start, end = cues[0]["start"], cues[-1]["end"]
        if excerpt["start"] != start or excerpt["end"] != end:
            raise ValueError("Excerpt must retain whole first and last cue boundaries")
        if not excerpt.get("english_context"):
            raise ValueError("English source context is required")
        source_seconds += end - start
        voices = excerpt["voices"]
        expected_text = " ".join(c["text"].strip() for c in cues)
        for arm, phrases in (("baseline", baseline), ("candidate", candidate)):
            if " ".join(p["text"] for p in phrases) != expected_text:
                raise ValueError("Grouping changed source text")
            for i, phrase in enumerate(phrases):
                voice = voices.get(phrase["speaker"])
                if voice not in {"M1", "M2"}:
                    raise ValueError("Only reviewed M1/M2 voice assignments are allowed")
                jobs.append({"id": f"{excerpt['id']}_{arm}_{i:02d}",
                             "excerpt_id": excerpt["id"], "arm": arm, "voice": voice,
                             "phrase": phrase, "next_phrase": phrases[i + 1] if i + 1 < len(phrases) else None,
                             "excerpt_start": start, "excerpt_end": end})
    if not math.isfinite(source_seconds) or source_seconds > 60 or not 1 <= len(jobs) <= MAX_WORKLOAD_CALLS:
        raise ValueError("Experiment exceeds reviewed source-time or synthesis-call cap")
    if "workload_calls" in data and data["workload_calls"] != len(jobs):
        raise ValueError("Frozen workload count no longer matches grouping")
    return jobs
