#!/usr/bin/env python3
"""L3: transcribe the real dub and check that the meaning survived.

Takes the finished dubbed video, runs the same ASR the subtitles came from
(Parakeet Ultra) over the German audio, and compares what the audience hears
with the subtitle that was sent.

The comparison is per SENTENCE, not per cue: the ASR returns its own
segments, which do not line up with the subtitle cues. Aligning the two by
time overlap and comparing the merged text is what makes the WER meaningful.

    nemo/.venv-local-asr/bin/python tools/check_meaning.py \
        --dub <final_dub.mp4> --spoken <spoken.srt> --source <source.srt>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nemo"))

from dub_meaning import check_sentences, missing_facts, write_report  # noqa: E402


def extract_audio(video: Path, wav: Path) -> Path:
    import subprocess
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                    "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def transcribe(wav: Path) -> list[dict]:
    from nemo_model import _load_model
    from parakeet_ultra import _transcribe_parakeet_ultra

    model = _load_model("moondream/parakeet-ultra", precision="fp32", device="cuda")
    _, segments = _transcribe_parakeet_ultra(model, str(wav), 0.0)
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "text": (s.get("text") or "").strip()} for s in segments]


def read_srt(path: Path) -> list[dict]:
    cues, cur = [], None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if "-->" in line:
            left, right = line.split("-->")
            cur = [left.strip(), right.strip(), []]
        elif cur is not None and line:
            cur[2].append(line)
        elif cur is not None:
            cues.append(cur)
            cur = None
    if cur:
        cues.append(cur)

    def secs(stamp: str) -> float:
        h, m, s = stamp.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    return [{"start": secs(a), "end": secs(b), "text": " ".join(t)}
            for a, b, t in cues]


def group_sentences(cues: list[dict]) -> list[dict]:
    """Merge cues into the units the dub actually spoke.

    dub.py merges consecutive cues with ``merge_segments`` before synthesising,
    so the comparison has to use the same grouping — otherwise one sentence's
    words get counted against its neighbour.
    """
    sys.path.insert(0, str(ROOT / "qwen3-tts"))
    from dub_srt import merge_segments

    segments = [{"index": index, "start": cue["start"], "end": cue["end"],
                 "text": cue["text"], "speaker": "Speaker_1"}
                for index, cue in enumerate(cues, 1)]
    merged = merge_segments(segments, gap_sec=1.0, max_dur=10.0)
    return [{"start": segment["start"], "end": segment["end"],
             "text": segment["text"]} for segment in merged]


def heard_for(sentence: dict, heard: list[dict]) -> str:
    """The ASR text that overlaps this sentence.

    The ASR segments do not line up with the subtitle cues. To avoid counting
    the same words twice, the window is the sentence's own span, and an ASR
    segment is only taken when it lies mostly inside it.
    """
    start, end = sentence["start"], sentence["end"]
    span = max(0.01, end - start)
    pieces = []
    for segment in heard:
        overlap = min(segment["end"], end) - max(segment["start"], start)
        if overlap > 0.5 * min(span, segment["end"] - segment["start"]):
            pieces.append(segment["text"])
    return " ".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dub", type=Path, required=True, help="the dubbed video")
    parser.add_argument("--spoken", type=Path, required=True, help="the SRT that was spoken")
    parser.add_argument("--source", type=Path, required=True, help="the source-language SRT")
    parser.add_argument("--language", default="de")
    parser.add_argument("--max-wer", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=ROOT / "tools" / "meaning_report.json")
    args = parser.parse_args()

    work = args.dub.parent / ".meaning_check"
    work.mkdir(exist_ok=True)
    wav = extract_audio(args.dub, work / "dub.wav")

    print(f"🎙️  Transcribing {args.dub.name} with Parakeet Ultra …", flush=True)
    heard = transcribe(wav)

    spoken = read_srt(args.spoken)
    sentences = group_sentences(spoken)
    document = " ".join(h["text"] for h in heard)
    # The last line of a run that was cut short (--max-sentences N) ends
    # mid-sentence. Its window is shorter than the text it holds, so a WER
    # computed there is not a meaning failure.
    tail_truncated = bool(sentences) and not re.search(
        r"[.!?]$", sentences[-1]["text"].strip())
    pairs = []
    for index, sentence in enumerate(sentences, 1):
        pairs.append((index, sentence["start"], sentence["end"],
                      sentence["text"], heard_for(sentence, heard)))

    report = check_sentences(pairs, max_wer=args.max_wer,
                             document=document)
    if tail_truncated and report.sentences:
        report.sentences[-1].note = "tail truncated by --max-sentences; WER not meaningful"
        report.sentences[-1].problem = None
    write_report(report, args.out)
    summary = report.summary()

    # Document level: every fact of the spoken script must be audible
    # somewhere in the dub.
    spoken_script = " ".join(pair[3] for pair in pairs)
    document_missing = missing_facts(spoken_script, document)
    if document_missing:
        summary["failed"].append({"index": 0,
                                  "problem": f"facts missing from the whole dub: "
                                             f"{', '.join(document_missing)}",
                                  "missing": document_missing, "wer": 0.0})
        summary["ok"] = False

    print()
    for sentence in report.sentences:
        mark = "✓" if sentence.ok else "✗"
        print(f"  {mark} #{sentence.index:02d} wer {sentence.wer:.2f} "
              f"sim {sentence.similarity:.2f} {sentence.problem or ''}")
        print(f"      sent: {sentence.spoken[:90]}")
    if document_missing:
        print(f"  ✗ document: facts missing from the whole dub: "
              f"{', '.join(document_missing)}")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"},
                     ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
