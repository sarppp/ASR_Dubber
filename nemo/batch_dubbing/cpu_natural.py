"""Shared CPU worker implementation; called only inside a released Modal job."""
import json
import math
from copy import deepcopy
from pathlib import Path
import time
import uuid

from batch_dubbing.natural_pipeline import run_pipeline, validate_units
from batch_dubbing.natural_render import render_timeline, _child_limits

MODEL = "Supertone/supertonic-3"
REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"
ARTIFACT_ROOT = Path("/cache/natural-dubbing")


def select_units(data, seconds, language):
    """Select complete utterances, disclose range exclusions and clip corridors only."""
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or not 0 < seconds <= 7200:
        raise ValueError("Requested duration must be finite and in (0, 7200]")
    all_units = data.get("speech_units", {}).get(language, [])
    if not all_units:
        raise ValueError("Regenerate the speech stage with source-linked speech_units; old subtitle windows cannot safely drive natural dubbing")
    duration = data.get("duration_seconds")
    if not isinstance(duration, (float, int)) or isinstance(duration, bool) or not math.isfinite(duration) or duration < seconds:
        raise ValueError("Requested range exceeds the source duration")
    units, excluded = [], []
    for original in all_units:
        unit = deepcopy(original)
        end, start = unit.get("source_end"), unit.get("source_start")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)):
            raise ValueError("Invalid source unit bounds")
        if end > seconds:
            if start < seconds:
                excluded.append({"id": unit["id"], "source_start": start, "source_end": end,
                                 "reason": "utterance_crosses_requested_end"})
            continue
        unit["corridor_end"] = min(unit["corridor_end"], seconds)
        unit["target_language"] = language
        unit["source_language"] = data.get("source_language", unit.get("source_language"))
        units.append(unit)
    if not units:
        raise ValueError("Requested range contains no complete source utterance; review a range ending at an utterance boundary")
    return validate_units(units, seconds), {"requested_seconds": seconds,
            "included_source_end": units[-1]["source_end"], "excluded_boundary_units": excluded,
            "complete_utterances_only": True}


def synthesize_natural(data, seconds, voices, *, language="de", output_format="m4a", emit=None):
    """Consume source-linked translated units; never refit subtitle display cues.

    Existing old artifacts must be regenerated with source linkage. No source
    alignment is guessed from cross-language caption timestamps. Every raw take
    and report is persisted in the caller's mounted Modal cache volume.
    """
    units, scope = select_units(data, seconds, language)
    for unit in units:
        if not voices.get(unit["speaker"], {}).get("voice"):
            raise ValueError("Source speech unit has no resolved speaker voice")
    if output_format not in {"m4a", "wav"}:
        raise ValueError("Unsupported timeline format")
    import numpy as np
    import subprocess
    from huggingface_hub import snapshot_download
    from supertonic import TTS

    model_dir = snapshot_download(MODEL, revision=REVISION)
    model = TTS(model="supertonic-3", model_dir=model_dir, auto_download=False,
                intra_op_num_threads=2, inter_op_num_threads=1)
    if model.sample_rate != 44100:
        raise ValueError("Unexpected Supertonic sample rate")
    started = time.monotonic()
    destination = ARTIFACT_ROOT / uuid.uuid4().hex

    def generate(unit, text):
        voice = voices[unit["speaker"]]["voice"]
        text = text.translate(str.maketrans({'„': '"', '“': '"', '”': '"', '–': '-', '—': '-', '…': '...'}))
        audio, predicted = model.synthesize(text, voice_style=model.get_voice_style(voice), lang=language)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            raise ValueError("Expected mono speech")
        if emit:
            emit("segment_complete", id=unit["id"], native_seconds=len(audio) / model.sample_rate)
        return {"samples": audio, "sample_rate": model.sample_rate, "voice": voice,
                "predicted_duration": float(np.asarray(predicted).reshape(-1)[0])}

    report, stage = {}, "synthesis"

    def persist():
        report.update(model=MODEL, revision=REVISION, sample_rate=44100, seconds=seconds,
                      native_artifact_directory=str(destination), scope=scope,
                      worker_wall_seconds=time.monotonic() - started)
        temporary = destination / "worker-report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(destination / "report.json")

    try:
        report = run_pipeline(units, duration_seconds=seconds, output_dir=destination,
                              synthesize=generate, max_synthesis_calls=len(units))
        report["synthesis_wall_seconds"] = time.monotonic() - started
        if report["status"] != "ready":
            persist()
            return {"audio": None, "report": report}
        stage = "render"
        timeline = destination / "timeline.wav"
        report["rendered"] = render_timeline(report["render"], timeline, max_wall_seconds=600)
        media = timeline
        if output_format == "m4a":
            stage = "encode"
            pending = destination / "pending.m4a"
            media = destination / "timeline.m4a"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "1",
                            "-i", str(timeline), "-threads", "1", "-c:a", "aac", "-b:a", "128k", str(pending)],
                           check=True, timeout=90, preexec_fn=_child_limits)
            pending.replace(media)
        report.update(status="complete", segments=[{"index": i, "id": row["id"], "rate": row["tempo"]}
                                                   for i, row in enumerate(report["render"]["rows"])])
        audio_bytes = media.read_bytes()
        persist()
        return {"audio": audio_bytes, "report": report}
    except Exception as exc:
        saved_report = destination / "report.json"
        if not report and saved_report.exists():
            report = json.loads(saved_report.read_text())
        destination.mkdir(parents=True, exist_ok=True)
        report.update(status="failed", failure_stage=stage, error=str(exc))
        (destination / "pending.m4a").unlink(missing_ok=True)
        persist()
        return {"audio": None, "report": report}
