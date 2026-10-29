"""Serial comparison harness for preplanned native-duration synthesis candidates.

The supplied callback owns remote inference, raw audio retention and independent
activity measurement. This module imports no provider, starts no cloud job on
its own and never renders/publishes audio. An external controller must enforce
the released hardware, per-call timeouts and global execution limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from batch_dubbing.isochrony import ActivityTimeline, ACOUSTIC_PROVENANCE, compare_timelines
from batch_dubbing.isochrony_backend import (
    BackendCapabilities, CandidateSpec, MAX_CANDIDATES, plan_candidates,
)
from batch_dubbing.isochrony_revision import build_revision_request

# A comparison handles one contextual utterance, not an entire movie at once.
# Bound diagnostic expansion as well as provider calls; reports duplicate some
# evidence deliberately so each failed attempt remains independently reviewable.
MAX_COMPARISON_INTERVALS = 256


@dataclass(frozen=True)
class CandidateObservation:
    candidate_id: str
    activity: ActivityTimeline
    audio_ref: str
    audio_sha256: str
    playback_rate: float = 1.0
    waveform_time_stretched: bool = False
    speech_trimmed: bool = False

    def __post_init__(self):
        for name in ("candidate_id", "audio_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError(f"Nonempty bounded {name} required")
        if not isinstance(self.audio_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", self.audio_sha256):
            raise ValueError("Raw audio SHA256 required")
        if not isinstance(self.activity, ActivityTimeline):
            raise ValueError("Measured activity required")
        if len(self.activity.speech) > MAX_COMPARISON_INTERVALS:
            raise ValueError("Observed activity exceeds the per-comparison interval budget")
        if self.activity.evidence_id != "sha256:" + self.audio_sha256:
            raise ValueError("Activity evidence must identify the observed audio digest")
        # Validate processing declarations; policy violations are recorded as
        # rejected observations rather than discarded evidence.
        validated = compare_timelines(self.activity, self.activity, playback_rate=self.playback_rate,
                                      waveform_time_stretched=self.waveform_time_stretched,
                                      speech_trimmed=self.speech_trimmed)
        object.__setattr__(self, "playback_rate", validated["processing"]["playback_rate"])

    def to_dict(self):
        return asdict(self)


def run_comparison(source, capabilities, candidates, *, source_text, source_language,
                   synthesize, output_dir, max_calls, protected_terms=()):
    """Execute a fixed candidate list serially and retain every returned report.

    At most eight calls, no retries, no automatic text edits or fallback. All
    specs are validated before the first callback. On a provider failure, save
    partial evidence and stop. Earlier output is never overwritten. The callback
    must retain raw audio even for rejected attempts; references and digests are
    recorded here, not downloaded or independently verified by this harness.

    This is a comparison, not a publication path. Timing matches leave meaning,
    naturalness and artifact/process auditing pending. Synthetic tests have a
    separate result status and never become verified real-speech candidates.
    """
    if not isinstance(source, ActivityTimeline) or not source.speech:
        raise ValueError("A source activity timeline containing speech is required")
    if len(source.speech) > MAX_COMPARISON_INTERVALS:
        raise ValueError("Source activity exceeds the per-comparison interval budget")
    if source.provenance not in ACOUSTIC_PROVENANCE | {"synthetic"}:
        raise ValueError("Acoustic source evidence required; display/word timestamps are insufficient")
    if not isinstance(capabilities, BackendCapabilities) or not callable(synthesize):
        raise ValueError("Backend capabilities and explicit synthesis callback required")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= MAX_CANDIDATES:
        raise ValueError("Declare a call cap in [1, 8]")
    if not isinstance(candidates, tuple) or not 1 <= len(candidates) <= max_calls:
        raise ValueError("Candidate tuple must fit the declared global call cap")
    seen, identity = set(), None
    for spec in candidates:
        if not isinstance(spec, CandidateSpec):
            raise ValueError("A validated CandidateSpec is required")
        expected = plan_candidates(capabilities, spec.request, seeds=(spec.seed,), max_candidates=1)[0]
        if spec != expected or spec.candidate_id in seen:
            raise ValueError("Candidate identity changed or repeated")
        seen.add(spec.candidate_id)
        request = spec.request
        # Validate all translation context before any paid/provider callback.
        build_revision_request(source_text, request.text, source_language, request.language,
                               source, source, protected_terms)
        if request.target_duration_seconds != source.duration_seconds:
            raise ValueError("Candidate changes the original target duration")
        if request.target_activity is not None and request.target_activity != source:
            raise ValueError("Candidate changes the original speech/pause target")
        current_identity = (request.source_id, request.language)
        if identity is not None and identity != current_identity:
            raise ValueError("A comparison must keep source identity and target language fixed")
        identity = current_identity
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "policy": "strict-native-isochrony-v1", "status": "running",
        "source": source.to_dict(), "backend": capabilities.to_dict(),
        "source_text": source_text, "source_language": source_language,
        "protected_terms": list(protected_terms),
        "max_calls": max_calls, "calls": 0, "max_concurrency": 1, "retries": 0,
        "candidates": [spec.to_dict() for spec in candidates], "attempts": [],
        "timing_candidate_ids": [], "synthetic_match_ids": [],
        "publication": {"status": "blocked", "content_review": "pending",
                        "listening_review": "pending", "artifact_and_processing_audit": "pending"},
    }

    def persist():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")

    persist()
    try:
        for spec in candidates:
            attempt = {"candidate_id": spec.candidate_id, "status": "started"}
            report["attempts"].append(attempt)
            report["calls"] += 1
            persist()  # Count attempted calls, including failures.
            observation = synthesize(spec)
            if not isinstance(observation, CandidateObservation):
                raise ValueError("Provider must return a CandidateObservation")
            attempt["observation"] = observation.to_dict()
            if observation.candidate_id != spec.candidate_id:
                raise ValueError("Provider returned evidence for another candidate")
            result = compare_timelines(source, observation.activity,
                                       playback_rate=observation.playback_rate,
                                       waveform_time_stretched=observation.waveform_time_stretched,
                                       speech_trimmed=observation.speech_trimmed)
            attempt.update(status="measured", comparison=result)
            if (observation.playback_rate != 1.0 or observation.waveform_time_stretched
                    or observation.speech_trimmed):
                attempt["wording_feedback"] = {
                    "status": "synthesis_processing_repair_required", "revision_requested": False,
                    "comparison": result,
                    "instruction": "Restore native untrimmed speech before deriving wording feedback; do not use altered timings as native-duration evidence.",
                }
            else:
                attempt["wording_feedback"] = build_revision_request(
                    source_text, spec.request.text, source_language, spec.request.language,
                    source, observation.activity, protected_terms)
            if result["status"] == "matches_measurements":
                report["timing_candidate_ids"].append(spec.candidate_id)
            elif result["status"] == "synthetic_fixture":
                report["synthetic_match_ids"].append(spec.candidate_id)
            persist()
        report["status"] = ("timing_candidates_available" if report["timing_candidate_ids"]
                            else "synthetic_comparison_complete" if report["synthetic_match_ids"]
                            else "needs_revision")
        persist()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        if report["attempts"]:
            report["attempts"][-1]["status"] = "failed"
        persist()
        raise
    # Return the same JSON shape downstream consumers read from the audit file.
    return json.loads((output / "report.json").read_text())
