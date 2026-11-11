"""Declarative contract for bounded synthesis at unchanged waveform playback.

This module never loads a provider, generates audio, starts a task or reads a
file. A plan authorizes no execution. Executors must enforce the serial/no-retry
policy and measure every output independently. Native duration conditioning can
change generated articulation; playback 1.0 is not proof of natural speaking pace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batch_dubbing.isochrony import ActivityTimeline

MAX_CANDIDATES = 8
MAX_DURATION_SECONDS = 7200
POLICY = "strict-native-isochrony-v1"


def _text(value, label, maximum=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a nonempty bounded string")


def _language(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3}", value):
        raise ValueError("Language must be an explicit language code")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared provider abilities, never verification of generated speech.

    False means unsupported; omitted/unknown capability values are rejected.
    ``capability_evidence`` identifies the documentation/code behind the claim,
    not a listening test or proof that a particular request will succeed.
    """
    backend_id: str
    revision: str
    languages: tuple[str, ...]
    duration_conditioning: bool
    internal_pause_conditioning: bool
    capability_evidence: str

    def __post_init__(self):
        for name in ("backend_id", "revision", "capability_evidence"):
            _text(getattr(self, name), name)
        if not isinstance(self.languages, tuple) or not self.languages or len(self.languages) > 256:
            raise ValueError("languages must be a nonempty bounded tuple")
        for language in self.languages:
            _language(language)
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("Duplicate backend language")
        if type(self.duration_conditioning) is not bool or type(self.internal_pause_conditioning) is not bool:
            raise ValueError("Capabilities must explicitly be True or False")
        if self.internal_pause_conditioning and not self.duration_conditioning:
            raise ValueError("Strict pause conditioning requires duration conditioning")

    def to_dict(self):
        return {**asdict(self), "evidence_scope": "declared_capability_only",
                "achieved_timing_verified": False, "naturalness_verified": False}


@dataclass(frozen=True)
class SynthesisRequest:
    """Immutable translated text and requested timing, relative to one source unit.

    ``target_activity=None`` requests a total-duration budget only. Supplying a
    timeline requests its complete speech/pause pattern, and therefore requires
    an explicitly pause-capable provider; it is never silently downgraded.
    """
    source_id: str
    text: str
    language: str
    target_duration_seconds: float
    target_activity: ActivityTimeline | None = None
    playback_rate: float = 1.0
    speaking_rate_scale: float = 1.0

    def __post_init__(self):
        _text(self.source_id, "source_id")
        _text(self.text, "text", 10000)
        if len(self.text) > 10000:
            raise ValueError("Text exceeds the 10000-character request bound")
        _language(self.language)
        duration = _number(self.target_duration_seconds, "target_duration_seconds")
        if not 0 < duration <= MAX_DURATION_SECONDS:
            raise ValueError("target_duration_seconds must be in (0, 7200]")
        for name in ("playback_rate", "speaking_rate_scale"):
            if _number(getattr(self, name), name) != 1.0:
                raise ValueError(f"{name} must be exactly 1.0")
        object.__setattr__(self, "target_duration_seconds", duration)
        object.__setattr__(self, "playback_rate", 1.0)
        object.__setattr__(self, "speaking_rate_scale", 1.0)
        if self.target_activity is not None:
            from batch_dubbing.isochrony import ActivityTimeline
            if not isinstance(self.target_activity, ActivityTimeline):
                raise ValueError("target_activity must be an ActivityTimeline")
            if self.target_activity.duration_seconds != duration:
                raise ValueError("Activity timeline duration must equal target_duration_seconds")
            _text(self.target_activity.evidence_id, "target_activity.evidence_id")
            if not self.target_activity.speech:
                raise ValueError("A synthesis request must contain source speech")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate in a fixed serial plan; never a claim that audio exists."""
    candidate_id: str
    request: SynthesisRequest
    backend_id: str
    revision: str
    seed: int
    max_concurrency: int = field(default=1, init=False)
    retries: int = field(default=0, init=False)
    waveform_time_stretching_allowed: bool = field(default=False, init=False)
    speech_trimming_allowed: bool = field(default=False, init=False)

    def to_dict(self):
        return {**asdict(self), "policy": POLICY, "status": "planned_unverified",
                "requires_independent_activity_measurement": True,
                "requires_content_and_listening_review": True}


def plan_candidates(capabilities: BackendCapabilities, request: SynthesisRequest, *,
                    seeds: tuple[int, ...], max_candidates: int) -> tuple[CandidateSpec, ...]:
    """Validate one immutable request and prepare at most eight seeded attempts.

    There is no automatic retry, text rewrite, speed search, provider fallback or
    execution. A caller invoking a provider must consume this tuple serially and
    account for each attempted call against its separately approved budget.
    """
    if not isinstance(capabilities, BackendCapabilities) or not isinstance(request, SynthesisRequest):
        raise ValueError("Typed backend capabilities and synthesis request are required")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("max_candidates must be an integer in [1, 8]")
    if not isinstance(seeds, tuple) or not 1 <= len(seeds) <= max_candidates:
        raise ValueError("Declare a nonempty seed tuple within the candidate budget")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds would repeat the same declared candidate")
    if request.language not in capabilities.languages:
        raise ValueError("Backend does not declare the requested language")
    if not capabilities.duration_conditioning:
        raise ValueError("Backend does not declare native duration conditioning")
    if request.target_activity is not None and not capabilities.internal_pause_conditioning:
        raise ValueError("Backend cannot condition on the requested internal speech/pause pattern")
    specs = []
    for seed in seeds:
        payload = {"policy": POLICY, "backend": asdict(capabilities),
                   "request": request.to_dict(), "seed": seed}
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False)
        candidate_id = "candidate_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        specs.append(CandidateSpec(candidate_id, request, capabilities.backend_id, capabilities.revision, seed))
    return tuple(specs)
