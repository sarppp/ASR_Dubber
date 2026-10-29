"""Prepare meaning-preserving wording feedback from measured speech activity.

No translation provider, inference, media processing or timeline editing occurs.
Timing feedback is chronological evidence, not a semantic word alignment.
"""
import re

from batch_dubbing.isochrony import ActivityTimeline, compare_timelines

MAX_TEXT_CHARACTERS = 10000
MAX_PROTECTED_TERMS = 128


def _text(value, name, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters")


def _language(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3}", value):
        raise ValueError(f"{name} must be an explicit language code")


def _direction(error, slack):
    if error > slack:
        return "shorter_equivalent_wording"
    if error < -slack:
        return "longer_equivalent_wording"
    return "keep_wording_repair_pause_placement"


def build_revision_request(source_text, translated_text, source_language, target_language,
                           target, observed, protected_terms=()):
    """Return a JSON-compatible request or an explicit no-revision/evidence result.

    Larger budgets permit fuller equivalent wording, never invented details or
    filler. Timing alone cannot prove words are missing, infer text/block mapping,
    or establish which faithful wording will fit when spoken naturally.
    """
    _text(source_text, "source_text", MAX_TEXT_CHARACTERS)
    _text(translated_text, "translated_text", MAX_TEXT_CHARACTERS)
    _language(source_language, "source_language")
    _language(target_language, "target_language")
    if not isinstance(target, ActivityTimeline) or not isinstance(observed, ActivityTimeline):
        raise ValueError("Typed target and observed ActivityTimeline required")
    if not target.speech:
        raise ValueError("Source timeline must contain speech")
    if not isinstance(protected_terms, tuple) or len(protected_terms) > MAX_PROTECTED_TERMS:
        raise ValueError("protected_terms must be a tuple of at most 128 terms")
    for term in protected_terms:
        _text(term, "protected term", 256)
    if len(set(protected_terms)) != len(protected_terms):
        raise ValueError("Duplicate protected terms")

    comparison = compare_timelines(target, observed)
    synthetic = target.provenance == observed.provenance == "synthetic"
    sufficient = comparison["evidence_sufficient_for_speech_comparison"]
    needs_evidence = not sufficient and not synthetic
    matches = comparison["timing_matches_policy"]
    status = ("needs_activity_evidence" if needs_evidence else
              "no_revision_required" if matches else "revision_required")
    slack = comparison["numerical_slack_seconds"]
    blocks = None
    if comparison["speech_errors"] is not None:
        blocks = []
        for row in comparison["speech_errors"]:
            target_seconds = row["source_end"] - row["source_start"]
            observed_seconds = row["candidate_end"] - row["candidate_start"]
            exact = all(abs(row[key]) <= slack for key in
                        ("onset_error_seconds", "offset_error_seconds", "duration_error_seconds"))
            blocks.append({"index": row["index"], "correspondence": "chronological_only",
                           "target_start_seconds": row["source_start"],
                           "target_end_seconds": row["source_end"],
                           "observed_start_seconds": row["candidate_start"],
                           "observed_end_seconds": row["candidate_end"],
                           "target_speech_seconds": target_seconds,
                           "observed_speech_seconds": observed_seconds,
                           "suggestion": "no_change" if exact else _direction(row["duration_error_seconds"], slack)})
    if status != "revision_required":
        action = "collect_activity_evidence" if needs_evidence else "none"
    elif blocks is None:
        action = "repair_speech_and_pause_structure"
    else:
        actions = {row["suggestion"] for row in blocks}
        if {"shorter_equivalent_wording", "longer_equivalent_wording"} <= actions:
            action = "rebalance_phrase_lengths"
        else:
            action = _direction(comparison["metrics"]["speech_duration_error_seconds"], slack)
    instructions = []
    if status == "revision_required":
        instructions = [
            "Produce at most one faithful alternative translation using the full source and current translation as context.",
            "Preserve all meaning, facts, names, numbers, units, negation, modality, relationships and communicative intent. Keep protected terms unchanged; report a conflict if that prevents a faithful translation.",
            "Do not add facts, filler, repetition or information absent from the source; do not drop information to meet a timing budget.",
            "Where measured speech is too long, prefer a shorter equivalent expression. Where it is too short, consider a fuller equivalent expression only if no information is invented. Longer speech does not justify filler.",
            "Respect the complete target speech/pause pattern. Equal total duration is insufficient when speech or pause boundaries differ.",
            "Block indices are chronological timing comparisons, not verified semantic or word alignments. Do not assign text to blocks, drop content or reorder meaning solely from those indices. With unequal block counts no pairing is supplied.",
            "When speech budgets already match, prioritize a linguistically valid pause-placement repair; do not rewrite solely to change character count. Wording may remain unchanged when the repair belongs in synthesis.",
            "Never change target timestamps, borrow pause time, trim speech, time-stretch audio, or change playback_rate/speaking_rate_scale from 1.0.",
            "Return cannot_fit_faithfully when these constraints cannot be reconciled, or synthesis_repair_required when wording cannot solve the measured discrepancy. Do not claim successful timing or naturalness from text alone.",
            "Every wording candidate requires new native synthesis, independent speech/pause measurement, semantic review and listening review before acceptance.",
        ]
    return {"policy": "strict-isochrony-wording-revision-v1", "status": status,
            "revision_requested": status == "revision_required", "action": action,
            "evidence_scope": "acoustic_measurements" if sufficient else "synthetic_fixture" if synthetic else "insufficient",
            "source_text": source_text, "translated_text": translated_text,
            "source_language": source_language, "target_language": target_language,
            "protected_terms": list(protected_terms),
            "target": target.to_dict(), "observed": observed.to_dict(),
            "comparison": comparison, "speech_budgets": blocks,
            "target_speech_seconds": comparison["metrics"]["source_speech_seconds"],
            "observed_speech_seconds": comparison["metrics"]["candidate_speech_seconds"],
            "constraints": {"playback_rate": 1.0, "speaking_rate_scale": 1.0,
                            "target_timestamps_immutable": True, "waveform_time_stretching_allowed": False,
                            "speech_trimming_allowed": False},
            "instructions": instructions,
            "response_contract": {"allowed_statuses": ["candidate", "cannot_fit_faithfully", "synthesis_repair_required"],
                                  "candidate_fields": ["translated_text", "explanation"],
                                  "maximum_candidates": 1},
            "quality_limit": "Timing feedback does not establish semantic fidelity, intelligibility or naturalness; no provider was called and no candidate was accepted."}
