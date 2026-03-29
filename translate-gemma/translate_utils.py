"""
translate_utils.py — Pure translation helpers (no side effects on import).
Imported by translate_diarize.py and by the test suite.
"""
import copy
import os
import re
import time

import pysrt
from ollama import Client

# Matches a sentence terminator followed by whitespace and a new sentence start
# (capital letter or common conjunctions). Avoids abbreviations by requiring
# the token before the period to be longer than 2 chars.
_MID_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÜÄÖA-Z\u00C0-\u024F])')

LANG_MAP = {
    'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish',
    'it': 'Italian', 'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish',
    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese',
}

MODEL_NAME   = os.getenv("TRANSLATE_MODEL", "translategemma:12b")
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", 15))

# ── Regex ─────────────────────────────────────────────────────────────────────
# Handles ALL common model output formats:
#   [180] text     <180> text     180. text     180: text
#   (180) text     180) text      180 text      [<180>] text
#
# FIX for BUG-1: added '(' to opening delimiter → (180) text now matches
# FIX for BUG-2: added ')' to closing delimiter → 180) text no longer
#                captures the ')' as part of the translation
_LINE_RE = re.compile(r'^\[?<?\(?(\d+)[>\]\)]*[\s\.\-\:]*(.*)' )


def translate_chunk(chunk_subs, src_code: str, tgt_code: str,
                    client: Client) -> dict[int, str]:
    """
    Translate one chunk of subtitles via Ollama.
    Returns {index: translated_text}. Missing indices mean the model
    skipped or mis-formatted that line — handled by _translate_with_retry.
    """
    text_to_translate = ""
    speaker_map: dict[int, str] = {}
    src_name = LANG_MAP.get(src_code, src_code)
    tgt_name = LANG_MAP.get(tgt_code, tgt_code)
    start_idx = chunk_subs[0].index
    end_idx   = chunk_subs[-1].index

    for sub in chunk_subs:
        match = re.match(r'(\[Speaker\s+\d+\])\s*(.*)', sub.text, re.DOTALL)
        if match:
            tag, content = match.group(1), match.group(2)
            speaker_map[sub.index] = tag
        else:
            content = sub.text
            speaker_map[sub.index] = ""
        clean_text = content.replace('\n', ' | ')
        text_to_translate += f"[{sub.index}] {clean_text}\n"

    prompt = f"""You are a professional translator from {src_name} ({src_code}) to {tgt_name} ({tgt_code}).

    RULES:
    1. Translate the text accurately, but STRICTLY line-by-line.
    2. Keep the [index] format at the start of every single line.
    3. IMPORTANT: These are subtitles. They contain incomplete sentences and fragments. Translate the fragment exactly as it is cut. DO NOT merge lines together to form complete sentences!
    4. You MUST return exactly {len(chunk_subs)} lines.
    5. You must start at [{start_idx}] and you must NOT stop until you have translated [{end_idx}].
    6. Do NOT translate speaker tags.

    EXAMPLE INPUT:
    [9998] Ob das jetzt sinnvoll ist,
    [9999] mit habe und aber auch einfach

    EXAMPLE OUTPUT:
    [9998] Si cela a du sens maintenant,
    [9999] avec et mais aussi simplement

    TASK:
    Translate the following {len(chunk_subs)} lines from {src_name} to {tgt_name}:

    {text_to_translate}"""

    print(f"Sending {len(chunk_subs)} lines to {MODEL_NAME} ({src_code} -> {tgt_code})...")

    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.1, "num_ctx": 2048},
        )
        raw_output = response['response'].replace('<|endoftext|>', '').strip()
        results: dict[int, str] = {}
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if m:
                results[int(m.group(1))] = m.group(2).strip()

        final_results: dict[int, str] = {}
        for idx, txt_part in results.items():
            translated_content = txt_part.replace(" | ", "\n").replace("|", "\n")
            tag = speaker_map.get(idx, "")
            final_results[idx] = f"{tag} {translated_content}".strip() if tag else translated_content

        if not final_results:
            print(f"\n❌ ERROR: Completely failed to parse anything. Raw output was:\n{raw_output}\n")
        return final_results

    except Exception as e:
        print(f"\n💥 OLLAMA ERROR: {e}")
        return {}


def _group_into_sentences(chunk_subs) -> list:
    """
    Group consecutive subs into sentence groups by terminal punctuation.

    A group closes when the last non-whitespace character of a sub's content
    (after stripping speaker tags and line-break markers) is '.', '?' or '!'.
    Any trailing subs that don't end a sentence form a final open group.
    """
    groups: list = []
    current: list = []
    for sub in chunk_subs:
        current.append(sub)
        text = sub.text or ""
        m = re.match(r'(\[Speaker\s+\d+\])\s*(.*)', text, re.DOTALL)
        content = m.group(2) if m else text
        content = content.replace('\n', ' ').replace('|', ' ').strip()
        if content and content[-1] in '.?!':
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def split_mid_sentence_subs(subs: list) -> list:
    """
    Detect subtitle lines with a mid-line sentence terminator and split them
    into two entries with proportionally divided timestamps.

    E.g.:
      "viel weiß. Und auf der anderen Seite fällt mir es"
      →  entry A: "viel weiß."              (start … split_ms)
      →  entry B: "Und auf der anderen..."  (split_ms … end)

    Avoids splitting on abbreviations (word before period ≤ 2 chars or a digit).
    New entries are appended to the end of the list with indices > max existing.
    Returns a flat list sorted by start time.
    """
    next_idx = max(sub.index for sub in subs) + 1
    expanded = []

    for sub in subs:
        text = sub.text or ""
        m_tag = re.match(r'(\[Speaker\s+\d+\])\s*(.*)', text, re.DOTALL)
        tag     = m_tag.group(1) if m_tag else ""
        content = (m_tag.group(2) if m_tag else text).replace('\n', ' ').strip()

        m = _MID_SENT_RE.search(content)
        # Guard: both parts must be non-trivial and prev word must not be abbreviation
        if m and m.start() > 3 and m.end() < len(content) - 3:
            before_dot = content[:m.start()].rstrip('.!?').strip()
            last_word  = before_dot.split()[-1] if before_dot.split() else ""
            if len(last_word) <= 2 or last_word.isdigit():
                # Looks like abbreviation (Dr. / Nr. / 3.) — skip
                expanded.append(sub)
                continue

            part_a = content[:m.start() + 1].strip()   # includes the terminator
            part_b = content[m.end():].strip()

            ratio  = len(part_a) / len(content)
            dur_ms = sub.end.ordinal - sub.start.ordinal
            mid_ms = sub.start.ordinal + int(dur_ms * ratio)

            sub_a = copy.copy(sub)
            sub_b = copy.copy(sub)

            sub_a.text  = f"{tag} {part_a}".strip() if tag else part_a
            sub_a.end   = pysrt.SubRipTime.from_ordinal(mid_ms)

            sub_b.index = next_idx
            sub_b.text  = f"{tag} {part_b}".strip() if tag else part_b
            sub_b.start = pysrt.SubRipTime.from_ordinal(mid_ms)
            next_idx += 1

            print(f"   [split] line {sub.index}: "
                  f"'{part_a[:35]}' | '{part_b[:35]}'")
            expanded.append(sub_a)
            expanded.append(sub_b)
        else:
            expanded.append(sub)

    # Keep chronological order
    expanded.sort(key=lambda s: s.start.ordinal)
    return expanded


def translate_chunk_sentences(chunk_subs, src_code: str, tgt_code: str,
                              client: Client) -> dict[int, str]:
    """
    Translate subtitles with sentence-awareness and duration constraints.

    Groups consecutive fragments into complete sentences, computes total
    wall-clock duration, and asks the LLM to produce a translation that
    fits naturally in that time.  Returns {index: translated_text}.
    """
    src_name = LANG_MAP.get(src_code, src_code)
    tgt_name = LANG_MAP.get(tgt_code, tgt_code)
    results: dict[int, str] = {}

    for group in _group_into_sentences(chunk_subs):
        total_dur = (group[-1].end.ordinal - group[0].start.ordinal) / 1000.0
        n = len(group)

        speaker_map: dict[int, str] = {}
        text_block = ""
        for sub in group:
            m = re.match(r'(\[Speaker\s+\d+\])\s*(.*)', sub.text or "", re.DOTALL)
            if m:
                tag, content = m.group(1), m.group(2)
                speaker_map[sub.index] = tag
            else:
                content = sub.text or ""
                speaker_map[sub.index] = ""
            text_block += f"[{sub.index}] {content.replace(chr(10), ' | ')}\n"

        if n == 1:
            prompt = (
                f"You are a professional dubbing translator from {src_name} to {tgt_name}.\n\n"
                f"Translate this subtitle for voice dubbing. "
                f"The speaker has {total_dur:.1f} seconds — keep it concise if needed.\n"
                f"Return exactly 1 line with the [index] prefix. "
                f"Do NOT add extra lines.\n\n"
                f"{text_block}"
            )
        else:
            prompt = (
                f"You are a professional dubbing translator from {src_name} to {tgt_name}.\n\n"
                f"Translate this complete sentence for voice dubbing.\n"
                f"The speaker has EXACTLY {total_dur:.1f} seconds to say the full translation.\n"
                f"The sentence spans {n} subtitle fragments — return exactly {n} lines.\n\n"
                f"RULES:\n"
                f"1. Keep [index] at the start of every line.\n"
                f"2. Do NOT merge or split lines — return exactly {n} lines.\n"
                f"3. Your translation must be speakable in {total_dur:.1f}s at a natural pace "
                f"(~3 words/second in {tgt_name}). Use concise wording if the time is tight.\n"
                f"4. Distribute the translation naturally across all {n} fragments.\n"
                f"5. Do NOT translate speaker tags like [Speaker 1].\n\n"
                f"{text_block}"
            )

        print(f"Sending {n} line(s) [{group[0].index}–{group[-1].index}] "
              f"({total_dur:.1f}s) to {MODEL_NAME} ({src_code}->{tgt_code})...", flush=True)

        try:
            response = client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            raw = response['response'].replace('<|endoftext|>', '').strip()
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m2 = _LINE_RE.match(line)
                if m2:
                    idx = int(m2.group(1))
                    txt = m2.group(2).strip().replace(" | ", "\n").replace("|", "\n")
                    txt = re.sub(r'^\[\d+\]\s*', '', txt)  # strip echoed index prefix
                    tag = speaker_map.get(idx, "")
                    results[idx] = f"{tag} {txt}".strip() if tag else txt

            if not results and not any(r for r in results.values()):
                print(f"\n❌ ERROR: No output for group [{group[0].index}–{group[-1].index}]."
                      f"\nRaw output:\n{raw}\n")

        except Exception as e:
            print(f"\n💥 OLLAMA ERROR: {e}")

    return results


def _translate_single_group(group: list, src_code: str, tgt_code: str,
                              client: Client) -> dict[int, str]:
    """Translate one pre-formed sentence group. Returns {index: translated_text}."""
    total_dur = (group[-1].end.ordinal - group[0].start.ordinal) / 1000.0
    n = len(group)
    src_name = LANG_MAP.get(src_code, src_code)
    tgt_name = LANG_MAP.get(tgt_code, tgt_code)

    speaker_map: dict[int, str] = {}
    text_block = ""
    for sub in group:
        m = re.match(r'(\[Speaker\s+\d+\])\s*(.*)', sub.text or "", re.DOTALL)
        if m:
            tag, content = m.group(1), m.group(2)
            speaker_map[sub.index] = tag
        else:
            content = sub.text or ""
            speaker_map[sub.index] = ""
        text_block += f"[{sub.index}] {content.replace(chr(10), ' | ')}\n"

    if n == 1:
        prompt = (
            f"You are a professional dubbing translator from {src_name} to {tgt_name}.\n\n"
            f"Translate this subtitle for voice dubbing. "
            f"The speaker has {total_dur:.1f} seconds — keep it concise if needed.\n"
            f"Return exactly 1 line with the [index] prefix. "
            f"Do NOT add extra lines.\n\n"
            f"{text_block}"
        )
    else:
        prompt = (
            f"You are a professional dubbing translator from {src_name} to {tgt_name}.\n\n"
            f"Translate this complete sentence for voice dubbing.\n"
            f"The speaker has EXACTLY {total_dur:.1f} seconds to say the full translation.\n"
            f"The sentence spans {n} subtitle fragments — return exactly {n} lines.\n\n"
            f"RULES:\n"
            f"1. Keep [index] at the start of every line.\n"
            f"2. Do NOT merge or split lines — return exactly {n} lines.\n"
            f"3. Your translation must be speakable in {total_dur:.1f}s at a natural pace "
            f"(~3 words/second in {tgt_name}). Use concise wording if the time is tight.\n"
            f"4. Distribute the translation naturally across all {n} fragments.\n"
            f"5. Do NOT translate speaker tags like [Speaker 1].\n\n"
            f"{text_block}"
        )

    print(f"Sending {n} line(s) [{group[0].index}–{group[-1].index}] "
          f"({total_dur:.1f}s) to {MODEL_NAME} ({src_code}->{tgt_code})...", flush=True)

    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        raw = response['response'].replace('<|endoftext|>', '').strip()
        results: dict[int, str] = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            m2 = _LINE_RE.match(line)
            if m2:
                idx = int(m2.group(1))
                txt = m2.group(2).strip().replace(" | ", "\n").replace("|", "\n")
                txt = re.sub(r'^\[\d+\]\s*', '', txt)  # strip echoed index prefix
                tag = speaker_map.get(idx, "")
                results[idx] = f"{tag} {txt}".strip() if tag else txt
        expected_ids = {sub.index for sub in group}
        missing = expected_ids - results.keys()
        # Also flag indices present but with empty or tag-only content
        _tag_re = re.compile(r'^\[Speaker\s+\d+\]\s*$')
        empty_ids = {idx for idx in results
                     if not results[idx].strip() or _tag_re.match(results[idx].strip())}
        if missing or empty_ids:
            print(f"\n[DEBUG] group [{group[0].index}–{group[-1].index}] "
                  f"expected {sorted(expected_ids)} got {sorted(results.keys())} "
                  f"empty/tag-only {sorted(empty_ids)}")
            print(f"[DEBUG] input sent:\n{text_block.strip()}")
            print(f"[DEBUG] raw output:\n{raw}\n[DEBUG END]")
            if missing and len(results) == len(group):
                # Model returned the right number of lines but renumbered from 1
                # (common with high indices like [18], [19], [20] → returns [1], [2], [3]).
                # Remap by position.
                sorted_expected = [sub.index for sub in group]  # preserve original order
                sorted_got = sorted(results.keys())
                results = {sorted_expected[i]: results[sorted_got[i]] for i in range(len(group))}
                print(f"   [remapped] {sorted_got} → {sorted_expected}")
                missing = expected_ids - results.keys()
        return results
    except Exception as e:
        print(f"\n💥 OLLAMA ERROR: {e}")
        return {}


def translate_group_with_retry(group: list, src_code: str, tgt_code: str,
                                client: Client, retries: int = 3) -> dict[int, str]:
    """
    Translate a pre-formed sentence group with retry.

    Each retry re-sends the whole group (full sentence context preserved).
    Final fallback uses translate_chunk (numbered format, line-by-line) for
    any indices still missing after all sentence-aware attempts.
    """
    expected = {sub.index for sub in group}
    best: dict[int, str] = {}

    for attempt in range(1, retries + 1):
        result = _translate_single_group(group, src_code, tgt_code, client)
        best.update(result)
        missing = sorted(expected - best.keys())
        if not missing:
            return best
        if attempt < retries:
            print(f"   ⚠️  Attempt {attempt}/{retries} missing {missing} — retrying in 2s...")
            time.sleep(2)

    still_missing = sorted(expected - best.keys())
    if still_missing:
        print(f"   ↩️  Falling back to line-by-line for {still_missing}...")
        missing_subs = [s for s in group if s.index in set(still_missing)]
        fallback = _translate_with_retry(missing_subs, src_code, tgt_code, client, retries=2)
        best.update(fallback)

    return best


def _translate_sentences_with_retry(chunk_subs, src_code: str, tgt_code: str,
                                    client: Client, retries: int = 3) -> dict[int, str]:
    """
    Sentence-aware translation with retry for missing indices.
    Uses translate_chunk_sentences; falls back to translate_chunk for
    any indices that remain missing after all attempts.
    """
    expected  = {sub.index for sub in chunk_subs}
    best: dict[int, str] = {}
    remaining = list(chunk_subs)

    for attempt in range(1, retries + 1):
        result = translate_chunk_sentences(remaining, src_code, tgt_code, client)
        if result:
            best.update(result)

        missing = sorted(expected - best.keys())
        if not missing:
            return best

        if attempt < retries:
            print(f"   ⚠️  Attempt {attempt}/{retries} missing indices {missing} — retrying in 2s...")
            missing_set = set(missing)
            remaining = [s for s in chunk_subs if s.index in missing_set]
            time.sleep(2)

    # Final fallback: plain fragment-by-fragment for anything still missing
    still_missing = sorted(expected - best.keys())
    if still_missing:
        print(f"   ↩️  Falling back to fragment mode for {still_missing}...")
        missing_set = set(still_missing)
        fallback_subs = [s for s in chunk_subs if s.index in missing_set]
        fallback = _translate_with_retry(fallback_subs, src_code, tgt_code, client, retries=2)
        best.update(fallback)

    return best


def _translate_with_retry(chunk_subs, src_code: str, tgt_code: str,
                          client: Client, retries: int = 3) -> dict[int, str]:
    """
    Translate chunk_subs with up to `retries` attempts.

    FIX for BUG-3: previously only retried when the ENTIRE result was empty.
    Now detects partially missing indices and retries only those lines,
    merging results across attempts so no line is silently lost.
    """
    expected  = {sub.index for sub in chunk_subs}
    best: dict[int, str] = {}
    remaining = list(chunk_subs)

    for attempt in range(1, retries + 1):
        result = translate_chunk(remaining, src_code, tgt_code, client)
        if result:
            best.update(result)

        missing = sorted(expected - best.keys())
        if not missing:
            return best  # all indices present — done

        if attempt < retries:
            if best:
                print(f"   ⚠️  Attempt {attempt}/{retries} missing indices {missing} — retrying in 2s...")
                # Next attempt: only the still-missing lines
                missing_set = set(missing)
                remaining = [s for s in chunk_subs if s.index in missing_set]
            else:
                print(f"   ⚠️  Attempt {attempt}/{retries} returned empty — retrying in 2s...")
                # Keep full chunk for next attempt
            time.sleep(2)

    return best  # best effort after all retries
