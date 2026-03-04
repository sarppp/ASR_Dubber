"""
translate_utils.py — Pure translation helpers (no side effects on import).
Imported by translate_diarize.py and by the test suite.
"""
import os
import re
import time

from ollama import Client

LANG_MAP = {
    'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish',
    'it': 'Italian', 'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish',
    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese',
}

MODEL_NAME   = os.getenv("TRANSLATE_MODEL", "translategemma:4b")
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
