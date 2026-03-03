import concurrent.futures
import glob
import os
import pysrt
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from ollama import Client

# ── Setup ─────────────────────────────────────────────────────────────────────
_ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME   = os.getenv("TRANSLATE_MODEL", "translategemma:4b")

LANG_MAP = {
    'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish',
    'it': 'Italian', 'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish',
    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese'
}

CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", 15))
OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", "1"))
OLLAMA_BIN          = os.getenv("OLLAMA_BIN", "ollama")
OLLAMA_DOCKER_IMAGE = os.getenv("OLLAMA_DOCKER_IMAGE", "ollama/ollama")
OLLAMA_CONTAINER    = os.getenv("OLLAMA_CONTAINER_NAME", "ollama")
OLLAMA_MODELS_DIR   = os.getenv("OLLAMA_MODELS_DIR",
                                 str(os.path.join(os.path.expanduser("~"),
                                                  "python-tools", ".ollama_models")))


# ── Ollama lifecycle ──────────────────────────────────────────────────────────

def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{_ollama_host}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False

def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False

def _resolve_ollama_bin(bin_path: str) -> str:
    if os.path.exists(bin_path):
        return bin_path
    resolved = shutil.which(bin_path)
    if resolved:
        return resolved
    for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama",
                      os.path.expanduser("~/.local/bin/ollama")]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find Ollama executable '{bin_path}' on PATH")

def _start_ollama() -> tuple[str, subprocess.Popen | None]:
    if _docker_available():
        print("🐳 Starting Ollama via Docker...", flush=True)
        subprocess.run(["docker", "rm", "-f", OLLAMA_CONTAINER], capture_output=True)
        cmd = ["docker", "run", "-d", "--rm", "--name", OLLAMA_CONTAINER,
               "-e", "OLLAMA_HOST=0.0.0.0",
               "-e", f"OLLAMA_NUM_PARALLEL={OLLAMA_NUM_PARALLEL}",
               "-p", "11434:11434"]
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            cmd += ["--gpus", "all"]
            print("   GPU detected — enabling NVIDIA runtime", flush=True)
        except Exception:
            pass
        if os.path.exists(OLLAMA_MODELS_DIR):
            cmd += ["-v", f"{OLLAMA_MODELS_DIR}:/root/.ollama"]
        else:
            print("   ℹ️ Models dir missing — Ollama will create a fresh cache", flush=True)
        cmd.append(OLLAMA_DOCKER_IMAGE)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "docker", proc

    resolved = _resolve_ollama_bin(OLLAMA_BIN)
    print(f"🚀 Starting Ollama natively ({resolved}, OLLAMA_NUM_PARALLEL={OLLAMA_NUM_PARALLEL})...",
          flush=True)
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(OLLAMA_NUM_PARALLEL)}
    proc = subprocess.Popen(
        [resolved, "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return "native", proc

def _wait_for_ollama(timeout: int = 40) -> None:
    for i in range(timeout):
        if _ollama_running():
            print(f"✓ Ollama ready at {_ollama_host} (took {i+1}s)")
            return
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"   waiting for Ollama... ({i+1}/{timeout})")
    raise RuntimeError("Ollama did not become ready in time")

def _stop_ollama(start_method: str, proc: subprocess.Popen | None) -> None:
    print("🛑 Stopping Ollama...", flush=True)
    if start_method == "docker":
        try:
            subprocess.run(["docker", "stop", OLLAMA_CONTAINER],
                           capture_output=True, timeout=15)
            print("✓ Ollama container stopped")
        except Exception as exc:
            print(f"⚠️ Could not stop Docker container: {exc}")
        return
    if proc is not None:
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=10)
            print("✓ Ollama stopped")
        except Exception as exc:
            print(f"⚠️ Could not stop Ollama cleanly: {exc}")


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def _get_vram_mib() -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        return sum(int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip())
    except Exception:
        return None

def _get_vram_total_mib() -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        return sum(int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip())
    except Exception:
        return None

def _poll_vram(stop_event: threading.Event, samples: list) -> None:
    while not stop_event.is_set():
        v = _get_vram_mib()
        if v is not None:
            samples.append(v)
        time.sleep(1)

def _compute_optimal_workers(
    vram_baseline: int, vram_idle: int, vram_peak: int | None,
    vram_total: int, ollama_num_parallel: int = 1, buffer_mib: int = 512,
    total_chunks: int = 1,
) -> tuple[int, dict]:
    model_mib      = vram_idle - vram_baseline
    available_mib  = vram_total - vram_idle - buffer_mib
    kv_delta       = (vram_peak - vram_idle) if vram_peak is not None else 0
    measurement_valid = kv_delta > 32
    base_workers   = max(2, ollama_num_parallel + 1)
    optimal        = min(4, total_chunks, base_workers)
    print(f"[DEBUG] base_workers={base_workers}, total_chunks={total_chunks}, optimal={optimal}")
    print(f"[DEBUG] ollama_num_parallel={ollama_num_parallel}")
    stats = {
        "model_mib": model_mib, "available_mib": available_mib,
        "buffer_mib": buffer_mib, "kv_delta_mib": kv_delta,
        "measurement_valid": measurement_valid, "ollama_num_parallel": ollama_num_parallel,
    }
    return optimal, stats

def _print_vram_summary(
    vram_baseline: int, vram_idle: int, vram_peak: int | None,
    vram_total: int, num_workers: int, stats: dict, total_chunks: int = 1,
) -> None:
    print(f"[VRAM] Baseline: {vram_baseline} MiB / {vram_total} MiB total")
    print(f"[VRAM] After warmup: model loaded = {vram_idle} MiB (+{stats['model_mib']} MiB)")
    if stats["measurement_valid"]:
        print(f"[VRAM] Peak during warmup: {vram_peak} MiB "
              f"(+{stats['kv_delta_mib']} MiB KV delta — measurement valid)")
    else:
        print(f"[VRAM] Peak during warmup == idle ({vram_idle} MiB) — "
              "chunk finished in <1 s, KV allocation unmeasurable. Ignoring.")
    n = stats["ollama_num_parallel"]
    print(f"[VRAM] Ollama parallelism: OLLAMA_NUM_PARALLEL={n} "
          f"({'serial — 1 forward pass at a time' if n == 1 else f'{n} concurrent forward passes'})")
    print(f"[VRAM] Worker logic: capped at 4 workers to eliminate idle gaps without overhead "
          f"(selected {num_workers} for {total_chunks} chunks)")
    print(f"[VRAM] ✓ Auto-selected {num_workers} workers")


# ── Translation ───────────────────────────────────────────────────────────────

def translate_chunk(chunk_subs, src_code, tgt_code, client: Client):
    text_to_translate = ""
    speaker_map       = {}
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
        results = {}
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.search(r'^\[?<?(\d+)>?\]?[\s\.\-\:]*(.*)', line)
            if m:
                results[int(m.group(1))] = m.group(2).strip()

        final_results = {}
        for idx, txt_part in results.items():
            translated_content = txt_part.replace(" | ", "\n").replace("|", "\n")
            tag = speaker_map.get(idx, "")
            final_results[idx] = f"{tag} {translated_content}" if tag else translated_content

        if not final_results:
            print(f"\n❌ ERROR: Completely failed to parse anything. Raw output was:\n{raw_output}\n")
        return final_results

    except Exception as e:
        print(f"\n💥 OLLAMA ERROR: {e}")
        return {}


def _translate_with_retry(chunk_subs, src_code, tgt_code, client: Client,
                           retries: int = 3) -> dict:
    """Retry translate_chunk up to `retries` times on empty response."""
    for attempt in range(1, retries + 1):
        result = translate_chunk(chunk_subs, src_code, tgt_code, client)
        if result:
            return result
        if attempt < retries:
            print(f"   ⚠️  Attempt {attempt}/{retries} returned empty — retrying in 2s...")
            time.sleep(2)
    return {}


# ── SRT discovery ─────────────────────────────────────────────────────────────

TARGET_LANG_CODE         = os.getenv("TARGET_LANG_CODE", "fr")
SOURCE_LANG_CODE_OVERRIDE = os.getenv("SOURCE_LANG_CODE", "")

folder_path = os.getenv("NEMO_DIR", os.path.join(os.path.dirname(__file__), '..', 'nemo'))
folder_path = os.path.realpath(folder_path)

srt_files = [f for f in glob.glob(os.path.join(folder_path, '*.srt'))
             if not re.search(r'_[a-z]{2}\.srt$', f)]

if not srt_files:
    print(f"❌ No valid un-translated .srt files found in '{folder_path}/'.")
    sys.exit(1)

input_file   = srt_files[0]
filename_only = os.path.basename(input_file)

if SOURCE_LANG_CODE_OVERRIDE:
    SOURCE_LANG_CODE = SOURCE_LANG_CODE_OVERRIDE
else:
    lang_match = re.search(r'\.([a-z]{2})\.', filename_only)
    SOURCE_LANG_CODE = lang_match.group(1) if lang_match else 'en'

base_name, ext = os.path.splitext(input_file)
output_file = f"{base_name}_{TARGET_LANG_CODE}{ext}"

print(f"🔍 Input:  {filename_only}")
print(f"🌐 Langs:  {SOURCE_LANG_CODE} -> {TARGET_LANG_CODE}")
print(f"💾 Output: {output_file}")

try:
    subs = pysrt.open(input_file)
except Exception as e:
    print(f"Could not open file: {e}")
    sys.exit(1)

chunks       = [subs[i:i + CHUNK_SIZE] for i in range(0, len(subs), CHUNK_SIZE)]
total_chunks = len(chunks)

# ── Ollama auto-start ─────────────────────────────────────────────────────────
start_method: str | None = None
auto_proc: subprocess.Popen | None = None

try:
    if _ollama_running():
        print(f"✓ Ollama already running at {_ollama_host}")
    else:
        start_method, auto_proc = _start_ollama()
        _wait_for_ollama()

    client = Client(host=_ollama_host)

    print(f"Starting translation: {len(subs)} subtitles total.", flush=True)
    start_total_time = time.time()
    missing_lines: list[int] = []

    # ── Phase 1: Baseline VRAM ────────────────────────────────────────────────
    vram_baseline = _get_vram_mib()
    vram_total    = _get_vram_total_mib()

    # ── Phase 2: Warmup chunk with VRAM polling ───────────────────────────────
    poll_samples: list[int] = []
    stop_event = threading.Event()
    poll_thread: threading.Thread | None = None

    if vram_baseline is not None and vram_total is not None:
        poll_thread = threading.Thread(
            target=_poll_vram, args=(stop_event, poll_samples), daemon=True)
        poll_thread.start()

    print(f"\n--- Chunk 1/{total_chunks} (warmup) ---", flush=True)
    warmup_translations = _translate_with_retry(
        chunks[0], SOURCE_LANG_CODE, TARGET_LANG_CODE, client)

    stop_event.set()
    if poll_thread is not None:
        poll_thread.join(timeout=3)

    vram_peak = max(poll_samples) if poll_samples else None
    vram_idle = _get_vram_mib()

    if not warmup_translations:
        print("\n❌ FATAL: chunk 1 returned nothing after 3 attempts — "
              "Ollama error or model not found.")
        print("   Make sure the model is pulled: ollama pull translategemma:4b")
        sys.exit(1)

    for sub in chunks[0]:
        if sub.index in warmup_translations:
            sub.text = warmup_translations[sub.index]
        else:
            print(f"⚠️ Warning: Missing line {sub.index}", flush=True)
            missing_lines.append(sub.index)

    # ── Phase 3: Compute optimal workers ─────────────────────────────────────
    vram_ok = all(v is not None for v in [vram_baseline, vram_idle, vram_total])
    print(f"[DEBUG CALL SITE] total_chunks={total_chunks}, type={type(total_chunks)}")
    if vram_ok:
        num_workers, stats = _compute_optimal_workers(
            vram_baseline, vram_idle, vram_peak,  # type: ignore[arg-type]
            vram_total, OLLAMA_NUM_PARALLEL, 512, total_chunks,
        )
        if num_workers > 2 and OLLAMA_NUM_PARALLEL == 1:
            print(f"[VRAM] Warning: {num_workers} workers requested but OLLAMA_NUM_PARALLEL=1")
            print(f"[VRAM] Most workers will be idle. Consider setting "
                  f"OLLAMA_NUM_PARALLEL={min(num_workers-1, 4)}")
        _print_vram_summary(
            vram_baseline, vram_idle, vram_peak, vram_total,  # type: ignore[arg-type]
            num_workers, stats, total_chunks,
        )
    else:
        print("[VRAM] nvidia-smi unavailable — using 2 workers (pipeline fill only)")
        num_workers = min(2, total_chunks)

    # ── Phase 4: Remaining chunks ─────────────────────────────────────────────
    remaining = chunks[1:]

    def _apply_translations(chunk, translations: dict, chunk_num: int) -> None:
        if not translations:
            print(f"\n❌ FATAL: chunk {chunk_num} returned nothing after 3 attempts — "
                  "Ollama error or model not found.")
            print("   Make sure the model is pulled: ollama pull translategemma:4b")
            sys.exit(1)
        for sub in chunk:
            if sub.index in translations:
                sub.text = translations[sub.index]
            else:
                print(f"⚠️ Warning: Missing line {sub.index}", flush=True)
                missing_lines.append(sub.index)

    if not remaining:
        pass  # only one chunk — already processed
    elif num_workers == 1:
        for chunk_idx, chunk in enumerate(remaining, 2):
            print(f"\n--- Chunk {chunk_idx}/{total_chunks} ---", flush=True)
            translations = _translate_with_retry(
                chunk, SOURCE_LANG_CODE, TARGET_LANG_CODE, client)
            _apply_translations(chunk, translations, chunk_idx)
    else:
        print(f"\nRunning {num_workers} parallel workers for {len(remaining)} remaining chunks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {
                executor.submit(_translate_with_retry,
                                chunk, SOURCE_LANG_CODE, TARGET_LANG_CODE, client): (chunk_idx, chunk)
                for chunk_idx, chunk in enumerate(remaining, 2)
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx, chunk = future_to_chunk[future]
                translations = future.result()
                _apply_translations(chunk, translations, chunk_idx)

    subs.save(output_file, encoding='utf-8')

    end_total_time = time.time()
    total_seconds  = end_total_time - start_total_time
    failure_rate   = len(missing_lines) / len(subs) if len(subs) else 1.0

    if failure_rate > 0.3:
        print(f"\n❌ TRANSLATION FAILED: {len(missing_lines)}/{len(subs)} lines missing "
              f"({failure_rate:.0%}). Check Ollama is running and model is pulled.")
        print(f"   Missing indices: {missing_lines}")
        sys.exit(1)

    if missing_lines:
        print(f"\n⚠️  {len(missing_lines)} line(s) untranslated — saving partial result.")

    minutes = total_seconds / 60 if total_seconds else 0
    print(f"\n✅ SUCCESS! Saved to {output_file}", flush=True)
    print(f"⏱️ Total Time: {total_seconds:.2f} seconds ({minutes:.2f} min)", flush=True)
    if total_seconds > 0:
        print(f"🚀 Speed: {len(subs)/total_seconds:.2f} lines per second", flush=True)

finally:
    if start_method is not None:
        _stop_ollama(start_method, auto_proc)
