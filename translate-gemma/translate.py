"""
--input / -i — source SRT path (default: INPUT_SRT env or momo.srt).
--output / -o — destination SRT path (default: <input>_<target>.srt).
--source-lang / --src — source language code (default: SOURCE_LANG_CODE env or en).
--target-lang / --tgt — target language code (default: TARGET_LANG_CODE env or fr).
--chunk-size — subtitles per request (default: CHUNK_SIZE env or 10).
--model — Ollama model name (default: TRANSLATE_MODEL env or translategemma:4b).
--ollama-host — base URL of the Ollama server (default: OLLAMA_HOST env or http://127.0.0.1:11434).
--ollama-bin — executable used when starting Ollama natively (default: OLLAMA_BIN env or ollama).
--ollama-docker-image — Docker image for auto-start (default: OLLAMA_DOCKER_IMAGE env or ollama/ollama).
--ollama-container-name — container name for the Docker run (default: OLLAMA_CONTAINER_NAME env or ollama).
--ollama-models-dir — host directory mounted into the container’s model cache (default: OLLAMA_MODELS_DIR env or ~/python-tools/.ollama_models).
--no-auto-ollama — skip auto-start; require Ollama to be running already.

uv run python translate.py --input "Que pouvons-nous encore manger.srt" --source-lang fr --target-lang de

"""

import argparse
import concurrent.futures
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
from pathlib import Path
from ollama import Client


def _get_vram_mib() -> int | None:
    """Return total VRAM used (MiB) across all GPUs, or None if unavailable."""
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
    """Return total VRAM capacity (MiB) across all GPUs."""
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
    """Background thread: record VRAM usage (MiB) every second."""
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
    """
    Worker count = OLLAMA_NUM_PARALLEL + 1 (one queued, so GPU never idles).

    OLLAMA_NUM_PARALLEL > 1 is only beneficial when requests are long enough to
    overlap — i.e., each forward pass takes longer than the inter-request gap.
    For sub-second chunks the slots sit idle while adding memory pressure.
    The VRAM delta between OLLAMA_NUM_PARALLEL=1 and N is the pre-allocated KV
    cost per extra slot; if the warmup chunk completed in <1 s the measurement
    is unreliable, but that itself is a signal the chunks are short.
    """
    model_mib = vram_idle - vram_baseline
    available_mib = vram_total - vram_idle - buffer_mib

    # Detect measurement failure: chunk finished before our 1-second poll caught the peak
    kv_delta = (vram_peak - vram_idle) if vram_peak is not None else 0
    measurement_valid = kv_delta > 32  # >32 MiB delta = real KV signal, not noise

    # Simple heuristic: keep Ollama's queue full without excess overhead
    # For sub-second chunks, 2-4 workers eliminate inter-request gaps efficiently
    # More workers just add Python overhead without benefit
    base_workers = max(2, ollama_num_parallel + 1)
    optimal = min(4, total_chunks, base_workers)
    print(f"[DEBUG] base_workers={base_workers}, total_chunks={total_chunks}, optimal={optimal}")
    print(f"[DEBUG] ollama_num_parallel={ollama_num_parallel}")

    stats = {
        "model_mib": model_mib,
        "available_mib": available_mib,
        "buffer_mib": buffer_mib,
        "kv_delta_mib": kv_delta,
        "measurement_valid": measurement_valid,
        "ollama_num_parallel": ollama_num_parallel,
    }
    return optimal, stats

def _print_vram_summary(
    vram_baseline: int, vram_idle: int, vram_peak: int | None,
    vram_total: int, num_workers: int, stats: dict, total_chunks: int = 1,
) -> None:
    """Print VRAM measurement and worker selection rationale."""
    print(f"[VRAM] Baseline: {vram_baseline} MiB / {vram_total} MiB total")
    print(f"[VRAM] After warmup: model loaded = {vram_idle} MiB (+{stats['model_mib']} MiB)")

    if stats["measurement_valid"]:
        print(
            f"[VRAM] Peak during warmup: {vram_peak} MiB "
            f"(+{stats['kv_delta_mib']} MiB KV delta — measurement valid)"
        )
    else:
        print(
            f"[VRAM] Peak during warmup == idle ({vram_idle} MiB) — "
            "chunk finished in <1 s, KV allocation unmeasurable. Ignoring."
        )

    n = stats["ollama_num_parallel"]
    print(
        f"[VRAM] Ollama parallelism: OLLAMA_NUM_PARALLEL={n} "
        f"({'serial — 1 forward pass at a time' if n == 1 else f'{n} concurrent forward passes'})"
    )
    print(
        f"[VRAM] Worker logic: capped at 4 workers to eliminate idle gaps without overhead "
        f"(selected {num_workers} for {total_chunks} chunks)"
    )
    print(f"[VRAM] ✓ Auto-selected {num_workers} workers")

def _find_srt_files() -> list[Path]:
    """Return SRT files found in the script's directory and the sibling nemo/ folder."""
    script_dir = Path(__file__).parent
    found: list[Path] = []
    for p in sorted(script_dir.glob("*.srt")):
        found.append(p)
    nemo_dir = script_dir.parent / "nemo"
    if nemo_dir.is_dir():
        for p in sorted(nemo_dir.glob("*.srt")):
            found.append(p)
    return found

def _pick_srt_file() -> Path:
    """Discover SRT files and let the user choose one interactively."""
    files = _find_srt_files()
    if not files:
        sys.exit("No SRT files found in script directory or nemo/. Provide --input explicitly.")
    print("Found SRT files:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f.relative_to(Path(__file__).parent.parent)}")
    while True:
        choice = input(f"Select file [1-{len(files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("Invalid choice, try again.")

def _ask_lang(prompt: str, default: str) -> str:
    """Prompt for a language code, showing available options."""
    print(f"\n{prompt}:")
    codes = list(LANG_MAP.items())
    for i in range(0, len(codes), 4):
        print("  " + "   ".join(f"{k}={v:<10}" for k, v in codes[i:i+4]))
    answer = input(f"  Enter code [{default}]: ").strip()
    return answer if answer else default

LANG_MAP = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate SRT files with TranslateGemma")
    parser.add_argument(
        "--input",
        "-i",
        default=os.getenv("INPUT_SRT"),
        help="Path to the source SRT file (default: interactive picker)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.getenv("OUTPUT_SRT"),
        help="Path for the translated SRT (default: <input>_<target>.srt)",
    )
    parser.add_argument(
        "--source-lang",
        "--src",
        default=os.getenv("SOURCE_LANG_CODE", "en"),
        help="Source language code (default: en)",
    )
    parser.add_argument(
        "--target-lang",
        "--tgt",
        default=os.getenv("TARGET_LANG_CODE", "fr"),
        help="Target language code (default: fr)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 3)),
        help="Number of subtitle lines per request (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("TRANSLATE_MODEL", "translategemma:4b"),
        help="Ollama model name (default: translategemma:4b)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--ollama-bin",
        default=os.getenv("OLLAMA_BIN", "ollama"),
        help="Executable to run when auto-starting Ollama (default: ollama)",
    )
    parser.add_argument(
        "--ollama-docker-image",
        default=os.getenv("OLLAMA_DOCKER_IMAGE", "ollama/ollama"),
        help="Docker image to use when starting Ollama via Docker",
    )
    parser.add_argument(
        "--ollama-container-name",
        default=os.getenv("OLLAMA_CONTAINER_NAME", "ollama"),
        help="Container name when running Ollama via Docker",
    )
    parser.add_argument(
        "--ollama-models-dir",
        default=os.getenv(
            "OLLAMA_MODELS_DIR", str(Path.home() / "python-tools" / ".ollama_models")
        ),
        help="Directory to mount as Ollama model cache (default: ~/python-tools/.ollama_models)",
    )
    parser.add_argument(
        "--no-auto-ollama",
        action="store_true",
        help="Disable automatic Ollama start/stop (expect server already running)",
    )
    parser.add_argument(
        "--ollama-num-parallel",
        type=int,
        default=int(os.getenv("OLLAMA_NUM_PARALLEL", "1")),
        help="Ollama parallel inference slots (OLLAMA_NUM_PARALLEL, default: 1). "
             "Sets how many requests Ollama processes simultaneously. Workers are set to this + 1. "
             "Default: 1 (serial, best for short subtitle chunks)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Parallel translation workers: 'auto' (OLLAMA_NUM_PARALLEL + 1) or integer (default: auto)",
    )
    return parser.parse_args()

def _ollama_running(host: str) -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False

def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def _resolve_ollama_bin(bin_path: str) -> str:
    if Path(bin_path).exists():
        return bin_path
    resolved = shutil.which(bin_path)
    if resolved:
        return resolved
    for candidate in [
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        str(Path.home() / ".local/bin/ollama"),
    ]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find Ollama executable '{bin_path}' on PATH (checked common locations)"
    )

def _start_ollama(
    bin_path: str,
    docker_image: str,
    container_name: str,
    models_dir: str | None,
    num_parallel: int = 1,
) -> tuple[str, subprocess.Popen | None]:
    if _docker_available():
        print("🐳 Starting Ollama via Docker...", flush=True)
        subprocess.run(
            ["docker", "rm", "-f", container_name], capture_output=True
        )
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-e", "OLLAMA_HOST=0.0.0.0",
            "-e", f"OLLAMA_NUM_PARALLEL={num_parallel}",
            "-p",
            "11434:11434",
        ]
        # Try enabling GPU support if available
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            cmd += ["--gpus", "all"]
            print("   GPU detected — enabling NVIDIA runtime", flush=True)
        except Exception:
            pass

        if models_dir and Path(models_dir).exists():
            cmd += ["-v", f"{models_dir}:/root/.ollama"]
        else:
            print("   ℹ️ Models dir missing — Ollama will create a fresh cache", flush=True)

        cmd.append(docker_image)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return "docker", proc

    resolved = _resolve_ollama_bin(bin_path)
    print(f"🚀 Starting Ollama natively via '{resolved}' (OLLAMA_NUM_PARALLEL={num_parallel})...")
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(num_parallel)}
    proc = subprocess.Popen(
        [resolved, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return "native", proc

def _wait_for_ollama(host: str, timeout: int = 40) -> None:
    for i in range(timeout):
        if _ollama_running(host):
            print(f"✓ Ollama ready at {host} (took {i + 1}s)")
            return
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"   waiting for Ollama... ({i + 1}/{timeout})")
    raise RuntimeError("Ollama did not become ready in time")

def _stop_ollama(start_method: str, proc: subprocess.Popen | None, container_name: str) -> None:
    print("🛑 Stopping Ollama...")
    if start_method == "docker":
        try:
            subprocess.run(
                ["docker", "stop", container_name], capture_output=True, timeout=15
            )
            print("✓ Ollama container stopped")
        except Exception as exc:
            print(f"⚠️ Could not stop Docker container: {exc}")
        return

    if start_method == "native" and proc is not None:
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=10)
            print("✓ Ollama stopped")
        except Exception as exc:
            print(f"⚠️ Could not stop Ollama cleanly: {exc}")

def translate_chunk(chunk_subs, client: Client, model_name: str, src_lang: str, tgt_lang: str):
    text_to_translate = ""
    speaker_map: dict[int, str] = {}

    src_name = LANG_MAP.get(src_lang, src_lang)
    tgt_name = LANG_MAP.get(tgt_lang, tgt_lang)

    # Grab the first and last ID of this chunk to force the model to finish
    start_idx = chunk_subs[0].index
    end_idx = chunk_subs[-1].index

    for sub in chunk_subs:
        match = re.match(r"(\[Speaker\s+\d+\])\s*(.*)", sub.text, re.DOTALL)
        if match:
            tag = match.group(1)
            content = match.group(2)
            speaker_map[sub.index] = tag
        else:
            content = sub.text
            speaker_map[sub.index] = ""

        clean_text = content.replace("\n", " | ")
        text_to_translate += f"[{sub.index}] {clean_text}\n"

    # We shifted the prompt fully to the left so no weird spaces get sent to the AI
    prompt = f"""You are a professional translator from {src_name} ({src_lang}) to {tgt_name} ({tgt_lang}).

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

    print(f"Sending {len(chunk_subs)} lines to {model_name} ({src_lang} -> {tgt_lang})...")

    try:
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "num_ctx": 2048, 
            },
        )

        raw_output = response["response"].replace("<|endoftext|>", "").strip()
        results = {}
        
        # Bulletproof parser looking for [1], 1., or <1> at the start of lines
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line: 
                continue
                
            match_line = re.search(r'^\[?<?(\d+)>?\]?[\s\.\-\:]*(.*)', line)
            if match_line:
                idx = int(match_line.group(1))
                txt_part = match_line.group(2).strip()
                results[idx] = txt_part

        # Apply formatting and original speaker tags
        final_results = {}
        for idx, txt_part in results.items():
            translated_content = txt_part.replace(" | ", "\n").replace("|", "\n")
            original_tag = speaker_map.get(idx, "")
            
            if original_tag:
                final_results[idx] = f"{original_tag} {translated_content}"
            else:
                final_results[idx] = translated_content
                
        if not final_results:
            print(f"\n❌ ERROR: Completely failed to parse anything. Raw output was:\n{raw_output}\n")

        return final_results

    except Exception as e:
        print(f"\n💥 OLLAMA ERROR: {e}")
        return {}
def main():
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    interactive = not args.input
    if interactive:
        input_path = str(_pick_srt_file())
        args.source_lang = _ask_lang("Source language", args.source_lang)
        args.target_lang = _ask_lang("Target language", args.target_lang)
    else:
        input_path = os.path.abspath(args.input)

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{args.target_lang}{ext or '.srt'}"

    start_method: str | None = None
    auto_proc: subprocess.Popen | None = None
    try:
        if args.no_auto_ollama:
            if not _ollama_running(args.ollama_host):
                raise RuntimeError(
                    "Ollama is not running. Start it manually or omit --no-auto-ollama."
                )
        else:
            if not _ollama_running(args.ollama_host):
                start_method, auto_proc = _start_ollama(
                    args.ollama_bin,
                    args.ollama_docker_image,
                    args.ollama_container_name,
                    args.ollama_models_dir,
                    num_parallel=args.ollama_num_parallel,
                )
                _wait_for_ollama(args.ollama_host)
            else:
                print(f"✓ Ollama already running at {args.ollama_host}")

        # Parse --workers value
        if args.workers == "auto":
            workers_mode = "auto"
        else:
            try:
                workers_mode = int(args.workers)
                if workers_mode < 1:
                    raise ValueError
            except ValueError:
                raise ValueError("--workers must be 'auto' or a positive integer")

        client = Client(host=args.ollama_host)

        subs = pysrt.open(input_path)
        chunk_size = args.chunk_size
        chunks = [subs[i : i + chunk_size] for i in range(0, len(subs), chunk_size)]
        total_chunks = len(chunks)

        print(f"Starting translation: {len(subs)} subtitles total.", flush=True)
        print(f"Input : {input_path}")
        print(f"Output: {output_path}")
        print(f"Langs : {args.source_lang} -> {args.target_lang}")

        start_total_time = time.time()
        missing_lines: list[int] = []

        # --- Phase 1: Baseline VRAM measurement ---
        vram_baseline = _get_vram_mib()
        vram_total = _get_vram_total_mib()

        # --- Phase 2: Warmup — process first chunk with VRAM polling ---
        poll_samples: list[int] = []
        stop_event = threading.Event()
        poll_thread: threading.Thread | None = None

        if vram_baseline is not None and vram_total is not None:
            poll_thread = threading.Thread(
                target=_poll_vram, args=(stop_event, poll_samples), daemon=True
            )
            poll_thread.start()

        print(f"\n--- Chunk 1/{total_chunks} (warmup) ---", flush=True)
        warmup_translations = translate_chunk(
            chunks[0], client, args.model, args.source_lang, args.target_lang
        )

        stop_event.set()
        if poll_thread is not None:
            poll_thread.join(timeout=3)

        vram_peak = max(poll_samples) if poll_samples else None
        vram_idle = _get_vram_mib()

        if not warmup_translations:
            print("\n❌ FATAL: chunk 1 returned nothing — Ollama error or model not found.")
            raise SystemExit(1)

        for sub in chunks[0]:
            if sub.index in warmup_translations:
                sub.text = warmup_translations[sub.index]
            else:
                print(f"⚠️ Warning: Missing line {sub.index}", flush=True)
                missing_lines.append(sub.index)

        # --- Phase 3: Compute optimal workers ---
        if workers_mode == "auto":
            # Use the value we configured Ollama with (or whatever was pre-running).
            # This is the ground truth: if OLLAMA_NUM_PARALLEL=1 (default), the GPU
            # is serial and VRAM math is irrelevant for choosing worker count.
            ollama_num_parallel = args.ollama_num_parallel

            vram_ok = all(v is not None for v in [vram_baseline, vram_idle, vram_total])
            print(f"[DEBUG CALL SITE] total_chunks={total_chunks}, type={type(total_chunks)}")
            if vram_ok:
                num_workers, stats = _compute_optimal_workers(
                    vram_baseline, vram_idle, vram_peak,  # type: ignore[arg-type]
                    vram_total, ollama_num_parallel, 512,  # buffer_mib=512 default
                    total_chunks,
                )
                # The _compute_optimal_workers function already handles capping
                
                # If we want more than 2 workers, we need higher OLLAMA_NUM_PARALLEL
                # Otherwise workers will just queue up serially
                if num_workers > 2 and ollama_num_parallel == 1:
                    print(f"[VRAM] Warning: {num_workers} workers requested but OLLAMA_NUM_PARALLEL=1")
                    print(f"[VRAM] Most workers will be idle. Consider setting OLLAMA_NUM_PARALLEL={min(num_workers-1, 4)}")
                    print(f"[VRAM] Tip: OLLAMA_NUM_PARALLEL=N ollama serve (before starting)")
                
                _print_vram_summary(
                    vram_baseline, vram_idle, vram_peak, vram_total,  # type: ignore[arg-type]
                    num_workers, stats, total_chunks,
                )
            else:
                print("[VRAM] nvidia-smi unavailable — using 2 workers (pipeline fill only)")
                num_workers = min(2, total_chunks)
        else:
            num_workers = workers_mode

        # --- Phase 4: Remaining chunks ---
        remaining = chunks[1:]

        def _apply_translations(chunk, translations: dict, chunk_num: int) -> None:
            if not translations:
                print(
                    f"\n❌ FATAL: chunk {chunk_num} returned nothing — Ollama error or model not found."
                )
                raise SystemExit(1)
            for sub in chunk:
                if sub.index in translations:
                    sub.text = translations[sub.index]
                else:
                    print(f"⚠️ Warning: Missing line {sub.index}", flush=True)
                    missing_lines.append(sub.index)

        if not remaining:
            pass  # Only one chunk total — already processed
        elif num_workers == 1:
            for chunk_idx, chunk in enumerate(remaining, 2):
                print(f"\n--- Chunk {chunk_idx}/{total_chunks} ---", flush=True)
                translations = translate_chunk(
                    chunk, client, args.model, args.source_lang, args.target_lang
                )
                _apply_translations(chunk, translations, chunk_idx)
        else:
            print(f"\nRunning {num_workers} parallel workers for {len(remaining)} remaining chunks...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_chunk = {
                    executor.submit(
                        translate_chunk, chunk, client, args.model, args.source_lang, args.target_lang
                    ): (chunk_idx, chunk)
                    for chunk_idx, chunk in enumerate(remaining, 2)
                }
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx, chunk = future_to_chunk[future]
                    translations = future.result()
                    _apply_translations(chunk, translations, chunk_idx)

        subs.save(output_path, encoding="utf-8")

        end_total_time = time.time()
        total_seconds = end_total_time - start_total_time

        failure_rate = len(missing_lines) / len(subs) if len(subs) else 1.0
        if failure_rate > 0.3:
            print(
                f"\n❌ TRANSLATION FAILED: {len(missing_lines)}/{len(subs)} lines missing "
                f"({failure_rate:.0%}). Check Ollama is running and model is pulled."
            )
            print(f"   Missing indices: {missing_lines}")
            raise SystemExit(1)

        if missing_lines:
            print(f"\n⚠️  {len(missing_lines)} line(s) untranslated — saving partial result.")

        minutes = total_seconds / 60 if total_seconds else 0
        print(f"\n✅ SUCCESS! File saved as: {output_path}", flush=True)
        print(f"⏱️ Total Time: {total_seconds:.2f} seconds ({minutes:.2f} min)", flush=True)
        if total_seconds > 0:
            print(f"🚀 Speed: {len(subs)/total_seconds:.2f} lines per second", flush=True)
    finally:
        if start_method is not None:
            _stop_ollama(start_method, auto_proc, args.ollama_container_name)


if __name__ == "__main__":
    main()