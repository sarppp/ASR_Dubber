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
import os
import pysrt
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from ollama import Client

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
        default=os.getenv("INPUT_SRT", "momo.srt"),
        help="Path to the source SRT file",
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
        default=int(os.getenv("CHUNK_SIZE", 5)),
        help="Number of subtitle lines per request (default: 5)",
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
            "-e",
            "OLLAMA_HOST=0.0.0.0",
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
    print(f"🚀 Starting Ollama natively via '{resolved}'...")
    proc = subprocess.Popen(
        [resolved, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

    prompt = f"""You are a professional translator from {src_name} ({src_lang}) to {tgt_name} ({tgt_lang}).

RULES:
1. Translate the text meaning accurately.
2. Keep the [index] at the start of every line.
3. Do NOT merge lines. Return exactly {len(chunk_subs)} lines.
4. Do NOT translate speaker tags or indices.

EXAMPLE INPUT:
[1] Hello world
[2] How are you?

EXAMPLE OUTPUT:
[1] Bonjour le monde
[2] Comment allez-vous ?

TASK:
Translate the following lines from {src_name} to {tgt_name}:


{text_to_translate}"""

    print(f"Sending {len(chunk_subs)} lines to {model_name} ({src_lang} -> {tgt_lang})...")

    try:
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "num_ctx": 4096,
            },
        )

        raw_output = response["response"].replace("<|endoftext|>", "").strip()
        translated_lines = raw_output.split("\n")

        results = {}
        for line in translated_lines:
            line = line.strip()
            if not line:
                continue
            match_line = re.match(r"\[(\d+)\]\s*(.*)", line)
            if match_line:
                idx = int(match_line.group(1))
                txt_part = match_line.group(2)

                translated_content = txt_part.replace(" | ", "\n").replace("|", "\n")
                original_tag = speaker_map.get(idx, "")
                if original_tag:
                    final_text = f"{original_tag} {translated_content}"
                else:
                    final_text = translated_content
                results[idx] = final_text
        return results

    except Exception as e:
        print(f"Error in model generation: {e}")
        return {}


def main():
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

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
                )
                _wait_for_ollama(args.ollama_host)
            else:
                print(f"✓ Ollama already running at {args.ollama_host}")

        client = Client(host=args.ollama_host)

        subs = pysrt.open(input_path)
        chunk_size = args.chunk_size
        total_chunks = (len(subs) + chunk_size - 1) // chunk_size

        print(f"Starting translation: {len(subs)} subtitles total.", flush=True)
        print(f"Input : {input_path}")
        print(f"Output: {output_path}")
        print(f"Langs : {args.source_lang} -> {args.target_lang}")

        start_total_time = time.time()

        missing_lines: list[int] = []

        for i in range(0, len(subs), chunk_size):
            chunk = subs[i : i + chunk_size]
            chunk_num = (i // chunk_size) + 1

            print(f"\n--- Chunk {chunk_num}/{total_chunks} ---", flush=True)

            translations = translate_chunk(
                chunk,
                client,
                args.model,
                args.source_lang,
                args.target_lang,
            )

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