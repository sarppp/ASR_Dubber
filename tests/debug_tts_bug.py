#!/usr/bin/env python3
"""
debug_tts_bug.py — Capture everything needed to reproduce a TTS bug.

Usage:
  python debug_tts_bug.py --text "L'horloge épigénétique s'est révélée" --language French --mode clone --ref-audio /path/to/ref.wav

Saves:
  - Environment snapshot (packages, CUDA, GPU state)
  - Model version info
  - Synthesis attempt (with multiple repetition_penalty values)
  - All logs and outputs
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def capture_environment(out_dir: Path) -> Path:
    """Save complete environment snapshot."""
    env_file = out_dir / "environment.txt"
    with env_file.open("w") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        # Python and packages
        f.write("=== Python ===\n")
        subprocess.run([sys.executable, "--version"], stdout=f, stderr=f)
        f.write(f"\nPython path: {sys.executable}\n\n")
        
        f.write("=== uv pip freeze ===\n")
        subprocess.run(["uv", "pip", "freeze"], stdout=f, stderr=f, cwd=out_dir.parent)
        
        f.write("\n=== CUDA ===\n")
        try:
            subprocess.run(["nvidia-smi"], stdout=f, stderr=f)
        except FileNotFoundError:
            f.write("nvidia-smi not found\n")
        
        f.write("\n=== Environment variables ===\n")
        for k, v in sorted(os.environ.items()):
            if any(x in k.upper() for x in ["CUDA", "GPU", "TORCH", "PYTORCH"]):
                f.write(f"{k}={v}\n")
        
        f.write("\n=== Git status ===\n")
        try:
            subprocess.run(["git", "status"], stdout=f, stderr=f, cwd=out_dir.parent)
            f.write("\n=== Git diff ===\n")
            subprocess.run(["git", "diff"], stdout=f, stderr=f, cwd=out_dir.parent)
        except:
            f.write("Git not available\n")
    
    return env_file


def test_synthesis(worker_path: Path, text: str, language: str, mode: str, 
                  voice: str, ref_audio: Path | None, out_dir: Path) -> dict:
    """Test synthesis with multiple repetition_penalty values."""
    results = {}
    
    for rep_pen in [1.0, 1.05, 1.1, 1.2]:
        label = f"rep_pen_{rep_pen}"
        out_wav = out_dir / f"{label}.wav"
        log_file = out_dir / f"{label}.log"
        
        # Build request
        req = {
            "text": text,
            "language": language,
            "output": str(out_wav),
        }
        if mode == "clone" and ref_audio:
            req["ref_audio"] = str(ref_audio)
            req["ref_text"] = ""
        else:
            req["voice"] = voice
        
        # Modify repetition_penalty by patching the worker temporarily
        worker_script = out_dir / f"worker_{label}.py"
        worker_code = worker_path.read_text()
        
        # Insert repetition_penalty override
        if "repetition_penalty=" in worker_code:
            worker_code = worker_code.replace(
                "repetition_penalty=1.05",
                f"repetition_penalty={rep_pen}"
            )
        else:
            # Add after model.load if not found
            worker_code = worker_code.replace(
                "model._use_faster = _use_faster",
                f"model._use_faster = _use_faster\n    # Override repetition_penalty\n    model.repetition_penalty = {rep_pen}"
            )
        
        worker_script.write_text(worker_code)
        
        # Run synthesis
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(worker_script), "--mode", mode],
                capture_output=True, text=True, timeout=300, cwd=out_dir.parent
            )
            duration = time.time() - start
            
            with log_file.open("w") as f:
                f.write(f"Command: {' '.join(proc.args)}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Return code: {proc.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(proc.stdout)
                f.write("\nSTDERR:\n")
                f.write(proc.stderr)
            
            # Check output
            if out_wav.exists():
                import wave
                with wave.open(str(out_wav)) as wf:
                    audio_dur = wf.getnframes() / wf.getframerate()
                    file_size = out_wav.stat().st_size
            else:
                audio_dur = file_size = 0
            
            results[label] = {
                "success": proc.returncode == 0 and out_wav.exists(),
                "duration": duration,
                "audio_duration": audio_dur,
                "file_size": file_size,
                "return_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            
            log.info(f"[{label}] {'SUCCESS' if results[label]['success'] else 'FAILED'} "
                    f"audio={audio_dur:.2f}s file={file_size//1024}KB")
            
        except subprocess.TimeoutExpired:
            log.error(f"[{label}] TIMEOUT after 300s")
            results[label] = {"success": False, "error": "timeout"}
        except Exception as e:
            log.error(f"[{label}] EXCEPTION: {e}")
            results[label] = {"success": False, "error": str(e)}
        
        finally:
            # Cleanup worker script
            worker_script.unlink(missing_ok=True)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Debug TTS bug reproduction")
    parser.add_argument("--text", required=True, help="Text that caused the bug")
    parser.add_argument("--language", default="French", help="Language code")
    parser.add_argument("--mode", choices=["clone", "custom"], default="custom", help="TTS mode")
    parser.add_argument("--voice", default="serena", help="Voice for custom mode")
    parser.add_argument("--ref-audio", help="Reference audio for clone mode")
    parser.add_argument("--worker", default="qwen_tts_worker.py", help="Path to TTS worker")
    parser.add_argument("--out-dir", help="Output directory (auto-generated if omitted)")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c for c in args.text[:30] if c.isalnum() or c in " _-").strip()
        out_dir = Path(f"tts_debug_{timestamp}_{safe_text}")
    
    out_dir.mkdir(exist_ok=True)
    log.info(f"Saving debug data to: {out_dir}")
    
    # Capture environment
    env_file = capture_environment(out_dir)
    log.info(f"Environment saved to: {env_file}")
    
    # Save input parameters
    params = {
        "text": args.text,
        "language": args.language,
        "mode": args.mode,
        "voice": args.voice,
        "ref_audio": str(args.ref_audio) if args.ref_audio else None,
        "worker": str(Path(args.worker).resolve()),
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    
    # Test synthesis
    worker_path = Path(args.worker)
    if not worker_path.exists():
        log.error(f"Worker not found: {worker_path}")
        return 1
    
    ref_audio = Path(args.ref_audio) if args.ref_audio and Path(args.ref_audio).exists() else None
    
    results = test_synthesis(
        worker_path, args.text, args.language, args.mode,
        args.voice, ref_audio, out_dir
    )
    
    # Save results
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    
    # Summary
    log.info("\n=== SUMMARY ===")
    for label, r in results.items():
        status = "✓" if r.get("success", False) else "✗"
        audio = r.get("audio_duration", 0)
        log.info(f"{status} {label}: audio={audio_dur:.2f}s")
    
    log.info(f"\nAll data saved in: {out_dir}")
    log.info("To reproduce later, use the same environment and run the same commands.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())