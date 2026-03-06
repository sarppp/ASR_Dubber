import argparse
import glob
import os
import re
import shutil
from pathlib import Path
import pysrt
 
# --- Folder Paths ---
ROOT = Path(__file__).resolve().parent.parent
NEMO_DIR = Path(os.getenv("INPUT_DIR",  str(ROOT / "nemo")))
END_PRODUCT_DIR = Path(os.getenv("OUTPUT_DIR", str(NEMO_DIR / "end_product")))
 
def get_shortened_filename(file_name: str, max_len: int = 60) -> str:
    """Shortens excessively long filenames to avoid Windows MAX_PATH limit."""
    if len(file_name) <= max_len:
        return file_name
        
    if ".nemo." in file_name:
        return "output" + file_name[file_name.index(".nemo."):]
    elif "_nemo" in file_name:
        return "output" + file_name[file_name.index("_nemo"):]
    else:
        _, ext = os.path.splitext(file_name)
        return "output" + ext

def clean_srt_files():
    print(f" 1. Cleaning subtitles in '{NEMO_DIR}' ...")
    all_srt_files = glob.glob(str(NEMO_DIR / '*.srt'))
 
    if not all_srt_files:
        print("   No .srt files found to clean.")
        return
 
    for file_path in all_srt_files:
        if file_path.endswith('_clean.srt'):
            continue
 
        try:
            subs = pysrt.open(file_path)
        except Exception as e:
            print(f"   Could not open {os.path.basename(file_path)}: {e}")
            continue
 
        change_count = 0
        for sub in subs:
            new_text = re.sub(r'\[Speaker\s+\d+\]\s*', '', sub.text)
            if new_text != sub.text:
                sub.text = new_text.strip()
                change_count += 1
 
        if change_count > 0:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_clean{ext}"
            subs.save(output_path, encoding='utf-8')
            print(f"   Cleaned {change_count} tags -> Created: {os.path.basename(output_path)}")
 
 
def move_final_products(run_label: str | None = None, dub_workdir: str | None = None) -> Path:
    destination_dir = END_PRODUCT_DIR if not run_label else END_PRODUCT_DIR / run_label
    print(f"\n2. Moving all SRTs, MP4s and intermediate files into '{destination_dir}' ...")
    os.makedirs(destination_dir, exist_ok=True)
 
    all_srts = glob.glob(str(NEMO_DIR / '*.srt'))
 
    # Find NeMo intermediate files for this run
    intermediate_files = []
    if run_label:
        # Extract base name from run_label to find related files.
        # Strip the trim suffix (_t40, _t200 …) — it's only in the SRT/run-label,
        # not in the intermediate JSON/WAV filenames which use the raw video/WAV stem.
        base_pattern = re.split(r"[._]nemo|__", run_label)[0]
        base_pattern = re.sub(r"_t\d+$", "", base_pattern)
        base_norm = re.sub(r"[^a-z0-9]", "", base_pattern.lower())
        for f in NEMO_DIR.glob("*.json"):
            f_norm = re.sub(r"[^a-z0-9]", "", f.stem.lower())
            if base_norm in f_norm:
                intermediate_files.append(str(f))
        for f in NEMO_DIR.glob("*_16k_*.wav"):
            f_norm = re.sub(r"[^a-z0-9]", "", f.stem.lower())
            if base_norm in f_norm:
                intermediate_files.append(str(f))
 
    # Find final_dub.mp4 — use explicit per-video workdir if given, else fall back to old shared path
    dubbed_videos = []
    if dub_workdir:
        dub_out = Path(dub_workdir) / "output"
        dubbed_videos = glob.glob(str(dub_out / '*.mp4'))
        if not dubbed_videos:
            print(f"   ⚠️  No dubbed MP4 found in {dub_out}")
    else:
        # Legacy fallback: old shared output path
        old_out = ROOT / 'qwen3-tts' / 'output' / 'dub' / 'output'
        dubbed_videos = glob.glob(str(old_out / '*.mp4'))
 
    files_to_move = all_srts + dubbed_videos + intermediate_files
 
    if not files_to_move:
        print("   No files found to move.")
        return destination_dir
 
    for file_path in files_to_move:
        file_name = os.path.basename(file_path)
        short_name = get_shortened_filename(file_name)
        dest_path = destination_dir / short_name
        try:
            shutil.move(file_path, dest_path)
            print(f"   Moved: {file_name} -> {short_name}")
        except Exception as e:
            print(f"   Failed to move {file_name}: {e}")
 
    return destination_dir
 
 
def copy_source_video(run_label: str | None = None) -> None:
    """Move the source video/WAV into the end_product run folder (only called on success)."""
    destination_dir = END_PRODUCT_DIR if not run_label else END_PRODUCT_DIR / run_label
 
    if not run_label:
        return
 
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wav"}
 
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
 
    label_base = _norm(re.split(r"[._]nemo|__", run_label)[0])
    label_base = re.sub(r"_t\d+$", "", label_base)  # strip trim suffix (_t40, _t200…)

    # WAV inputs have "_nemo_16k_full" baked into the stem — strip it before comparing
    def match_file(f):
        if f.suffix.lower() not in VIDEO_EXT:
            return False
        f_base = _norm(f.stem.split("_nemo")[0])
        return f_base == label_base

    # Check if already in destination
    if destination_dir.exists():
        for f in destination_dir.iterdir():
            if match_file(f):
                print(f"   Source file already in destination: {f.name}")
                return

    for f in NEMO_DIR.iterdir():
        if match_file(f):
            short_name = get_shortened_filename(f.name)
            dest = destination_dir / short_name
            try:
                shutil.move(str(f), str(dest))
                print(f"   Moved source file: {f.name} -> {short_name}")
            except Exception as e:
                print(f"   Failed to move source file {f.name}: {e}")
            return

    print(f"   ⚠️  Source video/WAV not found in {NEMO_DIR} (or destination) for run '{run_label}'")
 
 
def cleanup_wav_chunks() -> None:
    """Delete leftover _chunk_XXXX.wav files left behind by cancelled/interrupted runs."""
    chunks = list(NEMO_DIR.glob("_chunk_*.wav"))
    if not chunks:
        return
    print(f"\n3. Cleaning up {len(chunks)} leftover chunk WAV(s) in '{NEMO_DIR}' ...")
    for f in chunks:
        f.unlink()
        print(f"   Deleted: {f.name}")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Clean subtitles and gather outputs")
    parser.add_argument("--run-label",   default=None,
                        help="Subfolder name inside nemo/end_product for this run")
    parser.add_argument("--dub-workdir", default=None,
                        help="Per-video dub workdir (qwen3-tts/output/dub/<video_base>)")
    args = parser.parse_args()
 
    clean_srt_files()
    destination = move_final_products(args.run_label, args.dub_workdir)
    copy_source_video(args.run_label)
    cleanup_wav_chunks()
    print(f"\nWorkspace is clean! All files are neatly packed in: {destination}")
 
 
if __name__ == "__main__":
    main()