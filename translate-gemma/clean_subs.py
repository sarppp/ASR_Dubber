import argparse
import glob
import os
import re
import shutil
from pathlib import Path
import pysrt

# --- Folder Paths ---
ROOT = Path(__file__).resolve().parent.parent
NEMO_DIR = ROOT / 'nemo'
END_PRODUCT_DIR = NEMO_DIR / 'end_product'

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
    print(f"\n2. Moving all SRTs and MP4s into '{destination_dir}' ...")
    os.makedirs(destination_dir, exist_ok=True)

    all_srts = glob.glob(str(NEMO_DIR / '*.srt'))

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

    files_to_move = all_srts + dubbed_videos

    if not files_to_move:
        print("   No files found to move.")
        return destination_dir

    for file_path in files_to_move:
        file_name = os.path.basename(file_path)
        dest_path = destination_dir / file_name
        try:
            shutil.move(file_path, dest_path)
            print(f"   Moved: {file_name}")
        except Exception as e:
            print(f"   Failed to move {file_name}: {e}")

    return destination_dir


def copy_source_video(run_label: str | None = None) -> None:
    """Copy (not move) the source video into the end_product run folder."""
    destination_dir = END_PRODUCT_DIR if not run_label else END_PRODUCT_DIR / run_label

    # Infer which video this run is for from the run_label base name
    # run_label looks like: Debate_101_with_Harvard_s_...nemo.en.diarize__en_to_fr
    # We want to match the video file in nemo/ whose normalized stem matches
    if not run_label:
        return

    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    label_base = _norm(re.split(r"[._]nemo|__", run_label)[0])

    for f in NEMO_DIR.iterdir():
        if f.suffix.lower() not in VIDEO_EXT:
            continue
        if _norm(f.stem) == label_base:
            dest = destination_dir / f.name
            if dest.exists():
                print(f"   Source video already in destination: {f.name}")
                return
            try:
                shutil.copy2(str(f), str(dest))
                print(f"   Copied source video: {f.name}")
            except Exception as e:
                print(f"   Failed to copy source video {f.name}: {e}")
            return

    print(f"   ⚠️  Source video not found in {NEMO_DIR} for run '{run_label}'")


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
    print(f"\nWorkspace is clean! All files are neatly packed in: {destination}")


if __name__ == "__main__":
    main()