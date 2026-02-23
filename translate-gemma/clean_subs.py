import pysrt
import re
import glob
import os
import shutil

# --- Folder Paths ---
NEMO_DIR = '../nemo'
TTS_OUTPUT_DIR = '../qwen3-tts/output/dub/output'
# This creates the end_product folder INSIDE the nemo directory
END_PRODUCT_DIR = os.path.join(NEMO_DIR, 'end_product') 

def clean_srt_files():
    print(f"🧹 1. Cleaning subtitles in '{NEMO_DIR}' ...")
    all_srt_files = glob.glob(os.path.join(NEMO_DIR, '*.srt'))

    if not all_srt_files:
        print("   ℹ️ No .srt files found to clean.")
        return

    for file_path in all_srt_files:
        if file_path.endswith('_clean.srt'):
            continue

        try:
            subs = pysrt.open(file_path)
        except Exception as e:
            print(f"   ⚠️ Could not open {os.path.basename(file_path)}: {e}")
            continue

        change_count = 0
        for sub in subs:
            # Remove "[Speaker X]" and trailing spaces
            new_text = re.sub(r'\[Speaker\s+\d+\]\s*', '', sub.text)
            
            if new_text != sub.text:
                sub.text = new_text.strip()
                change_count += 1
                
        if change_count > 0:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_clean{ext}"
            subs.save(output_path, encoding='utf-8')
            print(f"   ✅ Cleaned {change_count} tags -> Created: {os.path.basename(output_path)}")


def move_final_products():
    print(f"\n📦 2. Moving all SRTs and MP4s into '{END_PRODUCT_DIR}' ...")
    
    # Create the end_product folder inside nemo
    os.makedirs(END_PRODUCT_DIR, exist_ok=True)

    # Gather ALL .srt files from nemo (originals, translations, and the new clean ones)
    # glob.glob does not search subdirectories, so it won't accidentally grab 
    # files already inside the end_product folder if you run this twice.
    all_srts = glob.glob(os.path.join(NEMO_DIR, '*.srt'))
    
    # Gather ALL .mp4 files from the TTS output folder
    dubbed_videos = glob.glob(os.path.join(TTS_OUTPUT_DIR, '*.mp4'))

    files_to_move = all_srts + dubbed_videos

    if not files_to_move:
        print("   ❌ No files found to move.")
        return

    for file_path in files_to_move:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(END_PRODUCT_DIR, file_name)
        
        try:
            # shutil.move deletes the file from the original location after moving
            shutil.move(file_path, dest_path)
            print(f"   🚚 Moved: {file_name}")
        except Exception as e:
            print(f"   ⚠️ Failed to move {file_name}: {e}")


# --- Execution ---
if __name__ == "__main__":
    #clean_srt_files()
    move_final_products()
    print(f"\n✨ Workspace is clean! All files are neatly packed in: {END_PRODUCT_DIR}")