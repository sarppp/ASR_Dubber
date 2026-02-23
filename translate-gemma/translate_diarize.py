import pysrt
import time
import re
import os
import glob
from ollama import Client

# 1. SETUP
client = Client(host='http://172.17.0.1:11434')
MODEL_NAME = 'translategemma:4b'

# Full language names for better prompting
LANG_MAP = {
    'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 
    'it': 'Italian', 'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish', 
    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese'
}

def translate_chunk(chunk_subs, src_code, tgt_code):
    text_to_translate = ""
    speaker_map = {} 
    
    src_name = LANG_MAP.get(src_code, src_code)
    tgt_name = LANG_MAP.get(tgt_code, tgt_code)

    # Prepare input text
    for sub in chunk_subs:
        # 1. Detach Speaker Tag
        match = re.match(r'(\[Speaker\s+\d+\])\s*(.*)', sub.text, re.DOTALL)
        if match:
            tag = match.group(1)
            content = match.group(2)
            speaker_map[sub.index] = tag 
        else:
            content = sub.text
            speaker_map[sub.index] = ""

        # 2. Flatten text (replace newlines with |)
        clean_text = content.replace('\n', ' | ')
        text_to_translate += f"[{sub.index}] {clean_text}\n"

    # 3. ONE-SHOT PROMPT (The Fix!)
    # We give it a concrete example so it knows exactly what to do.
    prompt = f"""You are a professional translator from {src_name} ({src_code}) to {tgt_name} ({tgt_code}).

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
    
    print(f"Sending {len(chunk_subs)} lines to {MODEL_NAME} ({src_code} -> {tgt_code})...")
    
    try:
        response = client.generate(
            model=MODEL_NAME, 
            prompt=prompt,
            options={
                "temperature": 0.1, # Slight bump helps avoid repetition loops
                "num_ctx": 4096     # Increased context window
            }
        )
        
        # 4. CLEAN GARBAGE
        raw_output = response['response']
        # Remove the <|endoftext|> spam if it appears
        raw_output = raw_output.replace('<|endoftext|>', '').strip()
        
        translated_lines = raw_output.split('\n')
        
        # 5. PARSE & REASSEMBLE
        results = {}
        for line in translated_lines:
            line = line.strip()
            if not line: continue
            
            # Robust Regex to find "[123] Text..."
            # This handles cases where the model forgets a space like "[1]Text"
            match_line = re.match(r'\[(\d+)\]\s*(.*)', line)
            
            if match_line:
                idx = int(match_line.group(1))
                txt_part = match_line.group(2)

                # Restore newlines
                translated_content = txt_part.replace(' | ', '\n').replace('|', '\n')
                
                # Reattach Speaker Tag
                original_tag = speaker_map.get(idx, "")
                if original_tag:
                    final_text = f"{original_tag} {translated_content}"
                else:
                    final_text = translated_content
                    
                results[idx] = final_text

        return results

    except Exception as e:
        print(f"Error: {e}")
        return {}


# ==========================================
# EXECUTION
# ==========================================

TARGET_LANG_CODE = 'fr' 
folder_path = '../nemo'

# 1. Find file
search_pattern = os.path.join(folder_path, '*.srt')
srt_files = [f for f in glob.glob(search_pattern) if not re.search(r'_[a-z]{2}\.srt$', f)]

if not srt_files:
    print(f"❌ No valid un-translated .srt files found in '{folder_path}/'.")
    exit()

input_file = srt_files[0]
filename_only = os.path.basename(input_file)

# 2. Detect Source Lang
lang_match = re.search(r'\.([a-z]{2})\.', filename_only)
SOURCE_LANG_CODE = lang_match.group(1) if lang_match else 'en'

# 3. Output Name
base_name, ext = os.path.splitext(input_file)
output_file = f"{base_name}_{TARGET_LANG_CODE}{ext}"

print(f"🔍 Input:  {filename_only}")
print(f"🌐 Langs:  {SOURCE_LANG_CODE} -> {TARGET_LANG_CODE}")
print(f"💾 Output: {output_file}")

try:
    subs = pysrt.open(input_file)
except Exception as e:
    print(f"Could not open file: {e}")
    exit()

# --- CHANGED CHUNK SIZE TO 5 ---
# Smaller chunks = Less confusion for small models
chunk_size = 5 
total_chunks = (len(subs) + chunk_size - 1) // chunk_size

print(f"Starting translation: {len(subs)} subtitles.", flush=True)
start_total_time = time.time()

for i in range(0, len(subs), chunk_size):
    chunk = subs[i:i + chunk_size]
    chunk_num = (i // chunk_size) + 1
    
    print(f"--- Chunk {chunk_num}/{total_chunks} ---", flush=True)
    
    translations = translate_chunk(chunk, SOURCE_LANG_CODE, TARGET_LANG_CODE)
    
    for sub in chunk:
        if sub.index in translations:
            sub.text = translations[sub.index]
        else:
            print(f"⚠️ Warning: Missing line {sub.index} (Model failed to return this line)")

# Save
subs.save(output_file, encoding='utf-8')

end_total_time = time.time()
print(f"\n✅ SUCCESS! Saved to {output_file}")
print(f"⏱️ Total Time: {end_total_time - start_total_time:.2f}s")