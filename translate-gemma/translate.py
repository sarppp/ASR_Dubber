import pysrt
import time
from ollama import Client

### WORKING VERSION ###
# 1. SETUP: Connect to your Docker Ollama instance
client = Client(host='http://172.17.0.1:11434')
MODEL_NAME = 'translategemma:4b'

def translate_chunk(chunk_subs):
    # Prepare text with indices. We use [idx] to help the model keep track.
    text_to_translate = ""
    for sub in chunk_subs:
        # Replace newlines within a single subtitle with ' | ' to keep it on one line
        clean_text = sub.text.replace('\n', ' | ')
        text_to_translate += f"[{sub.index}] {clean_text}\n"

    # 2. THE PROMPT: Official TranslateGemma Template
    # We use two blank lines before the text as recommended by Google for this model.
    prompt = f"""You are a professional English (en) to French (fr) translator. 
Your goal is to accurately convey the meaning while maintaining a poetic, storytelling tone. 
Produce only the French translation, without any additional explanations. 
Keep the [index] markers and the | separators exactly as they are.

Please translate the following English text into French:


{text_to_translate}"""
    
    print(f"Sending {len(chunk_subs)} lines to {MODEL_NAME}...")
    
    try:
        # 3. GENERATION: Use low temperature (0.0 or 0.1) for strict SRT format
        response = client.generate(
            model=MODEL_NAME, 
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_ctx": 2048  # Perfect size for 10-15 lines of context
            }
        )
        
        raw_output = response['response'].strip()
        translated_lines = raw_output.split('\n')
        
        # 4. PARSING: Extract text back into a dictionary
        results = {}
        for line in translated_lines:
            if ']' in line:
                try:
                    # Split "[index] text"
                    idx_part, txt_part = line.split(']', 1)
                    idx = int(idx_part.replace('[', '').strip())
                    # Convert ' | ' back to real newlines for the SRT file
                    translated_text = txt_part.strip().replace(' | ', '\n').replace('|', '\n')
                    results[idx] = translated_text
                except Exception:
                    continue
        return results

    except Exception as e:
        print(f"Error in model generation: {e}")
        return {}

# 5. EXECUTION: Load and Process
input_file = 'momo.srt'
output_file = 'momo_french.srt'

subs = pysrt.open(input_file)
chunk_size = 10 
total_chunks = (len(subs) + chunk_size - 1) // chunk_size

print(f"Starting translation: {len(subs)} subtitles total.", flush=True)
start_total_time = time.time() # Start the master clock

for i in range(0, len(subs), chunk_size):
    chunk = subs[i:i + chunk_size]
    chunk_num = (i // chunk_size) + 1
    
    print(f"\n--- Chunk {chunk_num}/{total_chunks} ---", flush=True)
    
    translations = translate_chunk(chunk)
    
    for sub in chunk:
        if sub.index in translations:
            sub.text = translations[sub.index]
        else:
            print(f"⚠️ Warning: Missing line {sub.index}", flush=True)

# 6. SAVE & PRINT STATS
subs.save(output_file, encoding='utf-8')

end_total_time = time.time()
total_seconds = end_total_time - start_total_time

print(f"\n✅ SUCCESS! File saved as: {output_file}", flush=True)
print(f"⏱️ Total Time: {total_seconds:.2f} seconds", flush=True)
print(f"🚀 Speed: {len(subs)/total_seconds:.2f} lines per second", flush=True)