'''
# Just run it - automatically finds the video!
modal run modal_app.py --language fr --model turbo

# With Demucs
modal run modal_app.py --language fr --model turbo --use-demucs

# Still works with specific filename
modal run modal_app.py --video-filename interview.mp4 --language en --model turbo

# Default htdemucs (good balance)
modal run modal_app.py --language fr --model turbo --use-demucs

# Fine-tuned htdemucs (better quality, slightly slower)
modal run modal_app.py --language fr --model turbo --use-demucs --demucs-model htdemucs_ft

# MDX Extra (highest quality, slower)
modal run modal_app.py --language fr --model turbo --use-demucs --demucs-model mdx_extra

# Process all videos in folder
cd /path/to/videos
modal run modal_app.py --language fr --model turbo --use-demucs

# French video, get English translation
modal run modal_app.py --language fr --model turbo --translate

# French video, get English translation with Demucs
modal run modal_app.py --language fr --model turbo --use-demucs --translate

# Without --translate flag = transcription in original language
modal run modal_app.py --language fr --model turbo

uv run --env-file .env modal run whisper_modal_asr.py --language de --model turbo --use-demucs

Nvidia B200 $0.001736 / sec
Nvidia H200 $0.001261 / sec
Nvidia H100 $0.001097 / sec
Nvidia A100, 80 GB $0.000694 / sec
Nvidia A100, 40 GB $0.000583 / sec
Nvidia L40S $0.000542 / sec
Nvidia A10 $0.000306 / sec
Nvidia L4 $0.000222 / sec
Nvidia T4 $0.000164 / sec
'''


import modal
from pathlib import Path
import subprocess
import os

# --- Configuration ---
REMOTE_IO_PATH = Path("/app/whisper_codes") 
VOLUME_NAME = "whisper-demucs-cache"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- IMAGE DEFINITION WITH DEMUCS ---
def download_whisper_models():
    """Pre-download Whisper models into the image"""
    import whisper
    # Download commonly used models during image build
    print("📦 Pre-downloading Whisper models...")
    whisper.load_model("turbo")
    whisper.load_model("large-v3")
    print("✅ Models cached in image")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install("numpy<2.0")
    .pip_install(
        "torch==2.1.0", 
        "torchvision==0.16.0", 
        "torchaudio==2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "openai-whisper", 
        "librosa",
        "python-dotenv",
        "diffq",
        "julius",
        "einops",
        "soundfile",
    )
    .pip_install("demucs==4.0.0")
    .run_function(download_whisper_models)  # Cache models in image!
)

app = modal.App(name="whisper-demucs-transcriber-v2")

# --- Helper Functions ---

def format_as_srt(segments) -> str:
    """Convert Whisper segments to SRT format"""
    srt_lines = []
    
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# --- Core Pipeline ---

def run_transcription_pipeline(
    video_filename: str, 
    input_dir: Path, 
    output_lang: str = "de", 
    model_name: str = "large-v3",
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    task: str = "transcribe"  # NEW: 'transcribe' or 'translate'
) -> bytes:
    import whisper
    import librosa
    import torch
    import warnings
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    remote_video_path = input_dir / video_filename
    
    if not remote_video_path.exists():
        print(f"❌ File not found: {remote_video_path}")
        try:
            print(f"   Contents of {input_dir}: {[f.name for f in input_dir.iterdir()]}")
        except Exception:
            pass
        raise FileNotFoundError(f"Video file not found: {video_filename}")

    base_name = remote_video_path.stem
    audio_path = input_dir / f"{base_name}.wav"

    # 1. Video to WAV Conversion
    print(f"🔄 Converting {remote_video_path.name} to WAV...")
    command = [
        "ffmpeg", "-y", "-i", str(remote_video_path), 
        "-acodec", "pcm_s16le", "-ar", "44100", str(audio_path)
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFMPEG Error: {e.stderr.decode()}")
        raise e

    # 2. Optional: Demucs vocal separation
    if use_demucs:
        print(f"🎵 Running Demucs ({demucs_model}) to isolate vocals...")
        separated_dir = input_dir / "separated"
        
        demucs_cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", demucs_model,
            "-o", str(separated_dir),
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(demucs_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                vocal_files = list(separated_dir.rglob("vocals.wav"))
                
                if vocal_files:
                    audio_path = vocal_files[0]
                    print(f"   ✅ Vocals isolated successfully")
                else:
                    print(f"   ⚠️ Vocals not found, using original audio")
            else:
                print(f"   ⚠️ Demucs failed, using original audio")
                
        except Exception as e:
            print(f"   ⚠️ Demucs error, using original audio")

    # 3. Whisper Transcription/Translation
    print(f"🧠 Loading Whisper model ('{model_name}')...")
    audio_data, _ = librosa.load(str(audio_path), sr=16000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    model = whisper.load_model(model_name, device=device)

    # Build transcription parameters based on task
    transcribe_params = {
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4
    }
    
    if task == "translate":
        print(f"🌐 Starting translation to English (auto-detecting source language)...")
        transcribe_params["task"] = "translate"
        # Don't specify language - let Whisper auto-detect for better translation
        # Some models ignore task="translate" when language is explicitly set
    else:
        print(f"🎧 Starting transcription (language: {output_lang})...")
        transcribe_params["language"] = output_lang
    
    result = model.transcribe(audio_data, **transcribe_params)
    
    # 4. Generate SRT
    print("📝 Generating SRT format...")
    srt_content = format_as_srt(result["segments"])
    
    # Cleanup
    if os.path.exists(input_dir / f"{base_name}.wav"):
        os.remove(input_dir / f"{base_name}.wav")
    
    if use_demucs and (input_dir / "separated").exists():
        import shutil
        shutil.rmtree(input_dir / "separated")
    
    return srt_content.encode('utf-8')


# --- Remote Function ---
@app.function(
    image=image,
    volumes={REMOTE_IO_PATH: volume},
    gpu="T4",
    timeout=1800,
)
def transcribe_remote(
    video_filename: str, 
    video_data: bytes, 
    language: str = "de", 
    model: str = "large-v3",
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    task: str = "transcribe"  # NEW
) -> bytes:
    """
    Transcribe or translate video
    
    Args:
        video_filename: Name of the video file
        video_data: Video file content as bytes
        language: Language code for source audio
        model: Whisper model ('turbo', 'large-v3', 'medium', etc.)
        use_demucs: Whether to isolate vocals before transcription
        demucs_model: Demucs model ('htdemucs', 'htdemucs_ft', 'mdx_extra')
        task: 'transcribe' (same language) or 'translate' (to English only)
    """
    REMOTE_IO_PATH.mkdir(parents=True, exist_ok=True)
    
    video_path = REMOTE_IO_PATH / video_filename
    with open(video_path, "wb") as f:
        f.write(video_data)
    
    result = run_transcription_pipeline(
        video_filename, 
        REMOTE_IO_PATH, 
        output_lang=language, 
        model_name=model,
        use_demucs=use_demucs,
        demucs_model=demucs_model,
        task=task  # NEW
    )
    
    if video_path.exists():
        video_path.unlink()
    
    return result


# --- Local Entrypoint ---
@app.local_entrypoint()
def main(
    video_filename: str = None,
    language: str = "de", 
    model: str = "large-v3",
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    translate: bool = False  # NEW: Add --translate flag for English translation
):
    """
    Transcribe or translate video to SRT subtitles
    
    Examples:
        # Transcribe (same language)
        modal run modal_app.py --language de --model turbo
        
        # Translate to English
        modal run modal_app.py --language fr --model turbo --translate
        
        # With Demucs vocal isolation + translation
        modal run modal_app.py --language fr --model turbo --use-demucs --translate
    """
    local_io_path = Path(".")
    
    # Determine task from translate flag
    task = "translate" if translate else "transcribe"
    
    # Auto-detect video file if not specified
    if video_filename is None:
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v']
        all_video_files = [f for f in local_io_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in video_extensions]
        
        if not all_video_files:
            print("❌ Error: No video files found in current directory")
            print(f"   Looking for: {', '.join(video_extensions)}")
            return
        
        # Determine SRT suffix based on task
        srt_suffix = ".en.srt" if task == "translate" else ".srt"
        
        # Filter out videos that already have SRT files
        video_files = []
        skipped_files = []
        for video in all_video_files:
            srt_file = video.parent / (video.stem + srt_suffix)
            if srt_file.exists():
                skipped_files.append(video.name)
            else:
                video_files.append(video)
        
        if skipped_files:
            print(f"⏭️  Skipping {len(skipped_files)} video(s) with existing SRT:")
            for name in skipped_files:
                print(f"   ✓ {name}")
        
        if not video_files:
            print("\n✅ All videos already have SRT files! Nothing to do.")
            return
        
        if len(video_files) == 1:
            local_video_path = video_files[0]
            print(f"🎯 Auto-detected video: {local_video_path.name}")
        else:
            print(f"\n📹 Found {len(video_files)} video(s) without SRT:")
            for i, f in enumerate(video_files, 1):
                print(f"   {i}. {f.name}")
            local_video_path = video_files[0]
            print(f"🎯 Using first file: {local_video_path.name}")
            print("   (Specify --video-filename to choose a different file)")
    else:
        local_video_path = local_io_path / video_filename
        
        if not local_video_path.exists():
            print(f"❌ Error: Video file not found at {local_video_path.absolute()}")
            return
        
        srt_suffix = ".en.srt" if task == "translate" else ".srt"
        srt_file = local_video_path.parent / (local_video_path.stem + srt_suffix)
        if srt_file.exists():
            print(f"⏭️  Skipping: SRT file already exists: {srt_file.name}")
            print("   (Delete the SRT file first if you want to regenerate it)")
            return

    print(f"\n📥 Found video at: {local_video_path.absolute()}")
    print(f"🌍 Source Language: {language}")
    print(f"📋 Task: {task.upper()}" + (" (to English)" if task == "translate" else ""))
    print(f"🤖 Whisper Model: {model}")
    if use_demucs:
        print(f"🎵 Demucs: ✅ Enabled ({demucs_model})")
    else:
        print(f"🎵 Demucs: ❌ Disabled")
    
    print("📤 Reading video file...")
    with open(local_video_path, "rb") as f:
        video_data = f.read()
    
    print("🚀 Sending job to Modal...")
    
    # Debug: Show what task is being used
    print(f"🔧 Debug: task = '{task}'")
    
    srt_bytes = transcribe_remote.remote(
        local_video_path.name,
        video_data, 
        language, 
        model,
        use_demucs,
        demucs_model,
        task  # Must pass task parameter!
    )
    
    # Save with appropriate suffix
    srt_suffix = ".en.srt" if task == "translate" else ".srt"
    srt_filename = local_video_path.stem + srt_suffix
    local_srt_path = local_io_path / srt_filename
    
    with open(local_srt_path, "wb") as f:
        f.write(srt_bytes)
    
    print("\n--- Final Status ---")
    print(f"✅ {'Translation' if task == 'translate' else 'Transcription'} complete. SRT saved to: {local_srt_path.absolute()}")