import runpod
import time
import torchaudio
import yt_dlp
import os
import tempfile
import base64
import time
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

model = None
output_filename = "output.wav"

def handler(event, responseFormat="base64"):
    input = event['input']    
    prompt = input.get('prompt')  
    yt_url = input.get('yt_url')  
    audio_b64 = input.get('audio_base64')

    print(f"New request. Prompt: {prompt}")

    try:
        # Get audio prompt, either from YouTube or base64 audio
        wav_file = prepare_audio_prompt(yt_url=yt_url, audio_b64=audio_b64)

        # Prompt Chatterbox
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=wav_file
        )
        # Save as WAV
        torchaudio.save(output_filename, audio_tensor, model.sr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"status": "error", "message": str(e)}

    # Convert to base64 string
    audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)

    if responseFormat == "base64":
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape)
            }
        }
    elif responseFormat == "binary":
        with open(output_filename, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        os.remove(output_filename)
        response = audio_data

    # Clean up temporary files
    if wav_file and os.path.exists(wav_file):
        os.remove(wav_file)

    return response

def prepare_audio_prompt(yt_url=None, audio_b64=None):
    """
    Returns path to WAV file (from base64 or downloaded from YouTube).
    """
    if audio_b64:
        # Save base64 to temp WAV file
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            temp_wav.write(audio_bytes)
            return temp_wav.name
    elif yt_url:
        # Download YouTube audio to WAV
        dl_info, wav_file = download_youtube_audio(yt_url, output_path="./my_audio", audio_format="wav")
        return wav_file
    else:
        raise ValueError("Either yt_url or audio_base64 must be provided.")

def audio_tensor_to_base64(audio_tensor, sample_rate):
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            os.unlink(tmp_file.name)
            return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        raise

def initialize_model():
    global model
    if model is not None:
        print("Model already initialized")
        return model
    print("Initializing ChatterboxTTS model...")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print("Model initialized")

def download_youtube_audio(url, output_path="./downloads", audio_format="mp3", duration_limit=60):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/output.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '44100'
        ],
        'prefer_ffmpeg': True,
    }
    if duration_limit:
        ydl_opts['postprocessors'].append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': audio_format,
        })
        ydl_opts['postprocessor_args'].extend([
            '-t', str(duration_limit)
        ])
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_duration = info.get('duration', 0)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            print(f"Uploader: {info.get('uploader', 'Unknown')}")
            if duration_limit:
                actual_duration = min(duration_limit, video_duration)
                print(f"Downloading first {actual_duration} seconds")
            print("Downloading audio...")
            ydl.download([url])
            print("Download completed successfully!")
            expected_filepath = os.path.join(output_path, f"output.{audio_format}")
            return info, expected_filepath
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler })

