import speech_recognition as sr
import ssl
import whisper_timestamped as whisper
import numpy as np
import torch

listener = sr.Recognizer()
ssl._create_default_https_context = ssl._create_unverified_context

language = "en"
model = "small.en"
model_path = "./model"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading whisper {model} model {language}...")
audio_model = whisper.load_model(model, download_root=model_path, device=device)

def transcribe_audio(audio_np):
    # Convert NumPy array to PyTorch tensor
    audio_tensor = torch.from_numpy(audio_np)
    audio_tensor = audio_tensor.to(torch.float32) / 32768.0
    audio_tensor = audio_tensor.to(device)

    # Transcribe audio using Whisper model
    result = audio_model.transcribe(audio_tensor, language=language, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    return text

while True:
    try:
        with sr.Microphone(sample_rate=24000) as source:
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.2)
            listener.dynamic_energy_threshold = True
            print("Say something...")

            # Record audio until a pause is detected
            audio_data = listener.listen(source, timeout=None, phrase_time_limit=7)

            # Convert audio data to NumPy array
            raw_audio_data = audio_data.get_raw_data(convert_rate=24000, convert_width=2)
            audio_np = np.frombuffer(raw_audio_data, dtype=np.int16)

            # Transcribe the audio using Whisper model
            text_whisper = transcribe_audio(audio_np)
            print(f"Whisper Transcription: {text_whisper}")

    except Exception as e:
        print(f"An error occurred: {e}")
