import time
from transformers import pipeline
import soundfile as sf
import speech_recognition as sr
import whisper

# Load the OpenAI Whisper ASR model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Use the microphone to capture audio
with sr.Microphone() as source:
    listener = sr.Recognizer()
    listener.energy_threshold = 4000
    listener.adjust_for_ambient_noise(source, duration=0.2)
    listener.dynamic_energy_threshold = True
    print("Say something:")
    audio = listener.listen(source, timeout=7, phrase_time_limit=7)

# Convert the audio data to a NumPy array
audio_data = audio.get_wav_data()

# Benchmark recognize_google
start_time = time.time()
x = listener.recognize_google(audio, language='eg-in')
google_transcription_time = time.time() - start_time

# Benchmark whisper.transcriber
start_time = time.time()
transcription = transcriber(audio_data)
whisper_transcription_time = time.time() - start_time

print("recognize_google Transcription:", x.lower())
print("whisper.transcriber Transcription:", transcription["text"].lower())

print("Time taken for recognize_google:", google_transcription_time, "seconds")
print("Time taken for whisper.transcriber:", whisper_transcription_time, "seconds")
