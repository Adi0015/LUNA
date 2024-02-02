import speech_recognition as sr
import ssl
from urllib.request import urlopen
import whisper_timestamped as whisper
import threading
import io
import wave
import librosa
import pandas as pd
import numpy as np 
import socket
import time
import pyaudio
import speaker_verification_toolkit.tools as svt
from pydub import AudioSegment
from pydub.playback import play
from vosk import Model, KaldiRecognizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import load_model

# ! Basic requirements
model = Model(r"vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model, 16000)
listener = sr.Recognizer()
ssl._create_default_https_context = ssl._create_unverified_context
model = whisper.load_model("tiny", device="cpu")
mic = pyaudio.PyAudio() 
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
# stream.start_stream()

# ! Variable to store connection status
verification = False
connected = True  
audio_chunks = []
#! Check if the system is connected to the internet and update 'connected'
def is_connected():
    global connected
    while True:
        try:
            # ! Attempt to connect to a well-known website (Google's public DNS server)
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            connected = True
        except OSError:
            connected = False
        time.sleep(10)  # ! Check every 10 seconds

#! Start the connection checking thread
connection_thread = threading.Thread(target=is_connected)
connection_thread.daemon = True
connection_thread.start()

#! Function to get microphone input when the system is offline
def offline_mic_input():
    try:
        with stream.start_stream():
            data = stream.read(45000)
            if recognizer.AcceptWaveform(data):
                text = recognizer.Result()
                print(text)
                if 'luna' in text:
                    # Save the current chunk for verification
                    audio_chunks.append(np.frombuffer(data, dtype=np.int16))
                    audio_data = np.concatenate(audio_chunks) 
                    # audio_chunks = []
                    verification_thread = threading.Thread(target=speaker_verification,args=(audio_data,))
                    verification_thread.start()
            stream.stop_stream()
            return text[14:3].lower()
    except Exception:
        pass
        luna_output()

#! Function to get microphone input when the system is online
def online_mic_input():
    try:
        with sr.Microphone() as source:
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.2)
            listener.dynamic_energy_threshold = True
            audio = listener.listen(source, timeout=7, phrase_time_limit=7)
            audio_data = audio.get_wav_data()
            x = listener.recognize_google(audio, language='eg-in')
            text = x.lower()
            print(text)
            if 'luna' in text:
                verification_thread = threading.Thread(target=speaker_verification, args=(audio_data))
                verification_thread.start()
            return text
            
    except Exception:  
        return ""

#! Function to extract timestamp from audio
def luna_timestamp(audio):
    result = whisper.transcribe(model, audio, language="en")
    if "segments" in result:
        for segment in result["segments"]:
             if "words" in segment:
                for word in segment["words"]:
                    if word["text"].lower() == "luna":
                        start_time = word["start"]
                        end_time = word["end"]
                        print(f"Start Time: {start_time}, End Time: {end_time}")
                        return start_time.astype(float), end_time.astype(float)

#! Function for speaker verification
def speaker_verification(audio_data):
    # audio_data = audio.get_wav_data()
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(45000)
        wf.writeframes(audio_data)
    audio_buffer.seek(0)
    audio_segment = AudioSegment.from_wav(audio_buffer)
    play(audio_segment)
    audio_df = audio_segment[0:]
    audio_df.export(".hidden/luna_audio.wav", format="wav")
    audio = whisper.load_audio(".hidden/luna_audio.wav")
    time_stamp = luna_timestamp(audio)
    print(time_stamp[0], time_stamp[1])
    audio_df = audio_segment[time_stamp[0]*1000:time_stamp[1]*1000]
    audio_df.export(".hidden/audio.wav", format="wav")
    audio_file = ".hidden/audio.wav"
    preprocessing_thread = threading.Thread(target=audio_preprocessing, args=(audio_file,))
    preprocessing_thread.start()

#! Function for audio preprocessing
def audio_preprocessing(audio_file):
    y, sr = librosa.load(audio_file, sr=45000, mono=True)
    y = svt.rms_silence_filter(y, threshold=0.002)
    y = librosa.effects.deemphasis(y)
    y = librosa.effects.deemphasis(y)
    y = librosa.effects.deemphasis(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i+1}' for i in range(20)])
    df['Time'] = librosa.frames_to_time(np.arange(len(df)), sr=sr)
    df["ID"] = 0.0
    x_test = scale_dataset(df, oversample=False) 
    global verification
    verification = model_prediction(x_test)
    return verification
    
#! Scale the dataset for speaker verification.
def scale_dataset(dataframe, oversample=False):
    mfcc_columns = dataframe.columns[:20]
    x_mfcc = dataframe[mfcc_columns].values
    if dataframe.shape[1] == 23:
        time_id_columns = dataframe.columns[20:-1]
        x_time_id = dataframe[time_id_columns].values
        y = dataframe[dataframe.columns[-1]].values
        scaler_mfcc = StandardScaler()
        x_mfcc_scaled = scaler_mfcc.fit_transform(x_mfcc)
        x_scaled = np.hstack((x_mfcc_scaled, x_time_id))
        if oversample:
            ros = RandomOverSampler()
            x_scaled, y = ros.fit_resample(x_scaled, y)
        data = np.hstack((x_scaled, np.reshape(y, (-1, 1))))
        return data.astype('float64'), x_scaled.astype('float64'), y.astype('float64')
    else:
        time_id_columns = dataframe.columns[20:]
        x_time_id = dataframe[time_id_columns].values
        scaler_mfcc = StandardScaler()
        x_mfcc_scaled = scaler_mfcc.fit_transform(x_mfcc)
        x_scaled = np.hstack((x_mfcc_scaled, x_time_id))
        x_scaled = np.hstack((x_mfcc_scaled, x_time_id))
        return x_scaled.astype('float64')

#! Function for model prediction
def model_prediction(x_test_set):
    model = load_model("lstm_model.h5")
    y_pred = (model.predict(x_test_set) > 0.5).astype(int).reshape(-1,)
    percentage_of_ones = (y_pred == 1).mean() * 100
    if percentage_of_ones >= 80:
        return True
    else:
        return False
    
#!  Voice training
# * def trainVoice():
    
#!  Determine the appropriate mic input method based on the connection status.
def luna_output():
    if connected:
        mic_text = online_mic_input()
    else:
        mic_text = offline_mic_input()
    return mic_text

# ! Main function to continuously run the code
def luna():
    print(verification)
    while True:
        mic_text = luna_output()
        if 'luna' in mic_text:
            print(mic_text)

if __name__ == '__main__':
    luna()
