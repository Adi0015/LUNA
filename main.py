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
import speaker_verification_toolkit.tools as svt
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import load_model

# ! Basic requirements

listener = sr.Recognizer()
ssl._create_default_https_context = ssl._create_unverified_context
model = whisper.load_model("base.en", device="cpu")
transcript = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# mic = pyaudio.PyAudio() 
# stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
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
        with sr.Microphone() as source:
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.2)
            listener.dynamic_energy_threshold = True
            audio = listener.listen(source, timeout=7, phrase_time_limit=7)
            audio_data = audio.get_wav_data()
            text = transcript(audio_data)
            if 'luna' in text["text"]:
                verification_thread = threading.Thread(target=speaker_verification, args=(audio_data,))
                verification_thread.start()
            return text["text"]
            
    except Exception:  
        return ""


#! Function to get microphone input when the system is online
def online_mic_input():
    try:
        with sr.Microphone(sample_rate=16000) as source:
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.2)
            listener.dynamic_energy_threshold = True
            print("say")
            audio = listener.listen(source, timeout=None, phrase_time_limit=7)
            audio_data = audio.get_wav_data()
            x = listener.recognize_google(audio, language='eg-in')
            text = x.lower()
            print(text)
            if 'luna' in text:
                # verification_thread = threading.Thread(target=speaker_verification, args=(audio_data,))
                # verification_thread.start()
                result = speaker_verification(audio_data)
                print(result)
            return text
            
    except Exception:  
        return ""

#! Function to extract timestamp from audio
def luna_timestamp(audio):
    result = whisper.transcribe(model, audio, language="en")
    print(result)
    if "segments" in result:
      segments = result["segments"]
      for i in range(len(segments)):
          segment = segments[i]
          if "words" in segment:
              for j in range(len(segment["words"])):
                  word = segment["words"][j]
                  if "luna" in word["text"].lower():
                      start_time = word["start"]
                      if j + 1 < len(segment["words"]):
                          end_time = segment["words"][j + 1]["start"]  # Set end time to start time of the next word
                      else:
                          end_time = segment["end"]  # If it's the last word in the segment, set end time to the end of the segment
                      print(f"Start Time: {start_time}, End Time: {end_time}")
                      return start_time, end_time
                        # You can use start_time and end_time as needed


#! Function for speaker verification
def speaker_verification(audio_data):
    # audio_data = audio.get_wav_data()
    audio_stream = io.BytesIO(audio_data)
    with wave.open(audio_stream, 'rb') as wave_file:
                # Get the sampling rate (frames per second)
        sr = wave_file.getframerate()
        ch = wave_file.getnchannels()
        sampwidth = wave_file.getsampwidth()
        print(f"Sampling Rate (sr): {sr} , Channel : {ch}, sampwidth : {sampwidth}")
        
    # audio_buffer = io.BytesIO()
    # with wave.open(audio_buffer, "wb") as wf:
    #     wf.setnchannels(1)
    #     wf.setsampwidth(2)
    #     wf.setframerate(48000)
    #     wf.writeframes(audio_data)
    # audio_buffer.seek(0)
    audio_segment = AudioSegment.from_wav(audio_stream)
    # play(audio_segment)
    # audio_df = audio_segment
    audio_segment.export(".hidden/luna_audio.wav", format="wav")
    audio = whisper.load_audio(".hidden/luna_audio.wav")
    print("loaded")
    time_stamp = luna_timestamp(audio)
    print(time_stamp[0], time_stamp[1])
    audio_df = audio_segment[time_stamp[0]*1000:time_stamp[1]*1000]
    audio_df.export(".hidden/audio.wav", format="wav")
    audio_file = ".hidden/audio.wav"
     # preprocessing_thread = threading.Thread(target=audio_preprocessing, args=(audio_file,))
    # preprocessing_thread.start()
    status = audio_preprocessing(audio_file)
    return status

#! Function for audio preprocessing
def audio_preprocessing(audio_file):
    y, sr = librosa.load(audio_file, sr=48000, mono=True)
    y = svt.rms_silence_filter(y, threshold=0.002)
    y = librosa.effects.deemphasis(y)
    y = librosa.effects.deemphasis(y)
    y = librosa.effects.deemphasis(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i+1}' for i in range(20)])
    df['Time'] = librosa.frames_to_time(np.arange(len(df)), sr=sr)
    df["ID"] = 0.0
    print(df)
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
    print(percentage_of_ones)
    if percentage_of_ones >= 80:
        return True
    else:
        return False
    
#!  Voice training
# * def trainVoice():
    
#!  Determine the appropriate mic input method based on the connection status.
def luna_output():
    if connected == True:
        print("yessss")
        mic_text = online_mic_input()
        # print(mic_text)
        return mic_text
    else:
        mic_offline_text = offline_mic_input()
        return mic_offline_text

# ! Main function to continuously run the code
def luna():
    print(verification)
    while True:
        mic_text = luna_output()
        if 'luna' in mic_text:
            print(mic_text)

if __name__ == '__main__':
    luna()
