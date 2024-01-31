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
from vosk import Model , KaldiRecognizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import load_model


#! basic requirements
model = Model(r"vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model,16000)
listener = sr.Recognizer()
ssl._create_default_https_context = ssl._create_unverified_context
model = whisper.load_model("tiny", device="cpu")

def is_connected():
    try:
        # Attempt to connect to a well-known website (in this case, Google's public DNS server)
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        pass
    return False
def offlineMicInput():
    try:
        with pyaudio.PyAudio() as mic:
          stream = mic.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True, frames_per_buffer=8192)
          stream.start_stream()
          data = stream.read(4096)
          if recognizer.AcceptWaveform(data):
            text = recognizer.Result()
            text = text[14:3].lower()
            # print(text[14:-3])
            print(text)
            return text
    except Exception:
        pass
        lunaOutput()
def onlineMicInput():
    try:
        with sr.Microphone() as source:  # ?! microphone as the audio source
            # listener.pause_threshold = 0.5
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.2)
            listener.dynamic_energy_threshold = True
            
            # ! listen for the first phrase and extract it into audio data
            audio = listener.listen(source, timeout=7, phrase_time_limit=7)

            # ! recognize speech using Google Speech Recognition
            x = listener.recognize_google(audio, language='eg-in')
            text = x.lower()
            print(text)
            if 'luna' in text:
                verification_thread = threading.Thread(target=speakerVerification, args=(audio,))
                verification_thread.start()
                # Verification(audio)
            return  text
            
    except Exception:  
        return ""

def lunaTimestamp(audio):
  # audio = whisper.load_audio(audio)
  # audio_array = np.frombuffer(audio_data, dtype=np.float32)
  result = whisper.transcribe(model, audio, language="en")
  if "segments" in result:
    for segment in result["segments"]:
         if "words" in segment:
            for word in segment["words"]:
                if word["text"].lower() == "luna":
                    start_time = word["start"]
                    end_time = word["end"]
                    print(f"Start Time: {start_time}, End Time: {end_time}")
                    return start_time.astype(float),end_time.astype(float)
                

def speakerVerification(audio):
    audio_data = audio.get_wav_data()
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(45000)
        wf.writeframes(audio_data)
    audio_buffer.seek(0)
    audio_segment = AudioSegment.from_wav(audio_buffer)
    audio_df = audio_segment[0:]
    audio_df.export(".hidden/luna_audio.wav", format="wav")
    audio = whisper.load_audio(".hidden/luna_audio.wav")
    timeStamp = lunaTimestamp(audio)
    print(timeStamp[0],timeStamp[1])
    audio_df = audio_segment[timeStamp[0]*1000:timeStamp[1]*1000]
    audio_df.export(".hidden/audio.wav", format="wav")
    audio_file = ".hidden/audio.wav"
    preprocessing_thread = threading.Thread(target=audioPreprocessing, args=(audio_file,))
    preprocessing_thread.start()
    # audioPreprocessing(audio_file)

    def audioPreprocessing(audioFile):
      y, sr = librosa.load(audioFile, sr=45000, mono=True)
      # t = librosa.frames_to_time(np.arange(len(y)),sr=45000)
      y = svt.rms_silence_filter(y,threshold=0.002)
      y = librosa.effects.deemphasis(y)
      y = librosa.effects.deemphasis(y)
      y = librosa.effects.deemphasis(y)
      mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
      df = pd.DataFrame(mfcc.T, columns=[f'MFCC_{i+1}' for i in range(20)])
      df['Time'] = librosa.frames_to_time(np.arange(len(df)), sr=sr)
      df["ID"]=0.0
      # print(df)
      # df.info()
      X_test = scale_dataset(df,oversample=False) 
      verified = modelPrediction(X_test)
      return verified

def modelPrediction(X_testSet):
    model = load_model("lstm_model.h5")
    y_pred =(model.predict(X_testSet)>0.5).astype(int).reshape(-1,)
    percentage_of_ones = (y_pred == 1).mean() * 100
    if percentage_of_ones >= 80:
        return True
    else:
        return False

def scale_dataset(dataframe, oversample=False):
    # Separate MFCC features from 'Time' and 'ID'
    mfcc_columns = dataframe.columns[:20]
    X_mfcc = dataframe[mfcc_columns].values
    if dataframe.shape[1] == 23:
      time_id_columns = dataframe.columns[20:-1]
      X_time_id = dataframe[time_id_columns].values
      y = dataframe[dataframe.columns[-1]].values
      scaler_mfcc = StandardScaler()
      X_mfcc_scaled = scaler_mfcc.fit_transform(X_mfcc)
      X_scaled = np.hstack((X_mfcc_scaled, X_time_id))
      if oversample:
          ros = RandomOverSampler()
          X_scaled,y = ros.fit_resample(X_scaled,y)
      data = np.hstack((X_scaled, np.reshape(y, (-1, 1))))
      return data.astype('float64'), X_scaled.astype('float64'), y.astype('float64')
    else:
      time_id_columns = dataframe.columns[20:]
      X_time_id = dataframe[time_id_columns].values
      scaler_mfcc = StandardScaler()
      X_mfcc_scaled = scaler_mfcc.fit_transform(X_mfcc)
      X_scaled = np.hstack((X_mfcc_scaled, X_time_id))
      X_scaled = np.hstack((X_mfcc_scaled, X_time_id)) 
      return X_scaled.astype('float64')

def lunaOutput():
    if is_connected():
      micText = onlineMicInput()
    else:
      micText = offlineMicInput()
    
    if 'luna' in micText:
        print(micText)

def Luna():  # Main Function
    while True:
        lunaOutput() # todo build all linked function

if __name__ == '__main__':
  Luna()

