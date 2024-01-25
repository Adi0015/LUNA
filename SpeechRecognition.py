import speech_recognition as sr
import io
import wave
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
from pydub.playback import play

listener = sr.Recognizer()


def micInput():
    try:
        with sr.Microphone() as source:  # ?! microphone as the audio source
            # listener.pause_threshold = 0.4
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.2)
            # ! listen for the first phrase and extract it into audio data
            audio = listener.listen(source, timeout=None, phrase_time_limit=10)
            # ! recognize speech using Google Speech Recognition
            x = listener.recognize_google(audio, language='eg-in')
            command = x.lower()
            if 'luna' in x.lower():
              Verifcation(audio)
            return  command
            # print(command)
    except Exception:  # ? speech is unintelligible
        return ""


def Verifcation(audio):
    
    audio_data = audio.get_wav_data()
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(45000)
        wf.writeframes(audio_data)
    audio_buffer.seek(0)
    # play(audio_data)
    print("step 1 done")
    audio_segment = AudioSegment.from_wav(audio_buffer)
    play(audio_segment)
    
        
def extract_luna_audio(audio_data, keyword='luna'):
    audio = AudioSegment(audio_data)

    
   
    
def lunaOutput():
    micText = micInput()
    if 'luna' in micText:
        print(micText)

def Luna():  # ! Main Function
    while True:
        lunaOutput() # todo build all linked function

if __name__ == '__main__':
  Luna()

