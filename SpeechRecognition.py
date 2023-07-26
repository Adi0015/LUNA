import speech_recognition as sr
import os
# Initialize the recognizer

def say(text):
    os.system(f"say {text}")

def takeCommand():
    listener = sr.Recognizer()
    with sr.Microphone() as source:
        listener.pause_threshold = 1
        listener.energy_threshold = 4000
        listener.adjust_for_ambient_noise(source, duration=0.1)
        audio = listener.listen(source, timeout=None,phrase_time_limit=7)  # listen for the first phrase and extract it into audio data
        x = listener.recognize_google(audio, language='eg-IN')  # recognize speech using Google Speech Recognition
        query = x.lower()
        print(f"User said :{query}")

if __name__ == '__main__' :
    # print("yay")
    say("Hey , i am LUNA your Virtual Asisitant")
    print("listening...")
    text = takeCommand()
    say (text)