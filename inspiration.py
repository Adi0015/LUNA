import datetime
import os
import smtplib
import socket
import subprocess
import webbrowser
from tkinter import *
import keyboard
import psutil  # pip install psutil
import pyautogui
import pyttsx3
import pywhatkit
import requests
import screen_brightness_control as sbc
import speech_recognition as sr
from pygame import mixer

listener = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 130)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
music = mixer.init()


def is_connected():
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False


def talk(command):
    #var1.set(command)
    engine.say(command)
    engine.runAndWait()

def text_box1(command):
    var.set(command)
    talk(command)

def text_box2(command):
    var1.set(command)
    talk(command)

def take_command():
    try:
        with sr.Microphone() as source:  # use the default microphone as the audio source
            listener.pause_threshold = 0.5
            listener.energy_threshold = 4000
            listener.adjust_for_ambient_noise(source, duration=0.1)
            audio = listener.listen(source, timeout=None,phrase_time_limit=7)  # listen for the first phrase and extract it into audio data
            x = listener.recognize_google(audio, language='eg-in')  # recognize speech using Google Speech Recognition
            command = x.lower()
    except Exception:  # speech is unintelligible
        return ""
    return command


def run_luna():
    command=take_command()
    var.set(command)
    print(command)
    if 'shut down' in command:
        run_shutdown()
    elif 'restart' in command:
        run_restart(command)
    elif 'song on youtube' in command:
        song = command.replace('play', '')
        talk("playing")
        pywhatkit.playonyt(song)
    elif 'send email' in command:
        try:
            text_box2("What should I say?")
            content = take_command().lower()
            text_box2(" to whom do u want to send email")
            #var1.set("to whom do u want to send email")
            emailaddress = take_command()
            sendEmail(emailaddress, content)
            var1.set("Email has been sent..!")
            talk("Email has been sent!")
        except Exception as e:
            print(e)
            var1.set("Sorry ma'am. I am not able to send this email")
            talk("Sorry ma'am. I am not able to send this email")
            wake_up()
    elif 'weather' in command:
        weather(command)
    elif 'open' in command:
        run_program(command)
    elif 'close' in command:
        close_program(command)
    elif 'group' in command:
        grp_info()
    elif 'time' in command:
        time()
    elif "peek at the desktop" in command:
        keyboard.send("Windows key + ,")
    elif 'go to desktop' in command:
        keyboard.send("windows + d")
    elif "snapshot" in command:
        keyboard.send("shift+windows+s")
    elif "screenshot" in command:
        keyboard.send("windows+print screen")
    elif "task manager" in command:
        keyboard.send("ctrl+shift+esc")
    elif "volume up" in command or "up volume" in command:
        volume_up(command)
    elif "volume down" in command or "down volume" in command:
        volume_down(command)
    elif "mute volume" in command:
        pyautogui.press("volumemute")
    elif "increase brightness" in command:
        brightness_up(command)
    elif "decrease brightness" in command:
        brightness_down(command)
    elif "find" in command:
        keyboard.send("Ctrl + F")
    elif 'search about' in command or 'search for' in command:
        if 'search about' in command:
            command = command.replace('search about ', '')
        elif 'search for' in command:
            command = command.replace('search for ', '')
        query = command
        webbrowser.open('https://www.google.com/search?q=' + query)
    else:
        if "" in command:
            wake_up()
        else:
            var1.set("Please say the command again")
            talk("Please say the command again")
            run_luna()


def sendEmail(to, content):
    email = smtplib.SMTP('smtp.gmail.com', 587)
    email.ehlo()
    email.starttls()
    email.login('your-mail', 'your-password')
    email.sendmail('your-mail', to, content)
    email.quit()


def brightness_up(command):
    if "by 25" in command:
        if sbc.get_brightness() < 100 :
            brightness = sbc.get_brightness()+25
            sbc.set_brightness(brightness)
            var1.set("Brightness increased by 25 %")
            talk("Brightness increased by 25 %")
        else:
            var1.set("Brightness is full..!")
            talk("Brightness is full")

    elif "by 50" in command:
        if sbc.get_brightness() < 100:
            brightness = sbc.get_brightness()+50
            sbc.set_brightness(brightness)
            var1.set("Brightness increased by 50% ")
            talk("Brightness increased by 50 %")
        else:
            var1.set("Brightness is full..!")
            talk("Brightness is full")
    else :
        if sbc.get_brightness() < 100:
            brightness = sbc.get_brightness()+10
            sbc.set_brightness(brightness)
            var1.set("Brightness increased by 10% ")
            talk("Brightness increased by 10 %")
        else:
            var1.set("Brightness is full..!")
            talk("Brightness is full")


def brightness_down(command):
    if "by 25" in command:
        if sbc.get_brightness() > 10:
            brightness = sbc.get_brightness()-25
            sbc.set_brightness(brightness)
            var1.set("Brightness decreased by 25% ")
            talk("Brightness decreased by 25 %")
        else:
            var1.set("Brightness is 10..")
            talk("Brightness is 10")
    elif "by 50" in command:
        if sbc.get_brightness() < 10:
            brightness = sbc.get_brightness()-50
            sbc.set_brightness(brightness)
            var1.set("Brightness decreased by 50% ")
            talk("Brightness decreased by 50 %")
        else:
            var1.set("Brightness is 10..")
            talk("Brightness is 10")
    else:
        if sbc.get_brightness() < 10:
            brightness = sbc.get_brightness()-10
            sbc.set_brightness(brightness)
            var1.set("Brightness decreased by 10% ")
            talk("Brightness decreased by 10 %")
        else:
            var1.set("Brightness is zero..!")
            talk("Brightness is zero")

def volume_up(command):
    if "by 10" in command:
        volume = 0;
        while volume < 5:
            pyautogui.press("volumeup")
            volume += 1;

    elif "by 20" in command:
        volume = 0;
        while volume < 8:
            pyautogui.press("volumeup")
            volume += 1;

    elif "by 50" in command:
        volume = 0;
        while volume < 20:
            pyautogui.press("volumeup")
            volume += 1;
            return volume

    else:
        volume = 0;
        pyautogui.press("volumeup")
        volume += 1;


def volume_down(command):
    if "by 10" in command:
        volume = 0
        while volume < 5:
            pyautogui.press("volumedown")
            volume += 1

    elif "by 20" in command:
        volume = 0
        while volume < 12:
            pyautogui.press("volumedown")
            volume += 1

    elif "by 50" in command:
        volume = 0
        while volume < 20:
            pyautogui.press("volumedown")
            volume += 1

    else:
        volume = 0
        while volume < 5:
            pyautogui.press("volumedown")
            volume += 1


def run_shutdown():
    talk("Do u want to switch off the computer sir ? ")
    #print("Do you want to switch off the computer sir ?")
    take = take_command()
    choice = take
    if choice == 'yes':
        # Shutting down
        #var1.set("Shutting down the computer")
        talk("Shutting the computer ")
        os.system("shutdown /s /t 5")
    if choice == 'no':
        # Idle
        #var1.set("Thank u sir")
        talk("Thank u sir")


def run_restart():
    talk("Do u want to Restart the computer sir ? ")
    var1.set("Do you want to Restart the computer sir ?")
    take = take_command()
    choice = take
    if choice == 'yes':
        # restarting computer
        var1.set("Restarting the computer")
        talk("Restarting the computer")
        os.system("shutdown /r /t 1")
    if choice == 'no':
        # Idle
        var1.set("Thank u sir")
        talk("Thank u sir")


def doesFileExists(filePathAndName):
    return os.path.exists(filePathAndName)


def checkIfProcessRunning(processName):
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;


def run_program(command):
    choice = command
    mixer.music.load("C:\sound\simple_notification.mp3")
    mixer.music.play()
    try:
        if "Open Task Manager" in command:
            talk('Opening task manager')
            keyboard.send("Ctrl + Shift + Esc")
        elif "Open Casting settings" in command:
            talk('Opening casting settings')
            keyboard.send("Windows key + K")
        elif "Open windows" in command:
            talk('Opening windows')
            keyboard.send("windows")
        elif "Open Settings app" in command:
            talk('Opening settings')
            keyboard.send("Windows key + I")
        elif "Open File Explorer" in command:
            keyboard.send("Windows key + E")
        elif "Open Action center" in command:
            talk('Opening action centre')
            keyboard.send("Windows key + A")
        elif "Open Clipboard" in command:
            talk('Opening clipboard')
            keyboard.send("Windows key + V")
        elif "Open emojis" in command:
            talk('Opening emojis panel')
            keyboard.send("Windows key + ;")
        elif "Open Quick Link menu" in command:
            talk('Opening quick link')
            keyboard.send("Windows key + X")
        elif "Open Search" in command:
            talk('Opening search bar')
            keyboard.send("Windows key + S ")
        elif "View open apps" in command:
            talk('Opening views')
            keyboard.send("Ctrl + Alt + Tab")
        elif "Open Task View" in command:
            talk('Opening task view')
            keyboard.send("Windows key + Tab")
        elif "Open new window" in command:
            talk('Opening new window')
            keyboard.send("Ctrl + N")
        elif "Open notifications" in command:
            talk('Opening notifications')
            keyboard.send("Windows + N")
        elif "Open Properties settings" in command:
            talk('Opening properties settings')
            keyboard.send("Alt + Enter")

        elif 'open chrome' in choice:
            if checkIfProcessRunning('chrome'):
                #var1.set("Application is already running..!")
                text_box2("Application is already running")
            else:
                #var1.set("Opening Chrome..!")
                text_box2('Opening Chrome')
                os.startfile("chrome.exe")

        elif 'open youtube'in choice:
            #var1.set("Opening Youtube..!")
            text_box2("opening youtube")
            webbrowser.open("https://www.youtube.com/")

        elif 'open amazon' in choice:
            #var1.set("Opening Amazon in chrome..!")
            text_box2("Opening Amazon in chrome")
            webbrowser.open('https://www.amazon.in/')


        elif 'open flipkart' in choice:
            var1.set("Opening Flipkart in chrome..!")
            talk("Opening Flipkart in chrome")
            webbrowser.open("https://www.flipkart.com/")


        elif 'open discord' in choice:
            var1.set("Opening Discord in chrome..!")
            talk("Opening Discord in chrome")
            webbrowser.open("https://discord.com/channels/@me")


        elif 'open cricbuzz' in choice:
            var1.set("Opening Cricbuzz in chrome..!")
            talk("Opening Cricbuzz in chrome")
            webbrowser.open("https://www.cricbuzz.com/")

        elif 'open instagram' in choice:
            var1.set("Opening instagram in chrome..!")
            talk("Opening instagram in chrome")
            webbrowser.open("https://www.instagram.com/")

        elif 'open file explorer' in choice:
            if checkIfProcessRunning('explorer "C:\path\of\folder"'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening file explorer..!")
                talk('Opening file explorer')
                subprocess.Popen('explorer "C:\path\of\folder"')

        elif 'open whatsapp' in choice:
            if doesFileExists("C:\\Users\\Ganga Madhukar Piska\\AppData\\Local\\WhatsApp\\Update.exe"):
                if checkIfProcessRunning("Whatsapp.exe"):
                    var1.set("Application is already running..!")
                    talk("Application is already running")
                else:
                    var1.set("Opening Whatsapp..!")
                    talk('Opening Whatsapp')
                    os.system('C:\\Users\\Ganga Madhukar Piska\\AppData\\Local\\WhatsApp\\Update.exe --processStart "Whatsapp.exe"')
            else:
                var1.set("Opening whatsapp in chrome..!")
                talk('Opening Whatsapp in Chrome')
                webbrowser.open("https://web.whatsapp.com/")



        elif 'open telegram' in choice:
            if doesFileExists("C:\\Users\\Ganga Madhukar Piska\\AppData\\Roaming\\Telegram Desktop\\Telegram.exe"):
                if checkIfProcessRunning("C:\\Users\\Ganga Madhukar Piska\\AppData\\Roaming\\Telegram Desktop\\Telegram.exe"):
                    var1.set("Application is already running..!")
                    talk("Application is already running")
                else:
                    var1.set("Opening Telegram..!")
                    talk('Opening Telegram')
                    os.startfile("C:\\Users\\Ganga Madhukar Piska\\AppData\\Roaming\\Telegram Desktop\\Telegram.exe")
            else:
                var1.set("Opening telegram..!")
                talk("Opening telegram")
                webbrowser.open("https://web.telegram.org/z/")


        elif 'open github' in choice:
            if doesFileExists("C:\\Users\\adity\\AppData\\Local\\GitHubDesktop\\app-2.9.4\\GitHubDesktop.exe"):
                if checkIfProcessRunning("GitHubDesktop.exe"):
                    var1.set("Application is already running..!")
                    talk("Application is already running")
                else:
                    var1.set("Opening Github..!")
                    talk('Opening GitHub')
                    os.system(
                        'C:\\Users\\adity\\AppData\\Local\\GitHubDesktop\\app-2.9.4\\GitHubDesktop.exe --processStart "GitHubDesktop.exe" ')
            else:
                var1.set("Opening Github in Chrome..!")
                talk("Opening Github in chrome")
                webbrowser.open("https://github.com/Adi0015")


        elif 'open pycharm' in choice:
            if checkIfProcessRunning('pycharm64.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening pycharm..!")
                talk('Opening Pychram')
                os.startfile("pycharm64.exe")


        elif 'open valorant' in choice:
            if checkIfProcessRunning('RiotClientServices.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening Valorant..!")
                talk('Opening Valorant')
                os.startfile("C:\Riot Games\Riot Client\RiotClientServices.exe")


        elif 'open spotify' in choice:
            if checkIfProcessRunning('spotify.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening Spotify..!")
                talk('Opening spotify')
                os.startfile("spotify.exe")



        elif 'open vs code' in choice:
            if checkIfProcessRunning('code.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening Vs Code..!")
                talk("Opening Vs code")
                os.startfile("C:\\Users\\Ganga Madhukar Piska\\AppData\\Local\\Programs\\Microsoft VS Code\\code.exe")


        elif 'open word' in choice:
            if checkIfProcessRunning('winword.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening MS Word..!")
                talk("Opening Ms word")
                os.startfile("winword.exe")


        elif 'open notepad' in choice:
            if checkIfProcessRunning('notepad.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening Notepad..!")
                talk("Opening notepad")
                os.startfile("notepad.exe")


        elif 'open powerpoint' in choice:
            if checkIfProcessRunning('powerpnt.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening PowerPoint..!")
                talk("Opening powerpoint")
                os.startfile("powerpnt.exe")


        elif 'open microsoft edge' in choice:
            if checkIfProcessRunning('msedge.exe'):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening MS Edge..!")
                talk("Opening msedge")
                os.startfile("msedge.exe")


        elif 'open ms teams' in choice:
            if checkIfProcessRunning("C:/Users/Ganga Madhukar Piska/AppData/Local/Microsoft/Teams/current/Teams.exe"):
                var1.set("Application is already running..!")
                talk("Application is already running")
            else:
                var1.set("Opening MS Teams..!")
                talk("Opening teams")
                os.startfile("C:/Users/Ganga Madhukar Piska/AppData/Local/Microsoft/Teams/current/Teams.exe")


        else:
            var1.set("Cant Open The Required File Or Application..!")
            talk("cant open the required file or application")


    except FileNotFoundError:
        var1.set("Required file does not exists on your device")
        talk("Required file does not exists on your device")
        wake_up()

def close_program(command):
    choice = command
    try:
        if "close chrome" in choice:
            if checkIfProcessRunning('chrome.exe'):
                talk('Closing Chrome')
                os.system('TASKKILL /F /IM chrome.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")

        elif "close valorant" in choice:
            if checkIfProcessRunning('RiotClientServices.exe'):
                talk('Closing valorant')
                os.system('TASKKILL /F /IM RiotClientServices.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")

        elif "close spotify" in choice:
            if checkIfProcessRunning('spotify.exe'):
                talk('Closing spotify')
                os.system('TASKKILL /F /IM spotify.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")

        elif "close vs code" in choice:
            if checkIfProcessRunning('code.exe'):
                var1.set("Closing VS Code..!")
                talk('Closing VS Code')
                os.system('TASKKILL /F /code.exe')

        elif "close word" in choice:
            if checkIfProcessRunning('winword.exe'):
                var1.set("Closing MS Word..!")
                talk('Closing MS Word')
                os.system('TASKKILL /F /IM winword.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")

        elif 'close whatsapp' in choice:
            if doesFileExists("C:\\Users\\adity\\AppData\\Local\\WhatsApp\\Update.exe"):
                if checkIfProcessRunning("Whatsapp.exe"):
                    talk('Closing Whatsapp')
                    os.system('TASKKILL /F /IM Whatsapp.exe')
                else:
                    var1.set("Application is already closed..!")
                    talk("Application is already closed")

        elif 'close github' in choice:
            if doesFileExists("C:\\Users\\adity\\AppData\\Local\\GitHubDesktop\\app-2.9.4\\GitHubDesktop.exe"):
                if checkIfProcessRunning("GitHubDesktop.exe"):
                    talk('Closing GitHub')
                    os.system('TASKKILL /F /IM GitHubDesktop.exe')
                else:
                    var1.set("Application is already closed..!")
                    talk("Application is already closed")

        elif 'close microsoft edge' in choice:
            if checkIfProcessRunning('msedge.exe'):
                talk('Closing Microsoft edge')
                os.system('TASKKILL /F /IM msedge.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")


        elif 'close notepad' in choice:
            if checkIfProcessRunning('notepad.exe'):
                talk('Closing notepad')
                os.system('TASKKILL /F /IM notepad.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")


        elif 'close powerpoint' in choice:
            if checkIfProcessRunning('powerpnt.exe'):
                talk('Closing Powerpoint')
                os.system('TASKKILL /F /IM powerpnt.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")


        elif 'close pycharm' in choice:
            if checkIfProcessRunning('pychram.exe'):
                talk('Closing pychram')
                os.system('TASKKILL /F /IM pychram.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")

        elif 'close telegram' in choice:
            if checkIfProcessRunning('Telegram.exe'):
                talk('Closing Powerpoint')
                os.system('TASKKILL /F /IM Telegram.exe')
            else:
                var1.set("Application is already closed..!")
                talk("Application is already closed")

    except FileNotFoundError:
        talk("Required file does not exists on your device")
        wake_up()


def weather(command):
    command = command
    command = command.replace("how is the weather in ", "")
    command = command.capitalize()
    # base URL
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    CITY = command
    API_KEY = "c07a9fcaab2d950fbcc19fef00a77360"
    # upadting the URL
    URL = BASE_URL + "q=" + CITY + "&appid=" + API_KEY
    # HTTP request
    response = requests.get(URL)
    # checking the status code of the request
    if response.status_code == 200:
        # getting data in the json format
        data = response.json()
        # getting the main dict block
        main = data['main']
        # getting temperature
        temperature = main['temp'] - 273.15
        temperature = int(temperature)
        # getting the humidity
        humidity = main['humidity']
        # getting the pressure
        pressure = main['pressure']
        # weather report
        report = data['weather']

        print(f"Weather Report: {report[0]['description']}")
        print(f"Temperature: {temperature}")
        print(f"Humidity: {humidity}")
        print(f"Pressure: {pressure}")


        talk(f"Weather Report: {report[0]['description']}")
        talk(f"Temperature: {temperature} degree celsius")
        talk(f"Humidity: {humidity} %")
        talk(f"Pressure: {pressure} Hectopascal")
    else:
        # showing the error message
        talk("Error "
              "   try asking how is the weather ")
        run_luna()


def grp_info():
    talk("""SE-4 , mini project group number 25 ,  
            leader :      Ganga piska ,
            teammates : , Aditya gohil,
                          Purva yadav,
                          Ishaan shah,            """)


def time():
    strTime = datetime.datetime.now().strftime("%H:%M:%S")
    talk(f"The time is {strTime}")


def wake_up():
    command=take_command()
    print(command)
    #var.set(f'{command}')
    if "luna" in command:
        text_box1(command)
        text_box2("yes , listening")
        talk("yes , listening")
        print(command)
        run_luna()
    else:
        wake_up()

#while True:
    #wake_up()

def get_main():
        talk('Hii')
        try:
            wake_up()
        except Exception as e:
            print(e)

def show():
        root = Tk()
        root.title("VOICE ASSISTANT GUI")
        root.geometry("350x600")
        root.configure(background='black')
        root.minsize(350, 600)
        root.maxsize(350, 600)
        frameCnt = 12
        frames = [PhotoImage(file='glow_ball.gif', format='gif -index %i' % (i)) for i in range(frameCnt)]

        def update(ind):
            frame = frames[ind]
            ind += 1
            if ind == frameCnt:
                ind = 0
            btn.configure(image=frame, width=290, height=290, command=get_main)

        btn = Button(root)

        root.after(100, update, 0)
        btn.pack()

        global label, var, var1, label1

        var = StringVar()
        var1 = StringVar()
        label = Label(root, textvariable=var, relief=RAISED, width=40, height=5, borderwidth=6, justify=LEFT)
        label1 = Label(root, textvariable=var1, relief=RAISED, width=40, height=5, borderwidth=6, justify=RIGHT)
        label.place(relx=0.0, rely=1.0, anchor='sw')
        label1.place(relx=1.0, rely=1.0, anchor='se')
        var.set("")
        label.pack()
        label1.pack()

        root.mainloop()

if __name__ == '__main__':
        show()