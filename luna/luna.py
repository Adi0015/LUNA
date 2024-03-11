from common import SpeechToText
from spea_regi import Registration
from threading import Thread

stt = SpeechToText()
registration = Registration()

def luna():
  while True:
    # stt.PASS.put(True)
    print("yes?")
    rec = Thread(target=stt.record_Microphone())
    offline_dtt = Thread(target=stt.offline_STT())
    tta =Thread(target=textToAction())
    rec.start()
    offline_dtt.start()
    tta.start()
    # tta.join()

def textToAction():
    text = stt.display()
    # print(text)
    if text != '' and text.split()[0] in ["luna", "luna.","luna,"]: 
        stt.verified()
        print(text)
        if 'register' in text :
          registration.train_My_Voice()
        else: pass

if __name__ == "__main__":
    luna()
