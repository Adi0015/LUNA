import librosa
import numpy as np 
import  os
# from common import SpeechToText
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# stt = SpeechToText()
scaler = MinMaxScaler()

class Verification():

  def __init__(self) :
    # self.verifiedStatus =  False
    self.n_mfcc=20  
    self.path = os.path.join(os.path.expanduser("~"), ".config", "luna")
    self.FRAME_RATE = 16000
    self.SAMPLE_SIZE = 2
    self.load_Keras_Model()
  
  def load_Keras_Model(self):
      if not os.path.exists(f"{self.path}/voiceModel/model.keras"):
         self.verifiedStatus = False
         self.kerasModel = ''
         print("you have not registered ur voice")
      else : self.kerasModel=load_model(f"{self.path}/voiceModel/model.keras")
     


  def get_Timestamps(self,result):
    segments = result.get("segments", [])
    for segment in segments:
        words = segment.get("words", [])
        for word in words:
            if "luna" in word["text"].lower():
                start_time = word["start"]
                end_time = word["end"]
                return float(start_time), float(end_time)
            else: return float(0),float(0)

  def extracted_Frame(self,audioframeData,start_time,end_time):
      start_index = int(start_time*self.FRAME_RATE*self.SAMPLE_SIZE)
      end_index = int((end_time+0.1)*self.FRAME_RATE*self.SAMPLE_SIZE)
      if (start_index-end_index)%2 != 0:
        end_index += 1
      extractedFrameData = audioframeData[start_index:end_index]
      # Extract audio data within the specified time range
      return extractedFrameData

  def normalize_Data(self,mfcc):
    # mfcc_data_reshaped = mfcc.reshape(-1, 1)
    mfcc_data_normalized = scaler.fit_transform(mfcc)
    mfccs= mfcc_data_normalized.flatten()
    return mfccs
  
  def compute_MFCCs(self,frameData):
      frameData = np.frombuffer(frameData,dtype=np.int16)
      data = frameData.astype(np.float32) / np.iinfo(np.int16).max
      mfccs = librosa.feature.mfcc(y=data,sr=self.FRAME_RATE,n_mfcc=self.n_mfcc)
      mfccs_flat = self.normalize_Data(mfcc=mfccs)
      return mfccs_flat

  def get_verified(self,data):
    if self.kerasModel != '':
       prediction = self.kerasModel.predict(data)
       print(prediction)
       result = (prediction > 0.5).astype(int).reshape(-1,)
       if result == 1: self.verifiedStatus=True
       else : self.verifiedStatus=False
    

