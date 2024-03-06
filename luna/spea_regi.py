import os
import pickle
import numpy as np
from queue import Queue
from tensorflow import keras
from common import SpeechToText    
from spea_verif import Verification
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping

verification =  Verification()
stt =  SpeechToText()

class Registration():

  def __init__(self) :
      self.registerd =False 
      self.trainingData = Queue()
      self.mfccsData = Queue()
      self.i = 0

  def model_Preprocessing(self):
      while  not self.trainingData.empty():
        # print(self.trainingData.get())
        data =b''.join(self.trainingData.get())
        mfccs = verification.compute_MFCCs(frameData=data)
        self.mfccsData.put(mfccs.copy())

  def train_Model(self):
      self.model_Preprocessing()
      home_dir = os.path.expanduser("~")
      self.path = os.path.join(home_dir, ".config", "luna")
      with open(f'{self.path}/data.pkl', 'rb') as f:
        spoofs = pickle.load(f)
      if self.mfccsData.qsize()==6:
        data = []
        for spoof in spoofs:
          data.append((self.mfccsData.get(), np.array([1]).astype(np.float32)))
          data.append((spoof.astype(np.float32), np.array([0]).astype(np.float32)))

        # with open('mfcc_data.pkl', 'wb') as f:
        #   pickle.dump(data, f)
        train_data = data[:8]
        val_data = data[8:]
        train_mfcc, train_labels = zip(*train_data)
        val_mfcc, val_labels = zip(*val_data)
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        max_length = max(len(seq) for seq in train_mfcc + val_mfcc)
        train_mfcc_padded = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in train_mfcc])
        val_mfcc_padded = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in val_mfcc])
        self.saveModel(max_length,train_mfcc_padded,train_labels,val_mfcc_padded,val_labels)

  def saveModel(self,max_length,train_mfcc_padded,train_labels,val_mfcc_padded,val_labels):
      model = Sequential()

      model.add(Dense(64, activation='relu', input_shape=(max_length,)))
      # model.add(Dropout(0.1))  # Adjust input shape according to your data
      model.add(Dense(64, activation='relu'))
      # model.add(Dropout(0.1))
      model.add(Dense(32, activation='relu'))
      # model.add(Dropout(0.1))
      model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
      model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.05),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

      early_stopping = EarlyStopping(monitor='val_loss', patience=10)

      history = model.fit(train_mfcc_padded, train_labels, epochs=100, batch_size=32, validation_data=(val_mfcc_padded, val_labels), callbacks=[early_stopping])
      history.model.fit(train_mfcc_padded, train_labels, epochs=100, batch_size=32, validation_data=(val_mfcc_padded, val_labels), callbacks=[early_stopping])
      if not os.path.exists(f"{self.path}/voiceModel"):
            os.makedirs(f"{self.path}/voiceModel")
      model.save(f"{self.path}/voiceModel/model.keras")

  def train_My_Voice(self):
      while self.i !=6:
        print("i:",self.i)
        stt.record_Microphone()
        stt.offline_STT()
        text = stt.display()
        print(text)
        if text.split()[0] in ["luna","luna.","luna,"]:
          frame_data = b''.join(stt.data)
          result = stt.result
          print(result)
          timestamp = verification.get_Timestamps(result) 
          extractedData = verification.extracted_Frame(frame_data,timestamp[0],timestamp[1])
          self.trainingData.put([extractedData].copy())
          print("size:",self.trainingData.qsize())
          if self.trainingData.qsize() == 6:
              self.train_Model()
          self.i +=1

