import os
import ssl
import socket
import numpy as np
import speech_recognition as sr 
from queue import Queue ,	LifoQueue
import whisper_timestamped as whisper

ssl._create_default_https_context = ssl._create_unverified_context

class SpeechToText:
		
		def __init__(self):
			self.AUDIO_QUEUE = Queue()
			# self.PASS = Queue(maxsize=1)
			self.TEXT =	Queue(maxsize=1)
			# self.EXTRACTED_DATA = Queue(maxsize=1)
			self.LISTENER = sr.Recognizer()
			self.CHANNELS = 1
			self.FRAME_RATE = 16000
			self.SAMPLE_SIZE = 2
			self.CHUNKS = 1024
			self.load_whisper_model()
		

		def addNewAudio(self,data):
			while not self.AUDIO_QUEUE.empty():
					self.AUDIO_QUEUE.get()
			self.AUDIO_QUEUE.put(data)
				

		def record_Microphone(self):
			# while not self.PASS.empty():
				with sr.Microphone(sample_rate=self.FRAME_RATE,chunk_size=self.CHUNKS) as source:
						frames = []
						self.LISTENER.energy_threshold = 500
						self.LISTENER.adjust_for_ambient_noise(source, duration=0.5)
						print("->")
						audio = self.LISTENER.listen(source,)
						data = audio.frame_data
						frames.append(data)
						SpeechToText.addNewAudio(self,frames.copy())
						frames=[]
						# self.PASS.get()


		def check_internet(self):
				try:
						socket.create_connection(("www.google.com", 80))
						return True
				except OSError:
						return False
				

		def load_whisper_model(self):
				home_dir = os.path.expanduser("~")
				download_root = os.path.join(home_dir, ".config", "luna", "whisperModel")
				self.model = whisper.load_model("small.en", device="cpu", download_root=download_root)
		

		def offline_STT(self):
				while not self.AUDIO_QUEUE.empty():
					self.data = self.AUDIO_QUEUE.get()
					frames = b''.join(self.data)
					np_data = np.frombuffer(frames,dtype=np.int16, count=len(frames)//2, offset=0)
					np_data	= np_data.astype(np.float32, order='C') / 32768.0
					self.result = whisper.transcribe(self.model,np_data, language="en")
					text = self.result["text"]
					self.TEXT.put(text.lower())
					# SpeechToText.checkLuna(data,text,result)


		def display(self):
			if not self.TEXT.empty():
					return(self.TEXT.get())
		

		# def checkLuna(self,data,text,result):
		# 	if text != None and text.split()[0] in	["luna","luna.","luna,"]:
		# 		segments = result.get("segments", [])
		# 		for segment in segments:
		# 				words = segment.get("words", [])
		# 				for i, word in enumerate(words):
		# 						if "luna" in word["text"].lower():
		# 								self.start_time = word["start"]
		# 								self.end_time = words[i + 1]["start"] if i + 1 < len(words) else segment["end"]
		# 								if self.EXTRACTED_DATA.empty():
		# 									SpeechToText.lunaExtraction(data)
											

		# def lunaExtraction(self,data):
		# 		start_index = int(self.start_time*(self.FRAME_RATE)*(self.SAMPLE_SIZE))
		# 		end_index = int((self.end_time+0.1)*(self.FRAME_RATE)*(self.SAMPLE_SIZE))
		# 		if (start_index-end_index)%2 != 0:
		# 			end_index += 1
		# 		# Extract audio data within the specified time range
		# 		extractedFrameData =[]
		# 		extractedFrameData.append(data[start_index:end_index])
		# 		self.EXTRACTED_DATA.put(extractedFrameData)


		# def online_STT(self):
		# 		while not self.AUDIO_QUEUE.empty():
		# 			try: 
		# 					data = self.AUDIO_QUEUE.get()
		# 					frames = b''.join(data)
		# 					self.LISTENER.energy_threshold = 300
		# 					audio = sr.AudioData(frame_data=frames,sample_rate=self.FRAME_RATE,sample_width=self.SAMPLE_SIZE)
		# 					text = self.LISTENER.recognize_google(audio,language='en-IN')
		# 					self.TEXT.put(text.lower())        
		# 			except sr.UnknownValueError :pass
