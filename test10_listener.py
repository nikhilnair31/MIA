# region Packages 
import re
import os
import io
import csv
import wave
import time
import json
import math
import boto3
import openai
import struct
import whisper
import gspread
import pyaudio
import asyncio
import aiofiles
import threading
import pvporcupine
import pandas as pd
from dotenv import main
from datetime import datetime, timedelta
from elevenlabs import set_api_key, generate, play, stream
from google.oauth2.service_account import Credentials
# endregion

# region Vars
main.load_dotenv()
THRESHOLD = float(os.getenv("THRESHOLD"))
TIMEOUT_LENGTH = float(os.getenv("TIMEOUT_LENGTH"))
CHANNELS = int(os.getenv("CHANNELS"))
SWIDTH = int(os.getenv("SWIDTH"))
CHUNK = int(os.getenv("CHUNK"))
RATE = int(os.getenv("RATE"))
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

short_normalize = (1.0/32768.0)
audio_format = pyaudio.paInt16

audio_name_directory = r'.\audio'
transcript_name_directory = r'.\transcripts'
# endregion

# region Class
class General:
    def __init__(self):
        # Delete all recordings
        files = os.listdir(audio_name_directory)
        for file in files:
            file_path = os.path.join(audio_name_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Delete all transcripts
        files = os.listdir(transcript_name_directory)
        for file in files:
            file_path = os.path.join(transcript_name_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
class GPT:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
    
    def whispercall(self, filename, showlog=True):
        print('Making Whisper request..\n')

        audio_file= open(filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en"
        )
        transcript_text = transcript["text"]

        if showlog:
            print(f'{"="*50}\nTranscript: {transcript_text}\n{"="*50}\n')
        
        return transcript_text

class Audio:
    # Set up recorder
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=audio_format,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
    
    @staticmethod
    def rms(frame):
        count = len(frame) / SWIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * short_normalize
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def listen(self):
        print(f'Listening...\n')

        self.last_audio_time = time.time()
        self.elapsed_time = 0
        while True:
            pcm = self.stream.read(CHUNK)
            rms_val = self.rms(pcm)
            if rms_val > THRESHOLD:
                asyncio.run(self.record())

    async def record(self):
        print(f'Recording!\n')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        
        recording = b''.join(rec)
        timestamp = time.strftime('%Y%m%d%H%M%S')

        writeaudiofile_thread = threading.Thread(target=self.writeaudiofile, args=(timestamp, recording))
        writeaudiofile_thread.start()

    def writeaudiofile(self, timestamp, recording):
        filename = os.path.join(audio_name_directory, '{}.wav'.format(timestamp))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(audio_format))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()

        print(f'Saved Recording! : {filename}\n')
        self.transcribe(timestamp, filename)
        
    def transcribe(self, timestamp, filename):
        print('Transcribing...\n')

        transcribe_output = gptObj.whispercall(filename)
        
        filename = os.path.join(docs_name_directory, f'{timestamp}.txt')

        with open(filename, "w") as outfile:
            outfile.write(content)
        
        print(f'Saved Transcript! : {filename}\n')
# endregion

# region Main
genObj = General()
gptObj = GPT()
audioObj = Audio()

audioObj.listen()
# endregion