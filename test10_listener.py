# region Packages 
import re
import os
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
import pvporcupine
import pandas as pd
from dotenv import main
from datetime import datetime, timedelta
from elevenlabs import set_api_key, generate, play, stream
from google.oauth2.service_account import Credentials
# endregion

# region Vars
# Loaded Env Vars
main.load_dotenv()
THRESHOLD = float(os.getenv("THRESHOLD"))
TIMEOUT_LENGTH = float(os.getenv("TIMEOUT_LENGTH"))
CHANNELS = int(os.getenv("CHANNELS"))
SWIDTH = int(os.getenv("SWIDTH"))
CHUNK = int(os.getenv("CHUNK"))
RATE = int(os.getenv("RATE"))
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

# Som extra vars
short_normalize = (1.0/32768.0)
audio_format = pyaudio.paInt16

# Path for audio and conversations
audio_name_directory = r'.\audio'
transcript_name_directory = r'.\transcripts'
# endregion

# region Class
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
            # options
        )
        transcript_text = transcript["text"]

        if showlog:
            print(f'{"-"*50}\nTranscript: {transcript_text}\n{"-"*50}\n')
        
        return transcript_text

class Audio:
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
        print(f'Listening beginning...\n')

        while True:
            pcm = self.stream.read(CHUNK)
            rms_val = self.rms(pcm)
            if rms_val > THRESHOLD:
                self.record()

    def record(self):
        print(f'Noise detected, recording beginning!\n')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        
        self.writeaudiofile(b''.join(rec))

    def writeaudiofile(self, recording):
        n_files = len(os.listdir(audio_name_directory))
        filename = os.path.join(audio_name_directory, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(audio_format))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print(f'Written to file: {filename}\n')
        print("Attempting to Transcribe Lastest File...\n")
        self.transcribe()

    def transcribe(self):
        n_files = len(os.listdir(audio_name_directory))-1
        audio_filename = os.path.join(audio_name_directory, '{}.wav'.format(n_files))
        print(f'Transcribing: {audio_filename}...\n')

        transcribe_output = gptObj.whispercall(audio_filename)
        
        text_filename = os.path.join(transcript_name_directory, '{}.txt'.format(n_files))
        print(f'Saving Transcript Filename: {text_filename}...\n')
        with open(text_filename, "w") as outfile:
            outfile.write(transcribe_output)
        print('Saved Transcript!\n')
# endregion

# region Main
gptObj = GPT()
audioObj = Audio()

audioObj.listen()
# endregion