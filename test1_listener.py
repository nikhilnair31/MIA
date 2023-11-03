# region Packages 
import os
import wave
import time
import math
import openai
import struct
import pyaudio
import asyncio
import threading
import pvporcupine
import pandas as pd
from dotenv import main

import pinecone
from tqdm.autonotebook import tqdm

from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter 
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
PINECONE_API_KEY = str(os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = str(os.getenv("PINECONE_ENV_KEY"))

short_normalize = (1.0/32768.0)
audio_format = pyaudio.paInt16

audio_name_directory = r'.\audio'
docs_name_directory = r'.\docs'
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
        files = os.listdir(docs_name_directory)
        for file in files:
            file_path = os.path.join(docs_name_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

class Vector:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 0

        self.embeddings = OpenAIEmbeddings()
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_ENV  # next to api key in console
        )
        self.index = pinecone.Index("mia")

    def load_text(self, text_filename):
        loader = TextLoader(text_filename)
        documents = loader.load()
        print (f'You have {len(documents)} document(s) in your data')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size, 
            chunk_overlap = self.chunk_overlap
        )
        documents = text_splitter.split_documents(documents)
        print(f'Doc Length: {len(documents)}')

        Pinecone.from_documents(documents, self.embeddings, index_name="mia")
        print(f'Upserted doc!\n')

    
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
    def __init__(self):
        self.max_record_time = 11 * 60

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
        elapsed_time = 0
        start_time = time.time()
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)

            # Check if 10 minutes have elapsed
            elapsed_time = current - start_time
            if elapsed_time >= self.max_record_time:
                start_time = current
                break
        
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
            outfile.write(transcribe_output)
        print(f'Saved Transcript! : {filename}\n')

        vecObj.load_text(filename)
# endregion

# region Main
genObj = General()
vecObj = Vector()
gptObj = GPT()
audioObj = Audio()

audioObj.listen()
# endregion