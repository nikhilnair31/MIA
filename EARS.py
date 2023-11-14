# region Packages 
import os
import io
import wave
import time
import math
import openai
import struct
import pyaudio
import librosa
import asyncio
import keyboard
import threading
import pvporcupine
import numpy as np
import pandas as pd
from dotenv import main
from pynput import keyboard
from datetime import datetime

import pinecone
from tqdm.autonotebook import tqdm

from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter 
# endregion

# region Vars
main.load_dotenv()

# DB Related Vars
INDEX_NAME = os.getenv("INDEX_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
SPEECH_GAP_DELAY = int(os.getenv("SPEECH_GAP_DELAY"))

# Audio Related Vars
MAX_REC_TIME = int(os.getenv("MAX_REC_TIME"))
THRESHOLD = float(os.getenv("THRESHOLD"))
TIMEOUT_LENGTH = float(os.getenv("TIMEOUT_LENGTH"))
CHANNELS = int(os.getenv("CHANNELS"))
SWIDTH = int(os.getenv("SWIDTH"))
CHUNK = int(os.getenv("CHUNK"))
RATE = int(os.getenv("RATE"))
short_normalize = (1.0/32768.0)
audio_format = pyaudio.paInt16
PAUSED = False

# API keys
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))
PINECONE_API_KEY = str(os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = str(os.getenv("PINECONE_ENV_KEY"))

# Folder paths
audio_name_directory = r'.\audio'
docs_name_directory = r'.\docs'
# endregion

# region Class
class EARS():
    # Setup objects and APIs
    def __init__(self):
        # Audio Setup
        self.max_record_time = MAX_REC_TIME * 60
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=audio_format,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        # OpenAI Setup
        openai.api_key = OPENAI_API_KEY

        # Vector DB Setup
        # Embeddings
        self.embeddings = OpenAIEmbeddings()
        # DB 
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index(INDEX_NAME)

        # Clean up folders
        if(len(os.listdir(docs_name_directory)) > 0):
            self.load_text()
        self.clearfolders()

        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()

    # region General
    def clearfolders(self):
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

        print(f'Cleared Folders!\n')
    
    def on_press(self, key):
        global PAUSED
        try:
            k = key.char 
        except:
            k = key.name
        if k == 'r':
            PAUSED = not PAUSED
            print(f'EARS has been {"PAUSED" if PAUSED else "UNPAUSED"}\n')

    def on_release(self, key):
        if key == keyboard.Key.esc:      
            return False

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
    # endregion

    # region OpenAI
    def whispercall(self, filename):
        print('Making Whisper request..\n')

        audio_file= open(filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en",
            prompt="don't translate or make up words to fill in the rest of the sentence. if background noise return ."
        )
        transcript_text = transcript["text"]
        print(f'{"="*50}\nTranscript: {transcript_text}\n{"="*50}\n')
        
        return transcript_text
    
    def gpt_chat_call(self, text, model="gpt-3.5-turbo", temp=0.7, maxtokens=512):
        print('Making ChatGPT request..\n')

        response = openai.ChatCompletion.create(
            model=model,
            messages=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0]["message"]["content"].lower()
        print(f'{"-"*50}\nChatGPT Response:\n{response_text}\n{"-"*50}\n')
        
        return response_text
    # endregion

    # region Audio
    def start_listening(self):
        print(f'EARS is Listening...\n')

        global PAUSED
        self.last_audio_time = time.time()
        self.elapsed_time = 0

        while True:
            pcm = self.stream.read(CHUNK)
            rms_val = self.rms(pcm)
            if rms_val > THRESHOLD and not PAUSED:
                asyncio.run(self.record())

    async def record(self):
        print(f'Recording!\n')

        global PAUSED
        rec = []
        elapsed_time = 0
        start_time = time.time()
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end and not PAUSED:

            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)

            # Check if max_record_time has elapsed
            elapsed_time = current - start_time
            if elapsed_time >= self.max_record_time:
                start_time = current
                break
        
        recording = b''.join(rec)
        timestamp = time.strftime('%Y%m%d%H%M%S')

        # Create new thread to save/transcribe audio file
        writeaudiofile_thread = threading.Thread(target=self.writeaudiofile, args=(timestamp, recording))
        writeaudiofile_thread.start()
    
    def writeaudiofile(self, timestamp, recording):
        filename = os.path.join(audio_name_directory, '{}.wav'.format(timestamp))

        rms_values = [self.rms(frame) for frame in [recording[i:i+CHUNK] for i in range(0, len(recording), CHUNK)]]
        silent_frames = [i for i, rms in enumerate(rms_values) if rms < THRESHOLD]
        if silent_frames:
            getKey = lambda item: item[0] if item[1] >= THRESHOLD else -1
            max_valued_item_index = max(enumerate(rms_values), key=getKey)[0]
            possible_trimmed_audio = recording[: (max_valued_item_index + 1) * CHUNK]
            
            if len(possible_trimmed_audio) >= RATE * 2:
                trimmed_audio = possible_trimmed_audio
            else:
                trimmed_audio = recording
        else:
            trimmed_audio = recording

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(audio_format))
        wf.setframerate(RATE)
        wf.writeframes(trimmed_audio)
        wf.close()
        print(f'Saved Recording! : {filename}\n')

        self.transcribe(timestamp, filename)

    def transcribe(self, timestamp, filename):
        print('Transcribing...\n')

        system_prompt = f"""
            You will receive a user's transcribed speech and are to process it to correct potential errors. 
            DO NOT DO THE FOLLOWING:
            - Generate any additional content 
            - Censor any of the content
            - Print repetitive content
            DO THE FOLLOWING:
            - Account for transcript include speech of multiple users
            - Only output corrected text 
            - If too much of the content seems erronous return '.' 
            Transcription: 
        """

        transcribe_output = self.whispercall(filename)
        clean_transcript = self.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": system_prompt + transcribe_output
            }], 
            model='gpt-4', temp=0
        )
        
        if clean_transcript != '.' and clean_transcript != '"."':
            filename = os.path.join(docs_name_directory, f'{timestamp}.txt')
            with open(filename, "w") as outfile:
                outfile.write(clean_transcript)
            print(f'Saved Transcript! : {filename}\n')

            vectorize_thread = threading.Thread(target=self.filecheck, args=(timestamp, ))
            vectorize_thread.start()
    # endregion
        
    # region Vector DB
    def filecheck(self, timestamp):
        latestfilename = os.path.join(docs_name_directory, f'{timestamp}.txt')
        all_files = os.listdir(docs_name_directory)
        all_files.sort(key=lambda x: os.path.getctime(os.path.join(docs_name_directory, x)), reverse=True)

        lastfilename = None
        if len(all_files) > 2:
            lastfilename = all_files[1]
        else:
            print("No files found in the folder.\n")

        if lastfilename is None:
            print("No files found in the folder.\n")
        else:
            first_file_creation_time = datetime.fromtimestamp(os.path.getctime(os.path.join(docs_name_directory, lastfilename)))
            specific_file_creation_time = datetime.fromtimestamp(os.path.getctime(latestfilename))
            time_gap = int((specific_file_creation_time - first_file_creation_time).total_seconds())
            print(f"Time gap from '{lastfilename}' to '{latestfilename}': {time_gap}\n")
            
            if time_gap > SPEECH_GAP_DELAY:
                self.load_text()
    
    def load_text(self):
        all_files = sorted(os.listdir(docs_name_directory), key = lambda x: os.path.getctime(os.path.join(docs_name_directory, x)))
        
        combined_content = ""
        for filename in all_files:
            filepath = os.path.join(docs_name_directory, filename)
            with open(filepath, 'r') as file:
                file_content = file.read()
                print(f'file_content: {file_content[:50]} ...')
                combined_content += file_content + " "
        
        first_file_ctime =  time.ctime(os.path.getctime(os.path.join(docs_name_directory, all_files[0])))
        facts_from_combined_content = self.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": f"""
                    You will receive transcribed speech from the environment and are to extract relevant facts from it. 
                    DO THE FOLLOWING:
                    - Extract a single statement about factual information from the content
                    - Account for transcript to be from various sources like the user, surrrounding people, video playback in vicinity etc.
                    - Only output factual text 
                    DO NOT DO THE FOLLOWING:
                    - Generate bullet points
                    - Generate any additional content
                    - Censor any of the content
                    - Print repetitive content
                    Content: 
                    {combined_content}
                """
            }], 
            model='gpt-4', 
            temp=0.5
        )

        temp_file_path = os.path.join(docs_name_directory, r'-1.txt')
        with open(temp_file_path, 'wb') as outfile:
            outfile.write(facts_from_combined_content.encode())
            print(f'Writing: {facts_from_combined_content}')

        text_splitter = RecursiveCharacterTextSplitter()
        text_loader = TextLoader(temp_file_path)
        docs = text_splitter.split_documents(text_loader.load())
        print(f'Loaded data from {len(docs)}')

        if len(docs) > 0:
            # Prepare texts and metadatas
            texts = [d.page_content for d in docs]
            metadatas = [{"file_ctime": first_file_ctime} for d in docs]

            # Inserting to index
            Pinecone.from_texts(texts, self.embeddings, index_name=INDEX_NAME, metadatas=metadatas)
            # Pinecone.from_documents(docs, self.embeddings, index_name=INDEX_NAME)
            print(f'Upserted doc!\n')

            self.clearfolders()
        else:
            print(f'Nothing to upsert!\n')
    # endregion
# endregion

ears = EARS()
ears.start_listening()