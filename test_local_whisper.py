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
import warnings
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

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword*")
# endregion

# region Vars
main.load_dotenv()

# DB Related Vars
INDEX_NAME = os.getenv("INDEX_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
SPEECH_GAP_DELAY_IN_SEC = int(os.getenv("SPEECH_GAP_DELAY_IN_SEC"))

# EARS - Audio Related Vars
MAX_AUDIO_REC_TIME_IN_MIN = int(os.getenv("MAX_AUDIO_REC_TIME_IN_MIN"))
EARS_THRESHOLD = float(os.getenv("EARS_THRESHOLD"))
EARS_TIMEOUT_LENGTH = float(os.getenv("EARS_TIMEOUT_LENGTH"))
EARS_CHANNELS = int(os.getenv("EARS_CHANNELS"))
EARS_SWIDTH = int(os.getenv("EARS_SWIDTH"))
EARS_CHUNK = int(os.getenv("EARS_CHUNK"))
EARS_RATE = int(os.getenv("EARS_RATE"))
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

# System Prompts
whisper_prompt = f"""
    don't translate or make up words to fill in the rest of the sentence. if background noise return .
"""
transcribe_system_prompt = f"""
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
facts_system_prompt = f"""
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
"""
# endregion

# region Class
class EARS():
    # Setup objects and APIs
    def __init__(self):
        # Audio Setup
        self.max_record_time = MAX_AUDIO_REC_TIME_IN_MIN * 60
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=audio_format,
            channels=EARS_CHANNELS,
            rate=EARS_RATE,
            input=True,
            frames_per_buffer=EARS_CHUNK
        )

        # OpenAI Setup
        openai.api_key = OPENAI_API_KEY

        # Vector DB Setup
        # Embeddings
        self.embeddings = OpenAIEmbeddings()
        # DB 
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index(INDEX_NAME)

        # If transcript files exist then load them into vector DB first
        if(len(os.listdir(docs_name_directory)) > 0):
            self.load_text()

        # Start a listener for pause/start of EARS
        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "distil-whisper/distil-medium.en"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model = model.to_bettertransformer()
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

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
        count = len(frame) / EARS_SWIDTH
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
        
        transcript = self.pipe(filename)
        transcript_text = transcript["text"]
        print(f'{"="*50}\nTranscript: {transcript_text}\n{"="*50}\n')
        
        return transcript_text
    
    def gpt_chat_call(self, text, model="gpt-4", temp=0.0, maxtokens=512):
        print(f'Making {("GPT-4" if model=="gpt-4" else "ChatGPT")} request..\n')

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
            pcm = self.stream.read(EARS_CHUNK)
            rms_val = self.rms(pcm)
            if rms_val > EARS_THRESHOLD and not PAUSED:
                asyncio.run(self.record())

    async def record(self):
        print(f'Recording!\n')

        global PAUSED
        rec = []
        elapsed_time = 0
        start_time = time.time()
        current = time.time()
        end = time.time() + EARS_TIMEOUT_LENGTH

        while current <= end and not PAUSED:

            data = self.stream.read(EARS_CHUNK)
            if self.rms(data) >= EARS_THRESHOLD: end = time.time() + EARS_TIMEOUT_LENGTH

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
        # Get RMS values of all frames in audio
        rms_values = [self.rms(frame) for frame in [recording[i:i+EARS_CHUNK] for i in range(0, len(recording), EARS_CHUNK)]]

        # Find all silent frames (RMS < threshold) to then trim audio
        silent_frames = [i for i, rms in enumerate(rms_values) if rms < EARS_THRESHOLD]
        if silent_frames:
            getKey = lambda item: item[0] if item[1] >= EARS_THRESHOLD else -1
            max_valued_item_index = max(enumerate(rms_values), key=getKey)[0]
            possible_trimmed_audio = recording[: (max_valued_item_index + 1) * EARS_CHUNK]
            
            if len(possible_trimmed_audio) >= EARS_RATE * 2:
                trimmed_audio = possible_trimmed_audio
            else:
                trimmed_audio = recording
        else:
            trimmed_audio = recording

        # Write audio data to file
        audio_filename = os.path.join(audio_name_directory, '{}.wav'.format(timestamp))
        wf = wave.open(audio_filename, 'wb')
        wf.setnchannels(EARS_CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(audio_format))
        wf.setframerate(EARS_RATE)
        wf.writeframes(trimmed_audio)
        wf.close()
        print(f'Saved Recording! : {audio_filename}\n')

        # Transcribe audio file saved above
        self.transcribe(timestamp, audio_filename)

    def transcribe(self, timestamp, audio_filename):
        print('Transcribing...\n')

        # Get transcribe of audio file and clean it
        transcribe_output = self.whispercall(audio_filename)
        clean_transcript = self.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": transcribe_system_prompt + transcribe_output
            }], 
            model='gpt-4', temp=0
        )
        
        # If errors in transcription then may return punctuation is various quotes so skip those
        if clean_transcript != '.' and clean_transcript != '"."':
            # Save cleaned transcribe into new file
            transcript_filename = os.path.join(docs_name_directory, f'{timestamp}.txt')
            with open(transcript_filename, "w") as outfile:
                outfile.write(clean_transcript)
            print(f'Saved Transcript! : {transcript_filename}\n')

            # Create new thread to check if transcripts are part of same conversation
            vectorize_thread = threading.Thread(target=self.filecheck, args=(timestamp, ))
            vectorize_thread.start()
    # endregion
        
    # region Vector DB
    def filecheck(self, timestamp):
        # Getting path of latest file
        latestfilename = os.path.join(docs_name_directory, f'{timestamp}.txt')
        # Getting all trasncript files sorted by descding (most recent first)
        all_files = os.listdir(docs_name_directory)
        all_files.sort(key=lambda x: os.path.getctime(os.path.join(docs_name_directory, x)), reverse=True)

        # If more than 2 files in transcipt folder, taking index 1 since index 0 is temp file with combined transcripts
        lastfilename = None
        if len(all_files) > 2:
            lastfilename = all_files[1]
        else:
            print("No files found in the folder.\n")

        # Calculating time gap to check if transcripts are part of same conversation
        if lastfilename is None:
            print("No files found in the folder.\n")
        else:
            first_file_creation_time = datetime.fromtimestamp(os.path.getctime(os.path.join(docs_name_directory, lastfilename)))
            specific_file_creation_time = datetime.fromtimestamp(os.path.getctime(latestfilename))
            time_gap = int((specific_file_creation_time - first_file_creation_time).total_seconds())
            print(f"Time gap from '{lastfilename}' to '{latestfilename}': {time_gap}\n")
            
            # If not then upsert to vector DB
            if time_gap > SPEECH_GAP_DELAY_IN_SEC:
                self.load_text()
    
    def load_text(self):
        # Get all transcript files in ascending order of creation time
        all_files = sorted(os.listdir(docs_name_directory), key = lambda x: os.path.getctime(os.path.join(docs_name_directory, x)))
        # Get time of creation for 1st transcript file
        first_file_ctime =  time.ctime(os.path.getctime(os.path.join(docs_name_directory, all_files[0])))
        
        # Append contents of all transcript files into one variable
        combined_content = ""
        for filename in all_files:
            filepath = os.path.join(docs_name_directory, filename)
            with open(filepath, 'r') as file:
                file_content = file.read()
                print(f'file_content: {file_content[:50]} ...')
                combined_content += file_content + " "
        
        # Pull factual information from all transcripts
        facts_from_combined_content = self.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": facts_system_prompt + combined_content
            }], 
            model='gpt-4', temp=0.5
        )

        # Write above info into a temp file
        temp_file_path = os.path.join(docs_name_directory, r'-1.txt')
        with open(temp_file_path, 'wb') as outfile:
            outfile.write(facts_from_combined_content.encode())
            print(f'Writing: {facts_from_combined_content}')

        # Split and create Docs and load
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
            print(f'Upserted doc!\n')
        else:
            print(f'Nothing to upsert!\n')

        # Clean up folders
        self.clearfolders()
    # endregion
# endregion

ears = EARS()
ears.start_listening()