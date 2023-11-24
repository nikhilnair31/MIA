# region Packages 
import os
import wave
import time
import json
import math
import openai
import struct
import whisper
import pyaudio
import asyncio
import aiofiles
import pinecone
import warnings
import pvporcupine
import pandas as pd
from dotenv import main

from elevenlabs import set_api_key, generate, play, stream

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword*")
# endregion

# region Vars
main.load_dotenv()

# MIA Flags
SPEAK_FLAG = False
MIA_FIRST_RUN = True

# MIA - Audio Related Vars
MIA_SWIDTH = int(os.getenv("MIA_SWIDTH"))
MIA_CHANNELS = int(os.getenv("MIA_CHANNELS"))
MIA_THRESHOLD = float(os.getenv("MIA_THRESHOLD"))
MIA_TIMEOUT_LENGTH = float(os.getenv("MIA_TIMEOUT_LENGTH"))
MIA_DEEP_SLEEP_IN_SEC = float(os.getenv("MIA_DEEP_SLEEP_IN_SEC"))

# API Keys
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))
PORCUPINE_API_KEY = os.getenv("PORCUPINE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PINECONE_API_KEY = str(os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = str(os.getenv("PINECONE_ENV_KEY"))

# Extra Vars
short_normalize = (1.0/32768.0)
audio_format = pyaudio.paInt16

# Path for audio and conversations
audio_name_directory = r'.\audio'
convo_name_directory = r'.\conversations'

# Conversations
mia_system_context = """
Your name is MIA and you're an AI companion of the user. Keep your responses short. This is your first boot up and your first interaction with the user so ensure that you ask details about them to remember for the future. This includes things like their name, job/university, residence etc. Ask anything about them until you think it's enough or they stop you.
Internally you have the personality of JARVIS and Chandler Bing combined. You tend to make sarcastic jokes and observations. Do not patronize the user but adapt to how they behave with you.
You help the user with all their requests, questions and tasks. Be honest and admit if you don't know something when asked. 
"""
mia_summarize_context = f"""
    The following is your context of the previous conversation with the user:
"""
whisper_prompt = f"""
    don't translate or make up words to fill in the rest of the sentence. if background noise return .
"""
# endregion

# region Class
class GPT:
    def __init__(self):
        # OpenAI Setup
        openai.api_key = OPENAI_API_KEY

    def gpt_chat_call(self, text, model="gpt-4", temp=1, maxtokens=512):
        print(f'Making {("GPT-4" if model=="gpt-4" else "ChatGPT")} request..\n')
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0]["message"]["content"].lower()
        
        # Check flag to see if MIA should speak
        global SPEAK_FLAG
        if SPEAK_FLAG: speak(response_text)
        
        return response_text

    def whispercall(self, filename):
        print('Making Whisper request..\n')

        audio_file= open(filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language = "en",
            prompt = whisper_prompt
        )
        transcript_text = transcript["text"]
        
        return transcript_text

class Audio:
    def __init__(self):
        # 11Labs Setup
        set_api_key(ELEVENLABS_API_KEY)
        
        self.elapsed_time = 0
        self.last_audio_time = time.time()
        self.require_hotword_mia_sleeping = False
        self.mia_deepsleep = False

        # Set up hotword detector
        self.porcupine = pvporcupine.create(
            access_key=PORCUPINE_API_KEY,
            keyword_paths=[r'other\hey-mia_en_windows_v3_0_0.ppn']
        )

        # Set up recorder
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=audio_format,
            channels=MIA_CHANNELS,
            rate=self.porcupine.sample_rate,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

        # Delete all recordings
        files = os.listdir(audio_name_directory)
        for file in files:
            file_path = os.path.join(audio_name_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    @staticmethod
    def rms(frame):
        count = len(frame) / MIA_SWIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * short_normalize
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def listen(self):
        print(f'Started Listening...\n')

        self.elapsed_time = 0
        self.last_audio_time = time.time()

        while True:
            pcm = self.stream.read(self.porcupine.frame_length)

            global MIA_FIRST_RUN
            if MIA_FIRST_RUN:
                print(f'Booting Up...\n')
                MIA_FIRST_RUN = False
                convObj.conversation_context = [{
                    "role": "system", "content": mia_system_context
                }]
                self.response(model='gpt-4')

            if self.require_hotword_mia_sleeping:
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    print("Hotword Detected!\n")
                    self.require_hotword_mia_sleeping = False
                    self.mia_deepsleep = False
                    self.record()
                else:
                    self.elapsed_time = time.time() - self.last_audio_time
                    
                    if self.elapsed_time > MIA_DEEP_SLEEP_IN_SEC:
                        self.mia_deepsleep = True
                        mia_thoughts = threading.Thread(target=taskObj.thoughts, args=())
                        mia_thoughts.start()

            else:
                rms_val = self.rms(pcm)
                if rms_val > MIA_THRESHOLD:
                    self.record()
                else:
                    self.elapsed_time = time.time() - self.last_audio_time

                    if self.elapsed_time >= MIA_TIMEOUT_LENGTH:
                        print("Sleeping zzz...\n")
                        self.require_hotword_mia_sleeping = True

    def record(self):
        print(f'Recording!\n')
        rec = []
        current = time.time()
        end = time.time() + MIA_TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(self.porcupine.frame_length)
            if self.rms(data) >= MIA_THRESHOLD: 
                end = time.time() + MIA_TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        
        recording = b''.join(rec)
        timestamp = time.strftime('%Y%m%d%H%M%S')
        self.writeaudiofile(timestamp, recording)

    def writeaudiofile(self, timestamp, recording):
        filename = os.path.join(audio_name_directory, '{}.wav'.format(timestamp))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(MIA_CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(audio_format))
        wf.setframerate(self.porcupine.sample_rate)
        wf.writeframes(recording)
        wf.close()
        print(f'Written to file: {filename}\n')

        self.transcribe(filename)

    def transcribe(self, filename):
        print('Transcribing...\n')

        transcribe_output = gptObj.whispercall(filename)
        print(f'{"-"*50}\nTranscript: {transcribe_output}\n{"-"*50}\n')

        if transcribe_output == '' or transcribe_output=='.':
            convObj.conversation_context.append({"role": "assistant", "content": "Sorry could you repeat that?"})
            asyncio.run(convObj.saveconversation())

            self.listen()
        else:
            convObj.conversation_context.append({"role": "user", "content": transcribe_output})
            asyncio.run(convObj.saveconversation())
            
            taskObj.whatisuserasking(transcribe_output)

    def response(self, model='gpt-3.5-turbo', continueconv = True):
        print('Responding...\n')

        gptresponse = gptObj.gpt_chat_call(text = convObj.conversation_context, model=model)
        print(f"{'-'*50}\n{('GPT-4' if model=='gpt-4' else 'ChatGPT')} Response:\n{gptresponse}\n{'-'*50}\n")
        
        convObj.conversation_context.append({"role": "assistant", "content": gptresponse})
        asyncio.run(convObj.saveconversation())

        if continueconv: self.listen()

    def speak(self, text):
        audio = generate(
            text=response_text,
            voice="Bella", model="eleven_monolingual_v1",
            stream=True
        )
        stream(audio)

class Conv:
    def __init__(self):
        # Check to see if file with past conversations exists
        filename = os.path.join(convo_name_directory, 'convo.json')
        try:
            print(f'Found Conversation File...\n')
            
            # If exists then it's NOT MIA's first conversation
            global MIA_FIRST_RUN
            MIA_FIRST_RUN = False

            # Pull data from past conversations
            past_conversations = json.load(open(filename))
            
            # Get MIA's reply based on her system context and your prior conversations
            past_convo_summary = gptObj.gpt_chat_call(
                text = [{
                    "role": "system", 
                    "content": mia_system_context + mia_summarize_context + str(past_conversations)
                }], 
                model='gpt-3.5-turbo'
            )
            print(f"{'-'*50}\n ChatGPT Response:\n{past_convo_summary}\n{'-'*50}\n")
            
            # Add to JSON to track conversation
            self.conversation_context = [
                {"role": "system", "content": system_context}, 
                {"role": "assistant", "content": past_convo_summary
            }]
            asyncio.run(self.saveconversation())

        except:
            print(f"Couldn't Find Conversation File... :(\n")
            pass
    
    async def saveconversation(self):
        # Async save conversation to existing JSON file
        filename = os.path.join(convo_name_directory, 'convo.json')
        async with aiofiles.open(filename, "w") as outfile:
            await outfile.write(json.dumps(self.conversation_context))
        
        print(f'ASYNC Saved Conversation! : {filename}\n')

class Tasks:
    def __init__(self):
        # Embeddings
        self.embeddings = OpenAIEmbeddings()
        # DB 
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index("mia")
        self.vectorstore = Pinecone(self.index, self.embeddings, "text")

    def whatisuserasking(self, transcribedtext):
        asktype = gptObj.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": f"""
                    You are an AI system that has access to prior conversations with the user and a knowledge bank. 
                    The conversational data is not robust and contains both transcripts and summaries of prior discussions.
                    In comparison, the knowledge bank has data from past transcripts of the user's daily life and other information provided by them.
                    
                    You will now receive a user's transcribed speech and are to return which option fits best:
                    1. Answer the user's question directly
                    2. Look for data in the knowledge bank to answer the user's question
                    3. Save the data provided by the user
                    4. User is done/exiting and MIA should sleep
                    5. Return ERROR 
                    
                    Transcription:
                    {transcribedtext}
                """
            }],
            model='gpt-4',
            temp=0.5,
        )
        print(f"{'-'*50}\nGPT-4 Response:\n{asktype}\n{'-'*50}\n")

        if '1' in asktype:
            audioObj.response()
        elif '2' in asktype:
            self.lookinKB(transcribedtext)
            audioObj.listen()
        elif '3' in asktype:
            # TODO: Add logic to upsert relevant data
            audioObj.response()
        elif '4' in asktype:
            audioObj.response(continueconv = False)
            audioObj.elapsed_time = 100000
        elif '5' in asktype:
            audioObj.response()
    
    def lookinKB(self, transcribedtext):
        print(f'~ Check for information in KB ~\n')

        query = gptObj.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": f"""
                    You will receive a user's transcribed speech and are to reword it to extract the topic discussed to feed into another system as a prompt. 
                    
                    Example: 
                    Transcript: Nice MIA but just tell me about the energy drink I'd heard about yesterday
                    Response: energy drink
                    
                    Transcript: {transcribedtext}
                    Response: 
                """
            }],
            model='gpt-4',
            temp=0,
        )
        print(f"{'~'*50}\nquery: {query}\n{'~'*50}\n")
        
        request = gptObj.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": f"""
                    You will receive a user's transcribed speech and are to reword it to extract the task requested. 
                    
                    Example:
                    Transcript: i heard something about gravity and quantum physics right?
                    Response: summarize

                    Transcript: {transcribedtext}
                    Response:
                """
            }],
            model='gpt-4',
            temp=0,
        )
        print(f"{'~'*50}\nrequest: {request}\n{'~'*50}\n")

        docs = self.vectorstore.similarity_search(query, k=3)
        all_content = '\n'.join([doc.page_content for doc in docs])
        print(f'all_content:\n{all_content}\n')
        
        reply = gptObj.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": f'''
                    {system_context}
                    You will receive a transcription of information and are to answer the user's request. If the answer isn't available then say that you don't know it. 
                    Transcription:
                    {all_content}
                    Request:
                    {request}
                '''
            }],
            model='gpt-3.5-turbo',
            temp=1
        )
        print(f"{'~'*50}\nChatGPT reply:\n{reply}\n{'~'*50}\n")

        convObj.conversation_context.append({"role": "assistant", "content": reply})
        asyncio.run(convObj.saveconversation())
        
    def thoughts(self):
        print(f'MIA in Deep Sleep and Thinking...\n')

        while not audioObj.mia_deepsleep:
            continue
        
        pass
# endregion

# region Main
gptObj = GPT()
convObj = Conv()
taskObj = Tasks()
audioObj = Audio()

audioObj.listen()
# endregion