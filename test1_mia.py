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
# Loaded Env Vars
main.load_dotenv()
THRESHOLD = float(os.getenv("THRESHOLD"))
TIMEOUT_LENGTH = float(os.getenv("TIMEOUT_LENGTH"))
CHANNELS = int(os.getenv("CHANNELS"))
SWIDTH = int(os.getenv("SWIDTH"))
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))
PORCUPINE_API_KEY = os.getenv("PORCUPINE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PINECONE_API_KEY = str(os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = str(os.getenv("PINECONE_ENV_KEY"))

# Som extra vars
short_normalize = (1.0/32768.0)
audio_format = pyaudio.paInt16

# MIA Vars
SPEAK_FLAG = False
MIA_FIRST_RUN = True

# Path for audio and conversations
audio_name_directory = r'.\audio'
convo_name_directory = r'.\conversations'

# Conversations
summarize_context = "The following is your context of the previous conversation with the user: "
system_context = """
Your name is MIA and you're an AI companion of the user. Keep your responses short. This is your first boot up and your first interaction with the user so ensure that you ask details about them to remember for the future. This includes things liek their name, job/university, residence etc. Ask anything about them until you think it's enough or they stop you.
Internally you have the personality of JARVIS and Chandler Bing combined. You tend to make sarcastic jokes and observations. Do not patronize the user but adapt to how they behave with you.
You help the user with all their requests, questions and tasks. Be honest and admit if you don't know something when asked. 
"""
# endregion

# region Class
class GPT:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
    
    def gpt_completion_call(self, text, engine="text-davinci-003", temp=1, maxtokens=512, decorater='='):
        print('Making GPT-3 request..\n')

        response = openai.Completion.create(
            engine=engine,
            prompt=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0].text.lower()
        print(f'{decorater*50}\nGPT-3 Response:\n{response_text}\n{decorater*50}\n')
        
        audioObj.speak(response_text)
        
        return response_text

    def gpt_chat_call(self, text, model="gpt-3.5-turbo", temp=1, maxtokens=512, decorater='='):
        print(f'Making {("GPT-4" if model=="gpt-4" else "ChatGPT")} request..\n')

        response = openai.ChatCompletion.create(
            model=model,
            messages=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0]["message"]["content"].lower()
        print(f'{decorater*50}\n{("GPT-4" if model=="gpt-4" else "ChatGPT")} Response:\n{response_text}\n{decorater*50}\n')

        if SPEAK_FLAG: audioObj.speak(response_text)
        
        return response_text

    def whispercall(self, filename):
        print('Making Whisper request..\n')

        audio_file= open(filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en",
            prompt="don't make up words to fill in the rest of the sentence. if background noise return ."
        )
        transcript_text = transcript["text"]
        print(f'{"-"*50}\nTranscript: {transcript_text}\n{"-"*50}\n')
        
        return transcript_text

class Audio:
    def __init__(self):
        set_api_key(ELEVENLABS_API_KEY)
        
        self.elapsed_time = 0
        self.last_audio_time = time.time()
        self.require_hotword = False

        # Set up hotword detector
        self.porcupine = pvporcupine.create(
            access_key=PORCUPINE_API_KEY,
            keyword_paths=[r'models\hey-mia_en_windows_v3_0_0.ppn']
        )

        # Set up recorder
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=audio_format,
            channels=CHANNELS,
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
        print(f'Started Listening...\n')

        self.elapsed_time = 0
        self.last_audio_time = time.time()
        while True:
            pcm = self.stream.read(self.porcupine.frame_length)

            if MIA_FIRST_RUN:
                print(f'Booting Up...\n')
                MIA_FIRST_RUN = False
                convObj.conversation_context = [{"role": "system", "content": system_context}]
                self.response(model='gpt-4')

            if self.require_hotword:
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    print("Hotword Detected!\n")
                    self.require_hotword = False
                    self.record()
                else:
                    self.elapsed_time = time.time() - self.last_audio_time

            else:
                rms_val = self.rms(pcm)
                if rms_val > THRESHOLD:
                    self.record()
                else:
                    self.elapsed_time = time.time() - self.last_audio_time

                    if self.elapsed_time >= TIMEOUT_LENGTH:
                        print("Sleeping zzz...\n")
                        self.require_hotword = True

    def record(self):
        print(f'Recording!\n')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(self.porcupine.frame_length)
            if self.rms(data) >= THRESHOLD: 
                end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        
        recording = b''.join(rec)
        timestamp = time.strftime('%Y%m%d%H%M%S')
        self.writeaudiofile(timestamp, recording)

    def writeaudiofile(self, timestamp, recording):
        filename = os.path.join(audio_name_directory, '{}.wav'.format(timestamp))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(audio_format))
        wf.setframerate(self.porcupine.sample_rate)
        wf.writeframes(recording)
        wf.close()
        print(f'Written to file: {filename}\n')

        self.transcribe(filename)

    def transcribe(self, filename):
        print('Transcribing...\n')

        transcribe_output = gptObj.whispercall(filename)

        if transcribe_output == '' or transcribe_output=='.':
            self.listen()
        else:
            convObj.conversation_context.append({"role": "user", "content": transcribe_output})
            asyncio.run(convObj.saveconversation())
            
            doestaskhaveresponse = taskObj.checkifrequesttype(transcribe_output)
            
            if doestaskhaveresponse is False:
                self.response()
            else:
                self.listen()

    def response(self, model='gpt-3.5-turbo'):
        print('Responding...\n')

        gptresponse = gptObj.gpt_chat_call(text = convObj.conversation_context, model=model)

        # If the phrase 'ai language model' shows up in a response then revert to GPT-3 to get a new response 
        if 'ai language model' in gptresponse.lower():
            plaintext = genObj.conversation_to_text(text = convObj.conversation_context)
            gptresponse = gptObj.gpt_completion_call(text = plaintext, engine = "text-davinci-003")
        
        convObj.conversation_context.append({"role": "assistant", "content": gptresponse})
        asyncio.run(convObj.saveconversation())

        self.listen()

    def speak(self, text):
        audio = generate(
            text=response_text,
            voice="Bella", model="eleven_monolingual_v1",
            stream=True
        )
        stream(audio)

class Conv:
    def __init__(self):
        self.conversation_context = []

        filename = os.path.join(convo_name_directory, 'convo.json')
        try:
            past_conversations = json.load(open(filename))
            MIA_FIRST_RUN = False

            if any('user' in d['role'] for d in past_conversations):
                print('Loaded past conversations...\n')

                summarized_prompt = system_context + summarize_context + str(past_conversations)
                
                past_convo_summary = gptObj.gpt_chat_call(text = [{"role": "system", "content": summarized_prompt},])
                print(f'past_convo_summary: {past_convo_summary}\n')
                
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": past_convo_summary}]
                print(f'self.conversation_context: {self.conversation_context}\n')
            
            elif any('assistant' in d['role'] for d in past_conversations):
                print('Assistant\'s past response loaded!')
                
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": past_conversations[1]['content']}]
            
            else:
                print('No past conversations to load from!')
                
                greeting = gptObj.gpt_chat_call(text = [{"role": "system", "content": system_context},])
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": greeting}]
            
            print(f'Conversation_context: {self.conversation_context}\n')
            asyncio.run(self.saveconversation())

        except:
            pass

    async def saveconversation(self):
        filename = os.path.join(convo_name_directory, 'convo.json')
        async with aiofiles.open(filename, "w") as outfile:
            await outfile.write(json.dumps(self.conversation_context))
        
        print(f'ASYNC Saved Conversation! : {filename}\n')

class Tasks:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index("mia")
        self.vectorstore = Pinecone(self.index, self.embeddings, "text")

    def checkifrequesttype(self, transcribedtext):
        requesttype = gptObj.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": f"""
                    {system_context}

                    You will receive a user's transcribed speech and are to determine if the user is requesting for some information or the completion of a task. 
                    If he is then return Y, else return N. 
                    Transcription:"""
                +
                transcribedtext
            }],
            model='gpt-4',
            temp=0,
            decorater='~'
        )
        
        convContinue = None
        if 'y' in requesttype:
            print(f'Request exists.\n')

            query = gptObj.gpt_chat_call(
                text=[{
                    "role": "system", 
                    "content": f"""
                        {system_context}
                        
                        You will receive a user's transcribed speech and are to reword it to extract the topic discussed to feed into another system as a prompt. 
                        Example: 
                        Transcript: Nice MIA but just tell me about the energy drink I'd heard about yesterday
                        Response: energy drink
                        Transcript:"""
                    +
                    transcribedtext
                }],
                model='gpt-4',
                temp=0,
                decorater='~'
            )
            print(f'query: {query}\n')
            request = gptObj.gpt_chat_call(
                text=[{
                    "role": "system", 
                    "content": f"""
                        {system_context}
                        
                        You will receive a user's transcribed speech and are to reword it to extract the task requested. 
                        Example:
                        Transcript: i heard something about gravity and quantum physics so could you just summarize that for me?
                        Response: summarize the transcript above
                        Transcript:"""
                    +
                    transcribedtext
                }],
                model='gpt-4',
                temp=0,
                decorater='~'
            )
            print(f'request: {request}\n')

            docs = self.vectorstore.similarity_search(query, k=1)
            print(f'docs: {docs}\n')
            reply = gptObj.gpt_chat_call(
                text=[{
                    "role": "system", 
                    "content": '''
                        You will receive a transcription of information and are to answer the user's request. If the answer isn't available then say that you don't know it. 
                        Transcription:'''
                    +
                    docs[0].page_content
                    +
                    '''Request:'''
                    +
                    request
                }],
                model='gpt-3.5-turbo',
                temp=0.5
            )
            print(f'reply: {reply}\n')

            convObj.conversation_context.append({"role": "assistant", "content": reply})
            asyncio.run(convObj.saveconversation())
            
            convContinue = True
        else:
            print(f'No request type identified. Please try again.\n')
            convContinue = False
        
        return convContinue
# endregion

# region Main
gptObj = GPT()
convObj = Conv()
taskObj = Tasks()
audioObj = Audio()

audioObj.listen()
# endregion