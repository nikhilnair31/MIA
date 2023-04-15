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
from dotenv import main
# endregion

# region Vars
# Loaded Env Vars
main.load_dotenv()
Threshold = float(os.getenv("THRESHOLD"))
TIMEOUT_LENGTH = float(os.getenv("TIMEOUT_LENGTH"))
gpt3_api_key = str(os.getenv("OPENAI_API_KEY"))
chunk = int(os.getenv("CHUNK"))
CHANNELS = int(os.getenv("CHANNELS"))
RATE = int(os.getenv("RATE"))
swidth = int(os.getenv("SWIDTH"))

SHORT_NORMALIZE = (1.0/32768.0)
FORMAT = pyaudio.paInt16

# Path for audio and conversations
audio_name_directory = r'.\audio'
convo_name_directory = r'.\conversations'

# API Key Auth
# openai.api_key = gpt3_api_key

# Prompt for transcribing
prompt = "MIA Food log entry. Rice 200 grams. End."

# Conversations
summarize_context = "The following is your context of the previous conversation with the user: "
system_context = "You are MIA, an AI desk robot that makes sarcastic jokes and observations. You have the personality of Chandler Bing from Friends and Barney Stinson from HIMYM. Initiate with a greeting. Never break character."
conversation_context = []
# endregion

# region Class
class GPT:
    def __init__(self):
        self.openai.api_key = gpt3_api_key

    def chatgptcall(self, text):
        print('Making GPT request..')

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=text
        )
        response_text = response["choices"][0]["message"]["content"]
        print(f'Response: {response_text}\n')
        return response_text

    def whispercall(self, filename):
        print('Making Whisper request..')

        audio_file= open(filename, "rb")
        # options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en"
            # options
        )
        transcript_text = transcript["text"]
        print(f'Transcript: {transcript_text}\n')
        return transcript_text

class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)
    
    def loadconversation(self):
        filename = os.path.join(convo_name_directory, 'convo.json')
        print(f'Conversation Filename: {filename}\n')

        past_conversations = json.load(open(filename))

        # FIXME: Fix how context of previous conversations is loaded and initiate with a greeting
        if any('user' in d['role'] for d in past_conversations):
            print('Loaded past conversations\n')

            summarized_prompt = system_context + summarize_context + str(past_conversations)
            print(f'Summarized_prompt: {summarized_prompt}\n')
            
            past_convo_summary = g.chatgptcall([{"role": "system", "content": summarized_prompt},])
            conversation_context.append([{"role": "system", "content": system_context}, {"role": "assistant", "content": past_convo_summary}])
            # conversation_context.append({"role": "assistant", "content": past_convo_summary})
            self.saveconversation(conversation_context)
        else:
            print('No past conversations to load from!')
            conversation_context.append({"role": "system", "content": system_context})
        
        print(f'Conversation_context: {conversation_context}\n')
        g.chatgptcall(conversation_context)

    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        
        self.write(b''.join(rec))

    def write(self, recording):
        n_files = len(os.listdir(audio_name_directory))

        filename = os.path.join(audio_name_directory, '{}.wav'.format(n_files))
        print('Filename: {}'.format(filename))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))

        self.transcribe(filename)

    def saveconversation(self, conversationjson):
        filename = os.path.join(convo_name_directory, 'convo.json')
        print('Filename: {}'.format(filename))

        with open(filename, "w") as outfile:
            json.dump(conversationjson, outfile)
        
        print('Save convo')

    def transcribe(self, filename):
        print('Transcribing..')

        transcribe_output = g.transcribe(filename)
        print(f'Transcript: {transcribe_output}\n')

        # Condition to check if user asked to make a log entry. If yes, check type of request
        if 'log' in transcribe_output and 'entry' in transcribe_output:
            print(f'User asked to make log entry\n')
            self.makelogentry(transcribe_output)

        conversation_context.append({"role": "user", "content": transcribe_output})
        self.saveconversation(conversation_context)
        self.response(conversation_context)

    def response(self, conversation_context):
        print('Responding..')

        conversation_context.append({"role": "assistant", "content": g.chatgptcall(conversation_context)})
        self.saveconversation(conversation_context)
        self.listen()
# endregion

# region Main
g = GPT()
a = Recorder()
a.loadconversation()
a.listen()
# endregion