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

main.load_dotenv()

gpt3_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = gpt3_api_key

# prompt = "MIA Food log entry. Rice 200 grams. End."

Threshold = 10

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 3

audio_name_directory = r'.\audio'
convo_name_directory = r'.\conversations'

conversation_context = [
    {
        "role": "system", 
        "content": "You are GlaedeBot, an AI-driven desk robot that makes sarcastic jokes and observations based on the user's input. You have the personality of Chandler Bind from Friends and Barney Stinson from HIMYM."
    },
]

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
        print('Returning to listening')

        self.transcribe(filename)

    def convo(self):
        filename = os.path.join(convo_name_directory, 'convo.json')
        print('Filename: {}'.format(filename))

        with open(filename, "w") as outfile:
            json.dump(conversation_context, outfile)
        
        print('Save convo')

    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.convo()
                self.record()

    def transcribe(self, filename):
        print('Transcribing..')

        audio_file= open(filename, "rb")
        # options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en"
            # options
        )
        transcript_text = transcript["text"]
        print('Transcript: {}'.format(transcript_text))

        conversation_context.append({"role": "user", "content": transcript_text})
        self.response(transcript_text)

    def response(self, transcript):
        print('Responding..')

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_context
        )
        response_text = response["choices"][0]["message"]["content"]
        print('Response: {}'.format(response_text))

        conversation_context.append({"role": "assistant", "content": response_text})
        self.listen()

a = Recorder()
a.listen()