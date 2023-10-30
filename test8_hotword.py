import os
import time
import math
import wave
import struct
import pyaudio
import pvporcupine
from dotenv import main

main.load_dotenv()
Threshold = float(os.getenv("THRESHOLD"))
swidth = int(os.getenv("SWIDTH"))
TIMEOUT_LENGTH = float(os.getenv("TIMEOUT_LENGTH"))
CHANNELS = int(os.getenv("CHANNELS"))
RATE = int(os.getenv("RATE"))
chunk = int(os.getenv("CHUNK"))
access_key = os.getenv("PICO_ACCESS_KEY")
gpt3_api_key = str(os.getenv("OPENAI_API_KEY"))

SHORT_NORMALIZE = (1.0/32768.0)
FORMAT = pyaudio.paInt16

# Path for audio and conversations
audio_name_directory = r'.\audio'
convo_name_directory = r'.\conversations'

porcupine = pvporcupine.create(
  access_key=access_key,
  keywords=['picovoice', 'bumblebee']
)

class Audio:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=FORMAT,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
    
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

    def listen(self):
        print('Listening beginning...')
        while True:
            pcm = self.stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Hotword Detected")
                self.record()

    def record(self):
        print(f'Noise detected, recording beginning!\n')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        
        self.writeaudiofile(b''.join(rec))

    def writeaudiofile(self, recording):
        n_files = len(os.listdir(audio_name_directory))
        filename = os.path.join(audio_name_directory, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print(f'Written to file: {filename}\n')

        self.transcribe(filename)

    def transcribe(self, filename):
        print('Transcribing...')

        # Transcribe the user's speech
        transcribe_output = gptObj.whispercall(filename)

        # Append user's transcribed speech to the conversation
        convObj.conversation_context.append({"role": "user", "content": transcribe_output})
        # Save the current conversation
        asyncio.run(convObj.saveconversation())
        
        # Use that transcription and check if it contains any task request and if so which?
        doestaskhaveresponse = taskObj.checkifrequesttype(transcribe_output)
        
        # Respond to user's transcribed speech
        if doestaskhaveresponse is False:
            self.response()
        else:
            self.listen()

    def response(self):
        print('Responding..')

        # Transcribe the user's speech
        gptresponse = gptObj.gpt_chat_call(text = convObj.conversation_context)
        
        # If the phrase 'ai language model' shows up in a response then revert to GPT-3 to get a new response 
        if 'ai language model' in gptresponse.lower():
            plaintext = genObj.conversation_to_text(text = convObj.conversation_context)
            gptresponse = gptObj.gpt_completion_call(text = plaintext, engine = "text-davinci-003")

        # Append GPT's response to the conversation
        convObj.conversation_context.append({"role": "assistant", "content": gptresponse})
        # Save the current conversation
        asyncio.run(convObj.saveconversation())

        self.listen()

audio_obj = Audio()
try:
    audio_obj.listen()
finally:
    audio_obj.stream.stop_stream()
    audio_obj.stream.close()
    audio_obj.p.terminate()
    porcupine.delete()