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
import pyaudio
import pandas as pd
from dotenv import main
from datetime import datetime, timedelta
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
tasks_name_directory = r'.\tasks'

# Prompt for transcribing
prompt = "MIA Food log entry. Rice 200 grams. End."

# Conversations
summarize_context = "The following is your context of the previous conversation with the user: "
system_context = "You are MIA, an AI desk robot that makes sarcastic jokes and observations. You have the personality of Chandler Bing from Friends and Barney Stinson from HIMYM. Initiate with a greeting."
# endregion

# region Class
class GPT:
    def __init__(self):
        openai.api_key = gpt3_api_key
    
    def gpt3call(self, text, temp=0.7, maxtokens=256):
        print('Making GPT-3 request..\n')
        
        print(f'GPT-3 Call: {text}\n')

        plaintext = ''
        for conv in text:
            if 'system' in conv['role']:
                plaintext += conv['content'] + '\n'
            if 'assistant' in conv['role']:
                plaintext += 'assistant: ' + conv['content'] + '\n'
            if 'user' in conv['role']:
                plaintext += 'user: ' + conv['content'] + '\n'
        print(f'GPT-3 Prompt:\n{plaintext}\n')

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=plaintext,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0].text.lower()
        print(f'GPT-3 Response:\n{response_text}\n')
        return response_text

    def chatgptcall(self, text, temp=0.7, maxtokens=256):
        print('Making ChatGPT request..\n')

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0]["message"]["content"].lower()
        print(f'Response:\n{response_text}\n')
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

class Tasks:
    def __init__(self):
        self.s3_client = boto3.client('s3')

        # Check if tasks folder exists else create it
        if not os.path.exists(tasks_name_directory):
            os.makedirs(tasks_name_directory)
            
        self.create_file_at_path(tasks_name_directory, "weight_log.csv", ['Date', 'Weight (kg)'])
        self.create_file_at_path(tasks_name_directory, "sizes_log.csv", ['Date', 'Chest (in)', 'Waist (in)', 'Bicep (in)', 'Thigh (in)'])
        self.create_file_at_path(tasks_name_directory, "selfcare_log.csv", ['Date', 'ShowerTime', 'Routine'])
        self.create_file_at_path(tasks_name_directory, "office_visits_log.csv", ['Date', 'Visited?'])

    def create_file_at_path(self, path, filename, headerlist):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            print("File already exists!")
        else:
            start_date = datetime(datetime.now().year, 1, 1)
            end_date = start_date + timedelta(days=365*5) - timedelta(days=1)
            date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
            with open(full_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headerlist)
                for date in date_list:
                    writer.writerow([date.strftime('%d-%m-%Y')])
            print(f"Created file {full_path}")

    def checkifrequesttype(self, transcribedtext):
        requesttype = gptObj.chatgptcall(
            [{
                "role": "system", 
                "content": '''
                    You will receive a user's transcribed speech and are to determine which of the following categories it belongs to. Can belong to more than 1 category. If it does then return Y followed by the category itself, else return N. 
                    Categories:
                    1. Weight: Something regarding user's weight
                    2. Body Measurement: Something regarding user's body measurement
                    3. Shower: Something regarding user's shower or haircare
                    4. WFO: Something regarding user working from office
                    Following is the transcription:'''
                +
                transcribedtext
            }],
            0
        )
        
        if 'y' in requesttype:
            if 'weight' in requesttype:
                print(f'weight entry: {requesttype}\n')
            if 'body measurement' in requesttype:
                print(f'body measurement entry: {requesttype}\n')
            if 'shower' in requesttype:
                print(f'haircare entry: {requesttype}\n')
            if 'wfo' in requesttype:
                print(f'wfo entry: {requesttype}\n')
        else:
            print(f'No request type identified. Please try again.\n')

    # FIXME: Rewrite this
    def makelogentry(self, transcribedtext):
        #TODO: Instead of using static conditions just use ChatGPT to confirm wheather the user's command is regarding logging their weight 
        #TODO: Help model understand difference between today/tomorrow/yesterday and fill date accordingly
        if 'log' in transcribedtext and 'entry' in transcribedtext:
            print(f'User asked to make log entry\n')
            if 'weight' in transcribedtext:
                self.weightlog(transcribedtext)
    
    # FIXME: Rewrite this
    def weightlog(self, transcribedtext):
        print(f'User asked to make WEIGHT entry\n')

        bucket_name = 'nik-bank-data'
        filename = "weight_log.csv"
        full_path = os.path.join(tasks_name_directory, filename)

        regex = r'(\d+(?:\.\d+)?)\s*(kg|KG|Kg|kG)?'
        matches = re.findall(regex, transcribedtext, re.IGNORECASE)
        if not matches:
            return None
        print(f'extract_weight: {matches[0][0]}\n')

        today_date = datetime.now().strftime('%d-%m-%Y')
        today_weight = float(matches[0][0])

        print(f'Writing date: {today_date} and weight: {today_weight} to CSV..\n')

        dates_df = pd.read_csv(full_path)
        dates_df['Date'] = pd.to_datetime(dates_df['Date'], format='%d-%m-%Y', errors='coerce')

        todays_weight_df = pd.DataFrame({'Date': [today_date], 'Weight (kg)': [today_weight]})
        todays_weight_df['Date'] = pd.to_datetime(todays_weight_df['Date'], format='%d-%m-%Y', errors='coerce')

        merged_df = pd.merge(dates_df, todays_weight_df, on='Date', how='left')
        # merged_df.drop('Weight (kg)_x', axis=1, inplace=True)
        # merged_df.rename(columns={'Weight (kg)_y': 'Weight'}, inplace=True)

        merged_df.to_csv(full_path, index=False)
        print(f'Wrote date and weight data!\n')

        #FIXME: Setup permissions/IAM roles to write to S3 bucket
        self.s3_client.upload_file(full_path, bucket_name, filename)
        print(f'Uploaded to S3!\n')

class Audio:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=chunk
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
        print(f'Listening beginning...\n')
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
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
        
        # Use that transcription and check if it contains any task request and if so which?
        taskObj.checkifrequesttype(transcribe_output)

        # Append user's transcribed speech to the conversation
        convObj.conversation_context.append({"role": "user", "content": transcribe_output})
        
        # Save the current conversation
        convObj.saveconversation()
        
        # Respond to user's transcribed speech
        self.response()

    def response(self):
        print('Responding..')

        # Transcribe the user's speech
        gptresponse = gptObj.chatgptcall(convObj.conversation_context)
        
        # If the phrase 'ai language model' shows up in a response then revert to GPT-3 to get a new response 
        if 'ai language model' in gptresponse.lower():
            gptresponse = gptObj.gpt3call(convObj.conversation_context)

        # Append GPT's response to the conversation
        convObj.conversation_context.append({"role": "assistant", "content": gptresponse})
        
        # Save the current conversation
        convObj.saveconversation()

        self.listen()

class Conversations:
    def __init__(self):
        self.conversation_context = []

        filename = os.path.join(convo_name_directory, 'convo.json')
        print(f'Conversation Filename: {filename}\n')

        past_conversations = json.load(open(filename))

        if any('user' in d['role'] for d in past_conversations):
            print('Loaded past conversations\n')

            summarized_prompt = system_context + summarize_context + str(past_conversations)
            print(f'Summarized_prompt: {summarized_prompt}\n')
            
            past_convo_summary = gptObj.chatgptcall([{"role": "system", "content": summarized_prompt},])
            self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": past_convo_summary}]
        else:
            print('No past conversations to load from!')
            self.conversation_context = [{"role": "system", "content": system_context}]
        
        print(f'Conversation_context: {self.conversation_context}\n')
        self.saveconversation()

    def saveconversation(self):
        filename = os.path.join(convo_name_directory, 'convo.json')
        print(f'Saveconversation Filename: {filename}\n')

        with open(filename, "w") as outfile:
            json.dump(self.conversation_context, outfile)
        
        print('Saved Conversation!\n')
# endregion

# region Main
gptObj = GPT()
taskObj = Tasks()
convObj = Conversations()
audioObj = Audio()

audioObj.listen()
# endregion