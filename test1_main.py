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
import pandas as pd
from dotenv import main
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
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
class General:
    # Define a function to get the current date based on user input
    def get_date_and_weight(self, text):
        # Regular expression pattern to match today/tomorrow/yesterday
        date_pattern = re.compile(r"(today|tomorrow|day after|yesterday|day before)", re.IGNORECASE)
        # Regular expression pattern to match weight value
        weight_pattern = re.compile(r"\d+(\.\d+)?")

        # Extract date information
        date_match = date_pattern.search(text)
        if date_match:
            date_text = date_match.group().lower()
            time_text = self.get_date(date_text)
            # print("Time:", time_text)
        else:
            print("No date found")

        # Extract weight information
        weight_match = weight_pattern.search(text)
        if weight_match:
            weight_text = weight_match.group()
            # print("Weight:", weight_text)
        else:
            print("No weight found")

        return time_text, weight_text

    # Define a function to get the current date based on user input
    def get_date(self, date_text):
        today = datetime.today().date()
        tomorrow = today + timedelta(days=1)
        dayafter = today + timedelta(days=2)
        yesterday = today - timedelta(days=1)
        daybefore = today - timedelta(days=2)
        date_dict = {
            'today': today,
            'tomorrow': tomorrow,
            'day after': dayafter,
            'yesterday': yesterday,
            'day before': daybefore,
        }
        # print(f'get_date: {date_dict[date_text.lower()]}\n')
        return date_dict[date_text.lower()].strftime('%d-%m-%Y')

class Sheets:
    def __init__(self):
        gs_credentials = {
            "type": "service_account",
            "project_id": "miax-230423",

            "private_key_id": os.environ.get("PRIVATE_KEY_ID"),
            "private_key": os.environ.get("PRIVATE_KEY").replace(r'\n', '\n'),
            "client_email": os.environ.get("CLIENT_EMAIL"),
            "client_id": os.environ.get("CLIENT_ID"),

            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",

            "client_x509_cert_url": os.environ.get("AUTH_PROVIDER_X509_CERT_URL")
        }
        self.gc = gspread.service_account_from_dict(gs_credentials)
        self.dumpfile = self.gc.open("Routine 2023 Dump")

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
                    2. Body Measurement: Something regarding user's muscle size measurement
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
                self.weightlog(transcribedtext)
            if 'body measurement' in requesttype:
                print(f'body measurement entry: {requesttype}\n')
            if 'shower' in requesttype:
                print(f'haircare entry: {requesttype}\n')
            if 'wfo' in requesttype:
                print(f'wfo entry: {requesttype}\n')
        else:
            print(f'No request type identified. Please try again.\n')

    def weightlog(self, transcribedtext):
        print(f'Weight Entry...\n')

        dumpfile_weightsheet = sheetObj.dumpfile.worksheet("Weight")
        # print(f'Current data in dumpfile weight sheet:\n{dumpfile_weightsheet.get_all_values()}\n')

        dateandweight = gptObj.chatgptcall(
            [{
                "role": "system", 
                "content": '''
                    You will receive a user's transcribed speech and are to determine the date of request and the weight of the user in kg. 
                    If time of measurement not mentioned assume it was today. Also if weight unit is not mentioned assume it was kg.
                    Example input: my weight is 75.9 
                    Example output: today - 75.9 kg. 
                    Following is the transcription:'''
                +
                transcribedtext
            }],
            0
        ).lower()
        print(f'dateandweight:\n{dateandweight}\n')

        date_val, weight_val = genObj.get_date_and_weight(dateandweight)
        print(f'Date: {date_val} - Weight: {weight_val}\n')

        # If error like "No date found" AND "No weight found" then end function here
        if "no" in [date_val, weight_val]:
            print(f'Weight sheet not updated.\n')
            return 
        
        # read csv at path and convert to df
        weight_csv_file_path = os.path.join(tasks_name_directory, "weight_log.csv")
        weightcsv_df = pd.read_csv(weight_csv_file_path)
        # weightcsv_df['Date'] = weightcsv_df['Date'].astype(str)
        # weightcsv_df['Weight  '] = weightcsv_df['Weight  '].astype(str)
        print(f'Local weight csv:\n{weightcsv_df.columns.tolist()}\n{weightcsv_df.dtypes}\n{weightcsv_df}\n')

        # Set the index of the dataframe to the 'Date' column
        weightcsv_df = weightcsv_df.set_index('Date')
        # Check if the date exists in the index
        if date_val in weightcsv_df.index:
            print(f'Date exists in Local weight csv\n')

            # If the date exists, insert the weight value at the corresponding row and 'Weight' column
            # weightcsv_df.loc[weightcsv_df['Date'] == date_val, 'Weight  '] = weight_val
            weightcsv_df.loc[date_val, 'Weight  '] = weight_val
        else:
            print(f'Date does NOT exist in Local weight csv\n')

            # If the date does not exist, create a new row with the date and weight values
            new_row = pd.DataFrame({'Weight  ': weight_val}, index=[date_val])
            weightcsv_df = pd.concat([weightcsv_df, new_row])
        
        # Reset the index of the dataframe
        weightcsv_df = weightcsv_df.reset_index()
        # Replace NaN values with ''
        weightcsv_df = weightcsv_df.fillna('')
        
        # write the date and weight values to the dumpfil
        dumpfile_weightsheet.update([weightcsv_df.columns.values.tolist()] + weightcsv_df.values.tolist())
        # write DataFrame to CSV file
        weightcsv_df.to_csv(weight_csv_file_path, index=False)
        
        print(f'Updated Weight sheet!\n')

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
genObj = General()

gptObj = GPT()
taskObj = Tasks()
sheetObj = Sheets()
convObj = Conversations()
audioObj = Audio()

audioObj.listen()
# endregion