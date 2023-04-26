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
import asyncio
import aiofiles
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
    async def create_create_file_at_path(self, path, filename, headerlist):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            print(f"File {full_path} already exists!")
        else:
            start_date = datetime(datetime.now().year, 1, 1)
            end_date = start_date + timedelta(days=365*5) - timedelta(days=1)
            date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
            async with aiofiles.open(full_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headerlist)
                for date in date_list:
                    writer.writerow([date.strftime('%d-%m-%Y')])
            print(f"Created file {full_path}")

    def get_weight(self, text):
        weight_pattern = re.compile(r"\d+(\.\d+)?")

        weight_match = weight_pattern.search(text)
        if weight_match:
            weight_text = weight_match.group()
            # print("Weight:", weight_text)
            return weight_text
        else:
            print("No weight found")
            return ''

    def get_shower_product_combo(self, text):
        haircare_pattern = re.compile(r"(none|na|shampoo only|conditioner only|shampoo and conditioner|shampoo \+ conditioner|shampoo|conditioner)", re.IGNORECASE)

        haircare_match = haircare_pattern.search(text)
        if haircare_match:
            haircare_text = haircare_match.group().lower()
            # print("Haircare:", haircare_text)
            return haircare_text
        else:
            print("No haircare products found")
            return 'na'

    def get_wfo_visit(self, text):
        if 'wfo' in text:
            wfovisit_text = '1'
            # print("WFO Visit:", wfovisit_text)
            return wfovisit_text
        else:
            print("No WO visit found")
            return ''

    def get_bodypart(self, text):
        sizes_pattern = re.compile(r"\b(bicep|biceps|arm|arms|waist|hips|thigh|legs|chest)s?\b", re.IGNORECASE)

        sizes_match = sizes_pattern.search(text)
        if sizes_match:
            sizes_text = sizes_match.group().lower()
            # print("Body Part Size:", sizes_text)
            return sizes_text
        else:
            print("No date found")
            return ''

    def get_morningevening_etc(self, text):
        timetext_pattern = re.compile(r"(morning|evening)", re.IGNORECASE)

        timetext_match = timetext_pattern.search(text)
        if timetext_match:
            time_text = timetext_match.group().lower()
            # print("Time:", time_text)
            return time_text
        else:
            print("No date found")
            return 'na'

    def get_date(self, text):
        timetext_pattern = re.compile(r"(today|tomorrow|day after|yesterday|day before)", re.IGNORECASE)
        timetext_match = timetext_pattern.search(text)
        
        today = datetime.today().date()
        tomorrow = today + timedelta(days=1)
        dayafter = today + timedelta(days=2)
        yesterday = today - timedelta(days=1)
        daybefore = today - timedelta(days=2)
        date_dict = {
            'morning': today,
            'evening': today,
            'today': today,
            'tomorrow': tomorrow,
            'day after': dayafter,
            'yesterday': yesterday,
            'day before': daybefore,
        }

        if timetext_match:
            time_text = timetext_match.group().lower()
            date_text = date_dict[time_text.lower()].strftime('%d-%m-%Y')
            # print(f'Time: {time_text} - Date: {date_text}\n')
            return date_text
        else:
            print("No date found")
            return ''

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
        client = gspread.service_account_from_dict(gs_credentials)
        self.routineSheetFile = client.open("Routine 2023 Dump")
        self.bankStatementSheetFile = client.open("Bank 2023 Dump")

class GPT:
    def __init__(self):
        openai.api_key = gpt3_api_key
    
    def gpt3call(self, text, temp=0.7, maxtokens=256, showLog=True):
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
        
        if showLog:
            print(f'GPT-3 Response:\n{response_text}\n')
        
        return response_text

    def chatgptcall(self, text, temp=0.7, maxtokens=256, showLog=True):
        print('Making ChatGPT request..\n')

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0]["message"]["content"].lower()
        
        if showLog:
            print(f'Response:\n{response_text}\n')
        
        return response_text

    def whispercall(self, filename, showLog=True):
        print('Making Whisper request..')

        audio_file= open(filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en"
            # options
        )
        transcript_text = transcript["text"]

        if showLog:
            print(f'Transcript: {transcript_text}\n')
        
        return transcript_text

# FIXME: Add ability to mention specific dates to update
class Tasks:
    def __init__(self):
        self.s3_client = boto3.client('s3')

        # Check if tasks folder exists else create it
        if not os.path.exists(tasks_name_directory):
            os.makedirs(tasks_name_directory)

        # async check for multiple files based on tasks
        asyncio.run(self.taskfilegen())

    async def taskfilegen(self):
        await asyncio.gather(
            genObj.create_create_file_at_path(tasks_name_directory, "weight_log.csv", ['Date', 'Weight (kg)']),
            genObj.create_create_file_at_path(tasks_name_directory, "sizes_log.csv", ['Date', 'Chest (in)', 'Waist (in)', 'Bicep (in)', 'Thigh (in)']),
            genObj.create_create_file_at_path(tasks_name_directory, "selfcare_log.csv", ['Date', 'ShowerTime', 'Routine']),
            genObj.create_create_file_at_path(tasks_name_directory, "office_visits_log.csv", ['Date', 'Visited?']),
            # genObj.create_create_file_at_path(tasks_name_directory, "test_log.csv", ['Date', 'ABC'])
        )

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
                    5. Haircut: Something regarding user getting their hair cut
                    Example input: Just note down my weight for today as 73 kg and also I showered in the evening without any soap or conditioner. 
                    Example output: y - weight, shower.
                    Following is the transcription:'''
                +
                transcribedtext
            }],
            0
        )
        
        # FIXME: Can throw error in case requesttype returns "y weight, n body measurement, n shower, n wfo"
        if 'y' in requesttype:
            if 'weight' in requesttype:
                self.weightlog(transcribedtext)
            if 'body measurement' in requesttype:
                self.sizemeasurementlog(transcribedtext)
            if 'shower' in requesttype:
                self.showerlog(transcribedtext)
            if 'wfo' in requesttype:
                self.wfolog(transcribedtext)
            if 'haircut' in requesttype:
                self.haircut()
        else:
            print(f'No request type identified. Please try again.\n')

    def weightlog(self, transcribedtext):
        print(f'Weight Entry...\n')

        prompt = [{
            "role": "system", 
            "content": '''
                You will receive a user's transcribed speech and are to determine the date of request and the weight of the user in kg. 
                If time of measurement not mentioned assume it was today. Also if weight unit is not mentioned assume it was kg.
                Example input: my weight is 75.9 
                Example output: today - 75.9 kg. 
                Following is the transcription:'''
            +
            transcribedtext
        }]
        temp = 0
        maxtokens = 256
        showlog = False

        dateandweighttext = gptObj.chatgptcall(prompt, temp, maxtokens, showlog)
        dateandweighttext = dateandweighttext.lower()
        
        # Use helper functions from General class to pull date and weight values
        date_val = genObj.get_date(dateandweighttext)
        weight_val = genObj.get_weight(dateandweighttext)
        print(f'dateandweighttext: {dateandweighttext}\n date: {date_val} - weight: {weight_val}\n')

        # If error like "No date found" AND "No weight found" then end function here
        if "no" in [date_val, weight_val]:
            print(f'Weight sheet not updated.\n')
            return 

        # Open sheet in Sheets file
        datasheet = 'Weight'
        dumpfile_weightsheet = sheetObj.routineSheetFile.worksheet(datasheet)
        # read csv at path and convert to df
        localdatacsv = 'weight_log.csv'
        weight_csv_file_path = os.path.join(tasks_name_directory, localdatacsv)
        weightcsv_df = pd.read_csv(weight_csv_file_path)
        print(f'Local weight csv:\n{weightcsv_df.columns.tolist()}\n{weightcsv_df.dtypes}\n{weightcsv_df}\n')

        # Set the index of the dataframe to the 'Date' column
        # Check if the date exists in the index
        # If yes then replace weight value, if not insert the date and weight value
        weightcsv_df = weightcsv_df.set_index('Date')
        # print(f'Local weight csv:\n{weightcsv_df.columns[:5]}\n')
        if date_val in weightcsv_df.index:
            print(f'Date exists in Local weight csv\n')
            weightcsv_df.loc[date_val, 'Weight  '] = weight_val
        else:
            print(f'Date does NOT exist in Local weight csv\n')
            new_row = pd.DataFrame({'Weight  ': weight_val}, index=[date_val])
            weightcsv_df = pd.concat([weightcsv_df, new_row])
        # print(f'Local weight csv:\n{weightcsv_df.columns[:5]}\n')
        
        # Reset the index of the dataframe and replace NaN values with ''
        weightcsv_df = weightcsv_df.reset_index()
        weightcsv_df = weightcsv_df.fillna('')
        # print(f'Local weight csv:\n{weightcsv_df.columns[:5]}\n')

        # write the dataFrame to CSV file
        weightcsv_df.to_csv(weight_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_weightsheet.update([weightcsv_df.columns.values.tolist()] + weightcsv_df.values.tolist())
        
        print(f'Updated Weight sheet!\n')
    
    def showerlog(self, transcribedtext):
        print(f'Shower Entry...\n')

        prompt = [{
            "role": "system", 
            "content": '''
                You will receive a user's transcribed speech and are to determine the date of request, time of shower and haircare products used. 
                If user has not showered, default time of shower to "na" and default haircare products used to "na". 
                If user has showered, default time of shower to "morning" unless mentioned otherwise and default haircare products used to "Shampoo + Conditioner" unless mentioned otherwise. 
                Default date of shower to "today" unless mentioned otherwise.
                Example input: I just showered and used both shampoo and conditioner 
                Example output: today - Morning - Shampoo and Conditioner. 
                Following is the transcription:'''
            +
            transcribedtext
        }]
        temp = 0
        maxtokens = 256
        showlog = True

        datetimeshower = gptObj.chatgptcall(prompt, temp, maxtokens, showlog)
        datetimeshower = datetimeshower.lower()

        date_val = genObj.get_date(datetimeshower)
        time_val = genObj.get_morningevening_etc(datetimeshower)
        haircare_val = genObj.get_shower_product_combo(datetimeshower)
        print(f'datetimeshower: {datetimeshower}\n date: {date_val} - time: {time_val} - haircare: {haircare_val}\n')

        # If error like "No XYZ found" then end function here
        if "no" in [date_val, time_val, haircare_val]:
            print(f'Shower sheet not updated.\n')
            return 
        
        # Open sheet in Sheets file
        datasheet = 'Selfcare'
        dumpfile_showersheet = sheetObj.routineSheetFile.worksheet(datasheet)
        # print(f'Current data in routineSheetFile weight sheet:\n{dumpfile_showersheet.get_all_values()}\n')
        # Read csv at path and convert to df
        localdatacsv = 'selfcare_log.csv'
        shower_csv_file_path = os.path.join(tasks_name_directory, localdatacsv)
        showercsv_df = pd.read_csv(shower_csv_file_path)
        print(f'Local shower csv:\n{showercsv_df.columns.tolist()}\n{showercsv_df.dtypes}\n{showercsv_df}\n')

        # Set the index of the dataframe to the 'Date' column
        # Check if the date exists in the index
        showercsv_df = showercsv_df.set_index('Date')
        print(f'Local shower csv:\n{showercsv_df.columns.tolist()}\n')
        if date_val in showercsv_df.index:
            print(f'Date exists in Local shower csv\n')
            showercsv_df.loc[date_val, 'Time'] = time_val
            showercsv_df.loc[date_val, 'Haircare'] = haircare_val
        else:
            print(f'Date does NOT exist in Local shower csv\n')
            new_row = pd.DataFrame({'Time': time_val, 'Haircare': haircare_val}, index=[date_val])
            showercsv_df = pd.concat([showercsv_df, new_row])
        print(f'Local shower csv:\n{showercsv_df.columns.tolist()}\n')
        
        # Reset the index of the dataframe and replace NaN values with ''
        showercsv_df = showercsv_df.reset_index()
        showercsv_df = showercsv_df.fillna('')
        print(f'Local shower csv:\n{showercsv_df.columns.tolist()}\n')

        # write the dataFrame to CSV file
        showercsv_df.to_csv(shower_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_showersheet.update([showercsv_df.columns.values.tolist()] + showercsv_df.values.tolist())
        
        print(f'Updated Shower sheet!\n')

    def wfolog(self, transcribedtext):
        print(f'WFO Entry...\n')

        prompt = [{
            "role": "system", 
            "content": '''
                You will receive a user's transcribed speech and are to determine the date of request, and whether they visited office on that date or not. 
                If date not mentioned assume it was today.
                Example input 1: I'll be heading to work now
                Example output 1: today - WFO
                Example input 2: I just had an update, I am not going to office tomorrow.
                Example output 2: tomorrow - WFH
                Following is the transcription:'''
            +
            transcribedtext
        }]
        temp = 0
        maxtokens = 256
        showlog = False

        dateofficevisit = gptObj.chatgptcall(prompt, temp, maxtokens, showlog)
        dateofficevisit = dateofficevisit.lower()

        date_val = genObj.get_date(dateofficevisit)
        visit_val = genObj.get_wfo_visit(dateofficevisit)
        print(f'dateofficevisit: {dateofficevisit}\n date: {date_val} - visit: {visit_val}\n')

        # If error like "No XYZ found" then end function here
        if "no" in [date_val, visit_val]:
            print(f'WFO sheet not updated.\n')
            return 
        
        # Open sheet in Sheets file
        datasheet = 'Office Visits'
        dumpfile_wfosheet = sheetObj.routineSheetFile.worksheet(datasheet)
        # print(f'Current data in routineSheetFile weight sheet:\n{dumpfile_wfosheet.get_all_values()}\n')
        # Read csv at path and convert to df
        localdatacsv = 'office_visits_log.csv'
        wfo_csv_file_path = os.path.join(tasks_name_directory, localdatacsv)
        wfocsv_df = pd.read_csv(wfo_csv_file_path)
        print(f'Local WFO csv:\n{wfocsv_df.columns.tolist()}\n{wfocsv_df.dtypes}\n{wfocsv_df}\n')

        # Set the index of the dataframe to the 'Date' column
        # Check if the date exists in the index
        wfocsv_df = wfocsv_df.set_index('Date')
        if date_val in wfocsv_df.index:
            print(f'Date exists in Local WFO csv\n')
            wfocsv_df.loc[date_val, 'Visit'] = visit_val
        else:
            print(f'Date does NOT exist in Local WFO csv\n')
            new_row = pd.DataFrame({'Visit': visit_val}, index=[date_val])
            wfocsv_df = pd.concat([wfocsv_df, new_row])
        
        # Reset the index of the dataframe and replace NaN values with ''
        wfocsv_df = wfocsv_df.reset_index()
        wfocsv_df = wfocsv_df.fillna('')

        # write the dataFrame to CSV file
        wfocsv_df.to_csv(wfo_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_wfosheet.update([wfocsv_df.columns.values.tolist()] + wfocsv_df.values.tolist())
        
        print(f'Updated WFO sheet!\n')

    def sizemeasurementlog(self, transcribedtext):
        print(f'Size Measurement Entry...\n')

        prompt = [{
            "role": "system", 
            "content": '''
                You will receive a user's transcribed speech and are to determine the date of request, the body part being measured and its size. 
                If date not mentioned assume it was today. If unit not mentioned assume it was inches.
                Group similar body parts into the following categories:
                1. Bicep: arm, arms, biceps, bicep 
                2. Chest: chest
                3. Waist: belly, hips, waist
                4. Thigh: leg, legs, thighs, thigh
                Example input: ok so my biceps are 14.25 inches
                Example output: today - Bicep - 14.25 in
                Following is the transcription:'''
            +
            transcribedtext
        }]
        temp = 0
        maxtokens = 256
        showlog = False

        datesize = gptObj.chatgptcall(prompt, temp, maxtokens, showlog)
        datesize = datesize.lower().title()

        date_val = genObj.get_date(datesize)
        bodypart_val = genObj.get_bodypart(datesize)
        size_val = genObj.get_weight(datesize)
        print(f'datesize: {datesize}\n date: {date_val} - body part: {bodypart_val} - size: {size_val}\n')

        # If error like "No XYZ found" then end function here
        if "no" in [date_val, bodypart_val, size_val]:
            print(f'Sizes sheet not updated.\n')
            return 
        
        # Open sheet in Sheets file
        datasheet = 'Sizes'
        dumpfile_sizesheet = sheetObj.routineSheetFile.worksheet(datasheet)
        # print(f'Current data in routineSheetFile weight sheet:\n{dumpfile_sizesheet.get_all_values()}\n')
        # Read csv at path and convert to df
        localdatacsv = 'sizes_log.csv'
        sizes_csv_file_path = os.path.join(tasks_name_directory, localdatacsv)
        sizecsv_df = pd.read_csv(sizes_csv_file_path)
        print(f'Local Sizes csv:\n{sizecsv_df.columns.tolist()}\n{sizecsv_df.dtypes}\n{sizecsv_df}\n')

        # Set the index of the dataframe to the 'Date' column
        # Check if the date exists in the index
        sizecsv_df = sizecsv_df.set_index('Date')
        if date_val in sizecsv_df.index:
            print(f'Date exists in Local Sizes csv\n')
            sizecsv_df.loc[date_val, bodypart_val] = size_val
        else:
            print(f'Date does NOT exist in Local Sizes csv\n')
            new_row = pd.DataFrame({bodypart_val: size_val}, index=[date_val])
            sizecsv_df = pd.concat([sizecsv_df, new_row])
        
        # Reset the index of the dataframe and replace NaN values with ''
        sizecsv_df = sizecsv_df.reset_index()
        sizecsv_df = sizecsv_df.fillna('')

        # write the dataFrame to CSV file
        sizecsv_df.to_csv(sizes_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_sizesheet.update([sizecsv_df.columns.values.tolist()] + sizecsv_df.values.tolist())

        print(f'Updated Sizes sheet!\n')

    def haircut(self):
        print(f'Haircut Enquiry...\n')
        
        # Open bank statement sheet in Sheets file
        datasheet = 'Data'
        dumpfile_datasheet = sheetObj.bankStatementSheetFile.worksheet(datasheet)
        # Convert this Google Sheets file output into a dataframe
        bankstatement_df = pd.DataFrame(dumpfile_datasheet.get_all_values())
        # Filter the dataframe based on string
        haircut_df = bankstatement_df[bankstatement_df['tag'].str.contains('Haircare')]
        # Show only 3 columns in df
        haircut_df = haircut_df[['Date']]
        # convert the 'dates' column to string and join the elements with line break
        dates_string = '\n'.join(df['Date'].astype(str).tolist())

        prompt = [{
            "role": "system", 
            "content": '''
                You are MIA, an AI desk robot that makes sarcastic jokes and observations. You have the personality of Chandler Bing from Friends and Barney Stinson from HIMYM. 
                You are to predict the exact date of the user's next haircut. You are provided data of the dates a user's hair was cut and based on the trend you determine the next exact date for their haircut. 
                If this date happens to be a weekday then pick the nearest weekend date. Provide only the final date with no details on the calculation. If a date cannot be determined reply to the user.
                The data:'''
            +
            dates_string
        }]
        temp = 0
        maxtokens = 24
        showlog = False

        nexthaircut_date = gptObj.chatgptcall(prompt, temp, maxtokens, showlog)
        print(f'{nexthaircut_date}\n')

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
        asyncio.run(convObj.saveconversation())
        
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
        asyncio.run(convObj.saveconversation())

        self.listen()

class Conversations:
    def __init__(self):
        self.conversation_context = []

        filename = os.path.join(convo_name_directory, 'convo.json')
        print(f'Conversation Filename: {filename}\n')

        try:
            past_conversations = json.load(open(filename))

            if any('user' in d['role'] for d in past_conversations):
                print('Loaded past conversations\n')

                summarized_prompt = system_context + summarize_context + str(past_conversations)
                print(f'Summarized_prompt: {summarized_prompt}\n')
                
                past_convo_summary = gptObj.chatgptcall([{"role": "system", "content": summarized_prompt},])
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": past_convo_summary}]
            elif any('assistant' in d['role'] for d in past_conversations):
                print('Assistant\'s past response loaded!')
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": past_conversations[1]['content']}]
            else:
                print('No past conversations to load from!')
                
                greeting = gptObj.chatgptcall([{"role": "system", "content": system_context},])
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": greeting}]
            
            print(f'Conversation_context: {self.conversation_context}\n')
            asyncio.run(self.saveconversation())

        except:
            pass

    async def saveconversation(self):
        print(f'ASYNC Saveconversation\n')
        filename = os.path.join(convo_name_directory, 'convo.json')
        print(f'Saveconversation Filename: {filename}\n')

        async with aiofiles.open(filename, "w") as outfile:
            # json.dump(self.conversation_context, outfile)
            await outfile.write(json.dumps(self.conversation_context))
        
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