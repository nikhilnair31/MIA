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
import pinecone
import pvporcupine
import pandas as pd
from dotenv import main
from tqdm.autonotebook import tqdm
from datetime import datetime, timedelta
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from elevenlabs import set_api_key, generate, play, stream
from google.oauth2.service_account import Credentials
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

# Path for audio and conversations
audio_name_directory = r'.\audio'
convo_name_directory = r'.\conversations'
tasks_name_directory = r'.\tasks'

# Conversations
summarize_context = "The following is your context of the previous conversation with the user: "
system_context = """
You are MIA, an AI desk robot that makes sarcastic jokes and observations. 
You have the personality of JARVIS, Chandler Bing and Barney Stinson combined. 
Do not patronize and just treat like you would a friend.
Keep responses short. Initiate with a greeting.
"""
# endregion

# region Class
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

class General:
    async def create_create_file_at_path(self, path, filename, headerlist):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            print(f"File {full_path} already exists!")
        else:
            start_date = datetime(datetime.now().year, 1, 1)
            end_date = start_date + timedelta(days=365) - timedelta(days=1)
            date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
            async with aiofiles.open(full_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headerlist)
                for date in date_list:
                    writer.writerow([date.strftime('%d-%m-%Y')])
            print(f"Created file {full_path}")

    def conversation_to_text(self, text):
        plaintext = ''
        for conv in text:
            if 'system' in conv['role']:
                plaintext += conv['content'] + '\n'
            if 'assistant' in conv['role']:
                plaintext += 'assistant: ' + conv['content'] + '\n'
            if 'user' in conv['role']:
                plaintext += 'user: ' + conv['content'] + '\n'
        print(f'conversation_to_text:\n{plaintext}\n')
        return plaintext

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

    # Basic version that uses syntax and regex to pull date
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

    # Uses GPT-4 to pull exact date from text by passing it today's date
    def get_date_gpt(self, text):
        today = datetime.today().date()
        prompt = [
            {
                "role": "system", 
                "content": 
                '''Extract the exact date from a string containing allusions to a date. Today is '''
                +today.strftime('%A')+
                ''' '''
                +today.strftime('%d-%m-%Y')+
                '''. Only give the final output date with no other text. 
                    Example:
                    Input: last friday Output: date of the last Friday before today's date. 
                    Input: 4 days ago Output: date that was 4 days ago from today's date. 
                    Input: yesterday Output: date that was yesterday's date.
                '''
            },
            {
                "role": "user", 
                "content": text
            }
        ]
        model = 'gpt-4'
        temp = 0
        maxtokens = 16
        showlog = False
        pattern = r"\d{2}-\d{2}-\d{4}"

        datetext = gptObj.gpt_chat_call(text=prompt, model=model, temp=temp, maxtokens=maxtokens, showlog=showlog)
        
        match = re.search(pattern, datetext)
        if match:
            date_str = match.group()
            date_obj = datetime.strptime(date_str, '%d-%m-%Y').strftime('%d-%m-%Y')
            print(f'prompt: {prompt}\n')
            return date_obj

class GPT:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
    
    # def gpt_completion_call(self, text, engine="text-davinci-003", temp=0.7, maxtokens=256, showlog=True):
    #     print('Making GPT-3 request..\n')

    #     response = openai.Completion.create(
    #         engine=engine,
    #         prompt=text,
    #         temperature=temp,
    #         max_tokens=maxtokens
    #     )
    #     response_text = response["choices"][0].text.lower()
    #     if showlog:
    #         print(f'{"-"*50}\nGPT-3 Response:\n{response_text}\n{"-"*50}\n')
        
    #     audioObj.speak(response_text)
        
    #     return response_text

    def gpt_chat_call(self, text, model="gpt-3.5-turbo", temp=0.7, maxtokens=256, showlog=True):
        print('Making ChatGPT request..\n')

        response = openai.ChatCompletion.create(
            model=model,
            messages=text,
            temperature=temp,
            max_tokens=maxtokens
        )
        response_text = response["choices"][0]["message"]["content"].lower()
        if showlog:
            print(f'{"-"*50}\nChatGPT Response:\n{response_text}\n{"-"*50}\n')

        audioObj.speak(response_text)
        
        return response_text

    def whispercall(self, filename, showlog=True):
        print('Making Whisper request..\n')

        audio_file= open(filename, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en"
            # options
        )
        transcript_text = transcript["text"]

        if showlog:
            print(f'{"="*50}\nTranscript: {transcript_text}\n{"="*50}\n')
        
        return transcript_text

class Tasks:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index("mia")
        self.vectorstore = Pinecone(self.index, self.embeddings, "text")

    async def taskfilegen(self):
        await asyncio.gather(
            genObj.create_create_file_at_path(tasks_name_directory, "weight_log.csv", ['Date', 'Weight (kg)']),
            genObj.create_create_file_at_path(tasks_name_directory, "sizes_log.csv", ['Date', 'Chest (in)', 'Waist (in)', 'Bicep (in)', 'Thigh (in)']),
            genObj.create_create_file_at_path(tasks_name_directory, "selfcare_log.csv", ['Date', 'ShowerTime', 'Routine']),
            genObj.create_create_file_at_path(tasks_name_directory, "office_visits_log.csv", ['Date', 'Visited?']),
            # genObj.create_create_file_at_path(tasks_name_directory, "test_log.csv", ['Date', 'ABC'])
        )

    def checkifrequesttype(self, transcribedtext):
        requesttype = gptObj.gpt_chat_call(
            text=[{
                "role": "system", 
                "content": '''
                    You will receive a user's transcribed speech and are to determine if the user is requesting for some information or the completion of a task. If he is then return Y, else return N. 
                    Following is the transcription:'''
                +
                transcribedtext
            }],
            model='gpt-4',
            temp=0
        )
        
        convContinue = None
        if 'y' in requesttype:
            print(f'Request exists.\n')

            query = gptObj.gpt_chat_call(
                text=[{
                    "role": "system", 
                    "content": '''
                        You will receive a user's transcribed speech and are to reword it to extract the topic discussed to feed into another system as a prompt. 
                        Example: 
                        Transcript: Nice MIA but just tell me about the energy drink I'd heard about yesterday
                        Response: energy drink
                        Following is the transcription:'''
                    +
                    transcribedtext
                }],
                model='gpt-4',
                temp=0
            )
            print(f'query: {query}\n')
            request = gptObj.gpt_chat_call(
                text=[{
                    "role": "system", 
                    "content": '''
                        You will receive a user's transcribed speech and are to reword it to extract the task requested. 
                        Example:
                        Transcript: i heard something about gravity and quantum physics so could you just summarize that for me?
                        Response: summarize the transcript above
                        Following is the transcription:'''
                    +
                    transcribedtext
                }],
                model='gpt-4',
                temp=0
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

        dateandweighttext = gptObj.gpt_chat_call(text=prompt, temp=temp, maxtokens=maxtokens, showlog=showlog)
        dateandweighttext = dateandweighttext.lower()
        
        # Use helper functions from General class to pull date and weight values
        date_val = genObj.get_date_gpt(dateandweighttext)
        weight_val = genObj.get_weight(dateandweighttext)
        print(f'date: {date_val} - weight: {weight_val}\n')

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

        # Set the index of the dataframe to the 'Date' column
        # Check if the date exists in the index
        # If yes then replace weight value, if not insert the date and weight value
        weightcsv_df = weightcsv_df.set_index('Date')
        if date_val in weightcsv_df.index:
            print(f'Date exists in Local weight csv\n')
            weightcsv_df.loc[date_val, 'Weight  '] = weight_val
        else:
            print(f'Date does NOT exist in Local weight csv\n')
            new_row = pd.DataFrame({'Weight  ': weight_val}, index=[date_val])
            weightcsv_df = pd.concat([weightcsv_df, new_row])
        
        # Reset the index of the dataframe and replace NaN values with ''
        weightcsv_df = weightcsv_df.reset_index(drop=False)
        weightcsv_df = weightcsv_df.fillna('')

        # write the dataFrame to CSV file
        weightcsv_df.to_csv(weight_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_weightsheet.update([weightcsv_df.columns.values.tolist()] + weightcsv_df.values.tolist())
        
        print(f'Updated Weight sheet!\n')
        return False
    
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

        datetimeshower = gptObj.gpt_chat_call(text=prompt, temp=temp, maxtokens=maxtokens, showlog=showlog)
        datetimeshower = datetimeshower.lower()

        date_val = genObj.get_date_gpt(datetimeshower)
        time_val = genObj.get_morningevening_etc(datetimeshower)
        haircare_val = genObj.get_shower_product_combo(datetimeshower)
        print(f'date: {date_val} - time: {time_val} - haircare: {haircare_val}\n')

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

        # Set the index of the dataframe to the 'Date' column
        # Check if the date exists in the index
        showercsv_df = showercsv_df.set_index('Date')
        if date_val in showercsv_df.index:
            print(f'Date exists in Local shower csv\n')
            showercsv_df.loc[date_val, 'Time'] = time_val
            showercsv_df.loc[date_val, 'Haircare'] = haircare_val
        else:
            print(f'Date does NOT exist in Local shower csv\n')
            new_row = pd.DataFrame({'Time': time_val, 'Haircare': haircare_val}, index=[date_val])
            showercsv_df = pd.concat([showercsv_df, new_row])
        
        # Reset the index of the dataframe and replace NaN values with ''
        showercsv_df = showercsv_df.reset_index(drop=False)
        showercsv_df = showercsv_df.fillna('')

        # write the dataFrame to CSV file
        showercsv_df.to_csv(shower_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_showersheet.update([showercsv_df.columns.values.tolist()] + showercsv_df.values.tolist())
        
        print(f'Updated Shower sheet!\n')
        return False

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

        dateofficevisit = gptObj.gpt_chat_call(text=prompt, temp=temp, maxtokens=maxtokens, showlog=showlog)
        dateofficevisit = dateofficevisit.lower()

        date_val = genObj.get_date_gpt(dateofficevisit)
        visit_val = genObj.get_wfo_visit(dateofficevisit)
        print(f'date: {date_val} - visit: {visit_val}\n')

        # If error like "No XYZ found" then end function here
        if "no" in [date_val, visit_val]:
            print(f'WFO sheet not updated.\n')
            return 
        
        # Open sheet in Sheets file
        datasheet = 'Office Visits'
        dumpfile_wfosheet = sheetObj.routineSheetFile.worksheet(datasheet)
        # Read csv at path and convert to df
        localdatacsv = 'office_visits_log.csv'
        wfo_csv_file_path = os.path.join(tasks_name_directory, localdatacsv)
        wfocsv_df = pd.read_csv(wfo_csv_file_path)

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
        wfocsv_df = wfocsv_df.reset_index(drop=False)
        wfocsv_df = wfocsv_df.fillna('')

        # write the dataFrame to CSV file
        wfocsv_df.to_csv(wfo_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_wfosheet.update([wfocsv_df.columns.values.tolist()] + wfocsv_df.values.tolist())
        
        print(f'Updated WFO sheet!\n')
        return False

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

        datesize = gptObj.gpt_chat_call(text=prompt, temp=temp, maxtokens=maxtokens, showlog=showlog)
        datesize = datesize.lower().title()

        date_val = genObj.get_date_gpt(datesize)
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
        # Read csv at path and convert to df
        localdatacsv = 'sizes_log.csv'
        sizes_csv_file_path = os.path.join(tasks_name_directory, localdatacsv)
        sizecsv_df = pd.read_csv(sizes_csv_file_path)

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
        sizecsv_df = sizecsv_df.reset_index(drop=False)
        sizecsv_df = sizecsv_df.fillna('')

        # write the dataFrame to CSV file
        sizecsv_df.to_csv(sizes_csv_file_path, index=False)
        # write the dataFrame to Sheet file
        dumpfile_sizesheet.update([sizecsv_df.columns.values.tolist()] + sizecsv_df.values.tolist())

        print(f'Updated Sizes sheet!\n')
        return False

    def haircut(self):
        print(f'Haircut Enquiry...\n')
        
        # Open bank statement sheet in Sheets file
        dumpfile_datasheet = sheetObj.routineSheetFile.worksheet('Full Dump')
        # Convert this Google Sheets file output into a dataframe
        data = dumpfile_datasheet.get_all_values()
        bankstatement_df = pd.DataFrame(data[1:], columns=data[0])
        # Filter the dataframe based on string
        haircut_df = bankstatement_df[bankstatement_df['tag'].str.contains('haircare')]
        # Show only 3 columns in df
        haircut_df = haircut_df[['Date']]
        # convert the 'dates' column to string and join the elements with line break
        dates_string = '\n'.join(haircut_df['Date'].astype(str).tolist())

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
        maxtokens = 256
        showlog = False

        nexthaircut_date = gptObj.gpt_chat_call(text=prompt, model='gpt-4', temp=temp, maxtokens=maxtokens, showlog=showlog)
        print(f'{nexthaircut_date}\n')

        # Append GPT's response to the conversation
        convObj.conversation_context.append({"role": "assistant", "content": nexthaircut_date})
        # Save the current conversation
        asyncio.run(convObj.saveconversation())

        return True

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
        print(f'Listening...\n')

        self.elapsed_time = 0
        self.last_audio_time = time.time()
        while True:
            pcm = self.stream.read(self.porcupine.frame_length)

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
                        print("Timed out!\n")
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

        convObj.conversation_context.append({"role": "user", "content": transcribe_output})
        asyncio.run(convObj.saveconversation())
        
        doestaskhaveresponse = taskObj.checkifrequesttype(transcribe_output)
        
        if doestaskhaveresponse is False:
            self.response()
        else:
            self.listen()

    def response(self):
        print('Responding..\n')

        gptresponse = gptObj.gpt_chat_call(text = convObj.conversation_context)
        
        # If the phrase 'ai language model' shows up in a response then revert to GPT-3 to get a new response 
        # if 'ai language model' in gptresponse.lower():
        #     plaintext = genObj.conversation_to_text(text = convObj.conversation_context)
        #     gptresponse = gptObj.gpt_completion_call(text = plaintext, engine = "text-davinci-003")

        convObj.conversation_context.append({"role": "assistant", "content": gptresponse})
        asyncio.run(convObj.saveconversation())

        self.listen()

    def speak(self, text):
        if SPEAK_FLAG:
            audio = generate(
                text=response_text,
                voice="Bella", model="eleven_monolingual_v1",
                stream=True
            )
            stream(audio)
        else:
            pass

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
                
                past_convo_summary = gptObj.gpt_chat_call(text = [{"role": "system", "content": summarized_prompt},])
                self.conversation_context = [{"role": "system", "content": system_context}, {"role": "assistant", "content": past_convo_summary}]
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
# sheetObj = Sheets()
convObj = Conversations()
audioObj = Audio()

audioObj.listen()
# endregion