import os
import openai
from dotenv import main

main.load_dotenv()

OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

openai.api_key = OPENAI_API_KEY

def whispercall(filename):
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

whispercall(r'.\audio\20231103160717.wav')