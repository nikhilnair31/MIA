import os
import struct
import pyaudio
import pvporcupine
from dotenv import main

main.load_dotenv()
access_key = os.getenv("PICO_ACCESS_KEY")

porcupine = None
pa = None
audio_stream = None

try:
    # porcupine = pvporcupine.create(keywords=["computer", "jarvis"])
    porcupine = pvporcupine.create(
        access_key=access_key,
        keywords=['picovoice', 'bumblebee']
    )

    pa = pyaudio.PyAudio()

    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Hotword Detected")
finally:
    porcupine.delete()