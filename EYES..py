# region Packages
import os
import cv2
import time
import base64
import keyboard
from PIL import Image
from io import BytesIO
from openai import OpenAI
from pynput import keyboard
from dotenv import load_dotenv
# endregion

# region Vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PIC_RATE = int(os.getenv("PIC_RATE"))

PAUSED = False

image_folder_path = r".\images"

system_prompt = "You are MIA an AI companion. You are viewing the world through the user's device. ONLY make short factual observations."
# endregion

# region Class
class EYES():
    # Setup objects and APIs
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()

        self.clearfolders()

    # region General
    def clearfolders(self):
        # Delete all images
        files = os.listdir(image_folder_path)
        for file in files:
            file_path = os.path.join(image_folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    def on_press(self, key):
        global PAUSED
        try:
            k = key.char 
        except:
            k = key.name
        if k == 'r':
            PAUSED = not PAUSED
            print(f'EYES has been {"PAUSED" if PAUSED else "UNPAUSED"}\n')

    def on_release(self, key):
        if key == keyboard.Key.esc:      
            return False
    # endregion

    # region Image
    def capture_image_from_webcam(self):
        cap = cv2.VideoCapture(0)  # 0 is typically the default camera
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            return None
        cap.release()
        
        # Save the captured frame to disk
        timestamp = time.strftime('%Y%m%d%H%M%S')
        filename = os.path.join(image_folder_path, '{}.jpg'.format(timestamp))
        cv2.imwrite(filename, frame)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB format

    def encode_image(self, image_array):
        img = Image.fromarray(image_array)
        img = img.resize((360, 360))

        with BytesIO() as buffer:
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def start_watching(self):
        print(f'EYES is Watching...\n')

        global PAUSED
        self.last_audio_time = time.time()
        self.elapsed_time = 0

        while True:
            image_array = self.capture_image_from_webcam()
            if image_array is not None and not PAUSED:
                encoded_string = self.encode_image(image_array)

                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": system_prompt},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                # {"type": "text", "text": "What’s in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        # "url": "https://raw.githubusercontent.com/davideuler/awesome-assistant-api/main/images/512x512.jpg"
                                        "url": f"data:image/jpeg;base64,{encoded_string}",
                                        "detail": "low"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=100,
                )
                print(f'Response:{response.choices[0].message.content}\nUsage:\n{response.usage.model_dump()}\n') 
            
            time.sleep(PIC_RATE)
    # endregion
# endregion

eyes = EYES()
eyes.start_watching()