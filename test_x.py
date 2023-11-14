import os
import base64
import time
import cv2
from PIL import Image
from openai import OpenAI
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

image_folder_path = r".\images"

def capture_image_from_webcam():
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

def encode_image(image_array):
    img = Image.fromarray(image_array)
    img = img.resize((360, 360))

    with BytesIO() as buffer:
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Delete all images
files = os.listdir(image_folder_path)
for file in files:
    file_path = os.path.join(image_folder_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Start image taking loop
while True:
    image_array = capture_image_from_webcam()
    if image_array is not None:
        encoded_string = encode_image(image_array)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
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

        print(response.choices[0])
    
    time.sleep(10)  # Wait for 5 seconds