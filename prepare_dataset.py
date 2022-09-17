import os
import pandas as pd
import cv2
import face_recognition
from tqdm import tqdm

def generate_encodings(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_loc)

    return encodings

df = pd.DataFrame()

print()
print('generating encodings...')

for user in tqdm(os.listdir('images')):
    user_path = os.path.join('images', user)
    
    for image_name in os.listdir(user_path):
        image_path = os.path.join(user_path, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = generate_encodings(image)

        data = pd.DataFrame([encodings[0]])
        data['class'] = user
        df = pd.concat([df, data], ignore_index=True)

df.to_csv('encodings.csv')
print()
print('saved encodings to csv file.')
