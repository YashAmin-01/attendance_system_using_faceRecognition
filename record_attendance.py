import pandas as pd
import numpy as np
import cv2
from datetime import datetime
import face_recognition
import joblib
import os
from warnings import filterwarnings

filterwarnings('ignore')

def generate_encodings(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_loc)
    return encodings

def click_picture():
    cap = cv2.VideoCapture(0)
    global frame

    while True:
        _, frame = cap.read()
        cv2.imshow('captured face', frame)

        if cv2.waitKey(1) == ord(' '):
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            continue

    return frame

def record_attendance(name, direction):
    if 'attendance.csv' not in os.listdir():
        attendance = pd.DataFrame(columns=['Name','Date','Time','Entry/Exit'])
        attendance.to_csv('attendance.csv')

    attendance_log = pd.read_csv('attendance.csv', index_col=[0])
    date = datetime.now()

    time = date.strftime('%X')
    day = f'{date.day}/{date.month}/{date.year}'

    attendance = pd.DataFrame([[name, day, time, direction]], columns=['Name','Date','Time','Entry/Exit'])
    attendance_log = pd.concat([attendance_log, attendance], ignore_index=True)

    attendance_log.to_csv('attendance.csv')

def detect_face(encodings):
    model = joblib.load('trained_models/facerec_model.model')
    prob = model.predict_proba(encodings)
    
    if prob.max() > 0.7:
        return model.classes_[np.argmax(prob)]
    else:
        return 'unknown'
    

# driver code
# image = face_recognition.load_image_file('../sagar/sagar.jpg')
# encodings = generate_encodings(image)
# name = detect_face(encodings)

print('1. Entry time')
print('2. Exit time')
direction_input = int(input('Select option to record Entry/Exit time: '))
direction = ''

if direction_input == 1:
    direction = 'Entry'
elif direction_input == 2:
    direction = 'Exit'
else:
    print('\nInvalid input\n')
    print('Closing application...')

if direction_input == 1 or direction_input == 2:
    image = click_picture()
    encodings = generate_encodings(image)
    name = detect_face(encodings)

    if name == 'unknown':
        print('Captured face not in database...')
        print('Please register and try again!')
    else:
        print('face detected...')
        print(f'Name: {name}')
        confirmation = input(f'Are you sure you want to continue recording attendance for {name}?(y/n): ')

        if confirmation == 'y' or confirmation == 'Y':
            record_attendance(name, direction)
            print(f'\n{direction} time has been recorded for {name}...')
        elif confirmation == 'n' or confirmation == 'N':
            print('Please try face capture again!')
        else:
            print('\nInvalid input\n')
            print('Closing application...')