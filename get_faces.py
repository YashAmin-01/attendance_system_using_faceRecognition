import cv2
import os
from datetime import datetime

def collect_images(name):
    cap = cv2.VideoCapture(0)

    faces = 0
    frames = 0
    max_faces = 20

    if os.path.exists(f'images/{name}'):
        print('Data for user already exists. Do you wish to replace data? ')
        response = input('Y / N ?: ')
        if response == 'Y' or response == 'y':
            files = os.listdir(f'images/{name}')
            for file in files:
                os.remove(os.path.join(f'images/{name}', file))
        else:
            return
    else:
        os.makedirs(f'images/{name}')

    while faces < max_faces:
        _, frame = cap.read()
        time_string = str(datetime.now().microsecond)

        if frames % 15 == 0:
            cv2.imwrite(os.path.join(f'images/{name}', f'{name}_{time_string}.jpg'), frame)
            faces += 1

        frames += 1
        cv2.imshow('Face capture', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

collect_images(input('Enter user name for image collection: '))