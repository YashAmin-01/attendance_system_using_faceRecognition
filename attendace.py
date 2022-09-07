import cv2
import time
import numpy as np
import face_recognition
from PIL import ImageGrab
import os
from datetime import datetime
path = 'images'
images=[]
class_names=[]
my_list = os.listdir(path)
print(my_list)

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
def captureScreen(bbox=(300,300,690+300,530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr

for cl in my_list:
    current_img=cv2.imread(f'{path}/{cl}')
    images.append(current_img)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(images):
    encode_list=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = face_recognition.face_encodings(img)[0]
        encode_list.append(encode_img)
    return encode_list

def mark_attendance(name):
    with open('attendance.csv','r+') as f:
        data_list = f.readlines()
        name_list=[]
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encode_list_for_known=find_encodings(images)
print('encoding completed')

cap = cv2.VideoCapture(0)

# while True:
success,img=cap.read()
# img = captureScreen()
image_small=cv2.resize(img,(0,0),None,0.25,0.25)
image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

faces_current_frame= face_recognition.face_locations(image_small)
encode_img_current_frame = face_recognition.face_encodings(image_small,faces_current_frame)

for encode_face,faceloc in zip(encode_img_current_frame,faces_current_frame):
    matches = face_recognition.compare_faces(encode_list_for_known,encode_face)
    face_distance=face_recognition.face_distance(encode_list_for_known,encode_face)
    #print(face_distance)
    match_index=np.argmin(face_distance)

    if matches[match_index]:
        name = class_names[match_index].upper()
        print(name)
        y1,x2,y2,x1=faceloc
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_ITALIC,1,(255,255,255),2)
        mark_attendance(name)

cv2.imshow('webcam',img)
cv2.waitKey(0)










'''faceloc = face_recognition.face_locations(ravi_img)[0]
encode_ravi=face_recognition.face_encodings(ravi_img)[0]
cv2.rectangle(ravi_img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(ravi_test)[0]
encode_ravi_test=face_recognition.face_encodings(ravi_test)[0]
cv2.rectangle(ravi_test,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

face_gp = face_recognition.face_locations(gp_img)[0]
encode_gp=face_recognition.face_encodings(gp_img)[0]
cv2.rectangle(gp_img,(face_gp[3],face_gp[0]),(face_gp[1],face_gp[2]),(255,0,255),2)'''


