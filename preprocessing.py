import cv2
import numpy as np
import face_recognition
import os



img = face_recognition.load_image_file(f"images/sagar.jpg")
cv2.imshow('image', img)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('image_RGB', img_rgb)

faceloc = face_recognition.face_locations(img_rgb)[0]
print(faceloc)
cv2.rectangle(img_rgb, (faceloc[3],faceloc[0]), (faceloc[1],faceloc[2]), (255,0,255), 2)
cv2.imshow('image_RGB', img_rgb)
# cv2.waitKey(0)

encoding = face_recognition.face_encodings(img_rgb)[0]
print(encoding)
# print(face_recognition.face_landmarks(img_rgb))
