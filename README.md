# attendance_system_using_faceRecognition

1. get_faces.py
-run this file to register new user and create database of new user by clicking 20 images
-for better results try to change angle of face while facing the camera so that every image is different

2. prepare_dataset.py
-uses all images in 'images' folder to generate encodings by detecting the face landmarks
-creates a csv file with encodings for each image and adds a new column containing name of the user to whom the encoding belongs to

3. train.py
-uses encodings file as training dataset and trains an svm model for classification
-name of user is set as target variable for classification
-polynomial kernel used for transformation; degree of polynomial selected automatically for highest accuracy

4. record_attendance.py
-run this file to open camera and capture image by pressing spacebar
-image encoding is generated and passed to svm model for classification
-classified label is displayed for confirmation and if label is correct, new attendance record is generated for user with current time 
-attendance is logged in 'attendance.csv' file
