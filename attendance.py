# libraries imported
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path of the images folder
path = 'ImageAttendance'

# List of all the images and names of all the images
images = []
classNames = []

# Grab list of images in ImageAttendance
myList = os.listdir(path)
print(myList)

# Reading each image and its name in list images and classNames
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# EVERY IMAGE HAS A ENCODING WHICH IS COMPARED AND USED FOR RECOGNITION

# Function to compute encoding of each image
def findEncodings(images):
    encodeList = []
    for img in images:
        # converts every image from BGR to RBG
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Finds the encoding of every image
        encode = face_recognition.face_encodings(img)[0]

        # Adding encodings in a list
        encodeList.append(encode)

    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# List of already encoded images in ImageAttendance
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initializing the Webcam
cap = cv2.VideoCapture(0)

# Capturing webcam image
while True:
    success, img = cap.read()

    # Resizing and changing webcam image from BGR to RBG
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Location of the current frame captured
    facesCurFrame = face_recognition.face_locations(imgS)

    # Encoding the webcam image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Coparing WebCam image and stored Image
    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):

        # Matches the two Images and also gives the diffrence of their Encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Stores the minimum encoding distance match
        matchIndex = np.argmin(faceDis)

        # Creates the rectangle on the webcam image
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceloc

            # As we resized the image adjusting the rectangle accordingly
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Creates the rectangle and shows the img name
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 +8, y2 -8), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

            markAttendance(name)




    # Shows the webcam image
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
