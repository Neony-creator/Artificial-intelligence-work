import cv2
import face_recognition
import numpy as np
import face_recognition as fcr
import os
from datetime import datetime

path = 'Image'
images =[]
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList :
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images) :
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','rt') as f :
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H/%M:%S')
            f.writelines(f'\n{name},{dtString}')



            markAttendance('Alexandre')



encodeListKnown = findEncodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = fcr.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2 .rectangle(img,(x1,y1),(x2,y2), (0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2), (0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)

#faceLoc = fcr.face_locations(imgMe)[0]
#encodeMe = fcr.face_encodings(imgMe)[0]
#cv2.rectangle(imgMe,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255, 0, 255),2)

#faceLocTest = fcr.face_locations(imgTest)[0]
#encodeTest = fcr.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]), (255, 0, 255),2)

#results = fcr.compare_faces([encodeMe], encodeTest)
#faceDis = fcr.face_distance([encodeMe], encodeTest)

#imgMe = fcr.load_image_file('Image/Photo cv (2).png')
#imgMe = cv2.cvtColor(imgMe, cv2.COLOR_BGR2RGB)
#imgTest = fcr.load_image_file('Image/Photo moche.jpg')
#imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
