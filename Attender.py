import os
import datetime
import cv2 as cv
import numpy as np
import face_recognition as fc

path = "Pics"
myList = os.listdir(path)
images = []
class_names = []
EncodingsList = []

for cl in myList:
    curImg = cv.imread(f"{path}/{cl}")
    if curImg is not None:
        images.append(curImg)
        class_names.append(os.path.splitext(cl)[0])

def Attendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.datetime.now()
            date=now.strftime("%H:%M:%S ")
            f.writelines(f"\n{name},{date}")

def FindEncodings(images):
    EncodingsList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)
        if encode:
            EncodingsList.append(encode[0])
    return EncodingsList

EncodingsListKnown = FindEncodings(images)

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    faceLoc = fc.face_locations(imgS)
    faceEncode = fc.face_encodings(imgS, faceLoc)

    for fe, fl in zip(faceEncode, faceLoc):
        matches = fc.compare_faces(EncodingsListKnown, fe)
        faceDis = fc.face_distance(EncodingsListKnown, fe)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = class_names[matchIndex].upper()
            y1, x2, y2, x1 = fl
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
            Attendance(name)

    width = int(img.shape[1] * 1.25)
    height = int(img.shape[0] * 1.25)
    dim = (width, height)
        
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    cv.imshow("Face-Recognizer", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  
