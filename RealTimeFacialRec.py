import os
import csv
import pyttsx3
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
        
def findEncodings(images):  #Identifies the faces
    EncodingsList=[]
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)
        if encode:
            EncodingsList.append(encode[0])
    return EncodingsList

EncodingsListKnown = findEncodings(images)

def capture_new_face(name):
    ret, img = cap.read()
    if ret:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encodings = fc.face_encodings(img_rgb)
        
        if encodings:
            # Saving the encoding and name
            new_encoding = encodings[0]
            EncodingsListKnown.append(new_encoding)
            class_names.append(name)
            
            img_path = f"{path}/{name}.jpg"
            cv.imwrite(img_path, img)
            print(f"New face added: {name}")
            
def mark_attendance(name):
    with open("Attendance.csv", "a") as f:
        writer = csv.writer(f)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, current_time])

def announce_name(name):
    engine = pyttsx3.init()
    engine.say(f"{name} recognized")
    engine.runAndWait()

# 0 is for built in Camera. Use 1, 2 or other numerals for external camera installed to your device. 
cap=cv.VideoCapture(0)

while True:
    success,img=cap.read()
    
    # Addin these 2 for better precaution and performance
    imgS=cv.resize(img,(0,0),None,0.25,0.25)
    imgS=cv.cvtColor(imgS,cv.COLOR_BGR2RGB)
    
    faceLoc = fc.face_locations(imgS)
    faceEncode=fc.face_encodings(imgS,faceLoc)
    
    recognized_faces = []
    
    for fe, fl in zip(faceEncode, faceLoc):
        matches = fc.compare_faces(EncodingsListKnown, fe)
        faceDis = fc.face_distance(EncodingsListKnown, fe)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
                name = class_names[matchIndex].upper()
                y1, x2, y2, x1 = fl
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                confidence = round((1 - faceDis[matchIndex]) * 100, 2)
                
                cv.putText(img, f"{name} ({confidence}%)", (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                mark_attendance(name)
                announce_name(name)
        
        #  Blurs unrecognized faces (privacy reasons)
        for fe, fl in zip(faceEncode, faceLoc):
            if not matches[matchIndex]:
                y1, x2, y2, x1 = fl
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                img[y1:y2, x1:x2] = cv.GaussianBlur(img[y1:y2, x1:x2], (99, 99), 30)


    sidebar_width = 200
    img_height = img.shape[0]
    img = cv.copyMakeBorder(img, 0, 0, 0, sidebar_width, cv.BORDER_CONSTANT, value=(50, 50, 50))

    # Names Sidebar
    y_offset = 30
    for i, name in enumerate(recognized_faces):
        cv.putText(img, name, (img.shape[1] - sidebar_width + 10, y_offset + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 
    
    cv.imshow("Live Feed",img)
    
    # Press 'n' to add a new face to the database
    if cv.waitKey(1) & 0xFF == ord('n'):
        new_name = input("Enter the name for the new face: ")
        capture_new_face(new_name)
        
    if cv.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()