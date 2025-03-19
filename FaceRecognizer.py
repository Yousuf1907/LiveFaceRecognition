import cv2 as cv
import numpy as np
import face_recognition as fc
    
img=fc.load_image_file("Pics/Yousuf.jpeg")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

faceLoc=fc.face_locations(img)[0]
encoded=fc.face_encodings(img)[0]

img_test=fc.load_image_file("Pics/Yousuf.jpg")
img_test=cv.cvtColor(img_test,cv.COLOR_BGR2RGB)

faceLoc_test=fc.face_locations(img_test)[0]
encoded_test=fc.face_encodings(img_test)[0]

cv.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),3)
cv.rectangle(img_test,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),3)

results=fc.compare_faces([encoded],encoded_test)
faceDis=fc.face_distance([encoded],encoded_test)
print(results,faceDis)

cv.putText(img_test,f"{results}: {faceDis[0]}",(50,50),cv.FONT_HERSHEY_DUPLEX,1,(50,50,50),2)

cv.imshow("Yousuf",img)
cv.imshow("Yousuf Test",img_test)

cv.waitKey(0)