import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)

    def FindFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
        best_detection = None
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                if draw:
                    cv.rectangle(img, bbox, (0, 255, 0), 2)
                if best_detection is None or detection.score[0] > best_detection[2][0]:
                    best_detection = (bbox, detection.score)
        
        return img, bboxes, best_detection

def main():
    cTime = 0
    pTime = 0
    
    detector = FaceDetector()
    cap = cv.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        img, bboxes, best_detection = detector.FindFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        if best_detection:
            bbox, score = best_detection
            cv.putText(img, f'{int(score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 1)
        
        cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
        cv.imshow("Image", img)
        cv.waitKey(10)

if __name__ == "__main__":
    main()
