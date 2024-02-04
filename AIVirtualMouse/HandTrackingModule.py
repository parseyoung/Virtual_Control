import cv2
import time
import numpy as np
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                       self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                if xList and yList:  # Check if xList and yList are not empty
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax
                    if draw:
                        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox
    
    # def findPosition(self, img, handNo=0, draw=True):
    #     lmList = []
    #     if self.results.multi_hand_landmarks:
    #         myHand = self.results.multi_hand_landmarks[handNo]
    #         for id, lm in enumerate(myHand.landmark):
    #             h, w, c = img.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             lmList.append([id, cx, cy])
    #             if draw:
    #                 cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
    #     return lmList


    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


if __name__ == "__main__":
    wCam, hCam = 640, 488

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0

    detector = handDetector(detectionCon=0.5, maxHands=2) #maxHand = número de mãos detectadas

    bar = 0
    barper = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=False) #detecta a mão

        lmList = detector.findPosition(img, draw=False)

        if len(lmList[0]) != 0:
            x1, y1 = lmList[0][4][1], lmList[0][4][2]
            x2, y2 = lmList[0][8][1], lmList[0][8][2]
            cx, cy = (x1 + x2) //2, (y1 + y2) //2 #acha a média dos pontos

            cv2.circle(img, (x1,y1), 10, (255, 0, 255), cv2.FILLED) #desenha bola
            cv2.circle(img, (x2,y2), 10, (255, 0, 255), cv2.FILLED) #desenha bola
            cv2.line(img, (x1,y1),(x2,y2), (255,0,255),3) #desenha linha entre os dedos
            cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED) #desenha bola

            length = math.hypot(x2 - x1, y2 - y1)

            # Hand range 50 - 300
            bar = np.interp(length, [50, 300], [400, 150])
            barper = np.interp(length, [50, 300], [0, 150])
            if length < 50:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED) #desenha bola

        cv2.rectangle(img, (50, int(bar)), (85, 400), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, f'{int(barper)} ', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 255), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
       
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == 27:  # press esc to exit
            break

    cv2.destroyAllWindows()
    cap.release()
