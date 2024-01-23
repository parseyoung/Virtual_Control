import cv2
import time
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller
#########################
# cvzon version : 1.4.1 # using findPosition()
# pynput : using for realkeypad
#########################
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
final_text = ""
detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", " "]]

keyboard = Controller()

########## basic keypad ###########
def draw_all(img, button_list):
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt =8)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 30, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# ######### upgrade keypad ##########
# def draw_all(img, buttonList):
#     imgNew = np.zeros_like(img, np.uint8)
#     for button in buttonList:
#         x, y = button.pos
#         cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
#                                    20, rt =8)
#         cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
#                       (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgNew, button.text, (x + 40, y + 60),
#                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
#     out = img.copy()
#     alpha = 0.5
#     mask = imgNew.astype(bool)
#     print(mask.shape)
#     out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]    
#     return out

class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.clicked_time = 0

button_list = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        button_list.append(Button([100 * j + 50, 100 * i + 50], key))

while True:

    success, img = cap.read()
    
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lm_list, bbox_info = detector.findPosition(img)


    img = draw_all(img, button_list)

    if lm_list:
        for button in button_list:
            x, y = button.pos
            w, h = button.size

            if x < lm_list[8][0] < x + w and y < lm_list[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w+5, y + h+5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                l, _, _ = detector.findDistance(8, 12, img, draw=False)

                 # when clicked
                if l < 30 and time.time() - button.clicked_time > 0.4:
                    keyboard.press(button.text)
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    final_text += button.text
                    button.clicked_time = time.time()
                    sleep(0.2)

                    
    cv2.rectangle(img, (50, 350), (700, 450), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, final_text, (60, 430), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()