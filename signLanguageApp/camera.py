import cv2
import numpy as np
import math
from keras.models import load_model
from typing import List, Tuple, Union
import mediapipe as mp
import threading
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation

detector = HandDetector(detectionCon=0, maxHands=1)

# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = load_model(
    "D:\django projects\SignLanguageDetection\media\\aslAlphabet_valacc_8934.h5")

segmentor = SelfiSegmentation()

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
           'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

    def get_frame(self):
        success, img = self.video.read()

        try:

            hands = detector.findHands(img, draw=False)  # with draw

            if hands:
                hand1 = hands[0]
                x, y, w, h = hand1["bbox"]  # Bounding box info x,y,w,h
                cv2.rectangle(img, (x-50, y-50),
                              (x+w+50, y+h+30), (0, 255, 0), 2)
                hand = img[y-50:y+h+30, x-50:x+w+50]
                hand = segmentor.removeBG(hand, (0, 0, 255), threshold=0.1)
                hand_resize = cv2.resize(hand, (200, 200))
                hand_rescale = hand_resize/255.0
                hand_reshape = np.reshape(hand_rescale, (1, 200, 200, 3))
                # hand_stack = np.vstack([hand_reshape])
                out = model.predict(hand_reshape)
                pos = np.argmax(out[0])
                # print(out, letters[pos])
                if type(x) is np.ndarray:
                    x = int(np.max(x[0]))
                if type(y) is np.ndarray:
                    y = int(np.max(y[0]))
                if type(w) is np.ndarray:
                    w = int(np.max(w[0]))
                if type(h) is np.ndarray:
                    h = int(np.max(h[0]))

                # print(x, y, w, h)

                cv2.rectangle(img, (x , y + h + 25 ),
                              (x + 100, y + h - 10), (0, 0, 0), -1)
                cv2.putText(img, letters[pos], (x + 20, y + h + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        except:
            pass
        ret, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()
