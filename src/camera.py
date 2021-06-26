from moviepy.editor import *
from datetime import datetime
import threading

import time
from datetime import timedelta
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import csv 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Process
from imutils.video import VideoStream
from imutils.video import FPS
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from flask import Flask, render_template, Response
import pdb

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.load_weights('model.h5')
        self.start = start = datetime.now()
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        
        # emotions will be displayed on your face from the webcam feed
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        # start the webcam feed
        f = open("out.csv","w+")
        writer = csv.writer(f)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            if(emotion_dict[maxindex]):
                writer.writerow([datetime.now()-self.start,emotion_dict[maxindex],])
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.     LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
