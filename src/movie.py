from moviepy.editor import *
from datetime import datetime
import threading
import time
from datetime import timedelta
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import csv 
from multiprocessing import Process
#Import necessary libraries
from flask import Flask, render_template, Response

#Initialize the Flask app
statDir = '../static/'
templateDir = '../templates/'
# initialize a flask object
app = Flask(__name__,static_folder=statDir,
            template_folder=templateDir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start = datetime.now()
#clip = VideoFileClip("demo2.mp4").subclip(0,12)
#clip = clip.resize(.6)
 
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
# If you want to train the same model or try other models, go for this
start = datetime.now()
# emotions will be displayed on your face from the webcam feed
model.load_weights('model.h5')

cap = cv2.VideoCapture(0)

def ML():
    global model
    global start
    global cap
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    # start the webcam feed
    f = open("out.csv","w+")
    writer = csv.writer(f)

    
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        scale_percent = 40 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        if not ret:
            break
        #lbpcascade_frontalface_improved also works
        facecasc = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            if(emotion_dict[maxindex]):
                writer.writerow([datetime.now()-start,emotion_dict[maxindex],])
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow('Video', cv2.resize(frame,(800,480),interpolation = cv2.INTER_CUBIC))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    f.close()
    cap.release()
    cv2.destroyAllWindows()

def preview():
    clip.preview(fps = 20,fullscreen = False)

def preview2():
    vid = cv2.VideoCapture("demo2.mp4")
    ret, frame = vid.read()
    scale_percent = 20 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
# resize image

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
    vid.release()
    cv2.destroyAllWindows()
def printTime():
    start = datetime.now()
    diff = datetime.now()-start
    while diff < timedelta(seconds=5):
        diff = datetime.now()-start
        print(diff)
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(ML(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(preview2(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    """p1 = Process(target=ML)
    p2 = Process(target=preview)
    
    p2.start()
    p1.start()
    p2.join()
    p1.join()"""
    app.run(debug=True)
    print("Done")
#thread2 = threading.Thread(target=ML)
#thread1 = threading.Thread(target=preview)


# Start new Threads
#thread2.start()
#thread1.start()


# Add threads to thread list
#thread1.join()
#thread2.join()

# Wait for all threads to complete