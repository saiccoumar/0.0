from pytube import YouTube
import numpy as np
import os
import time
import cv2
import moviepy.editor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# url = 'https://www.youtube.com/watch?v=eabrae0uPO8'

current_directory = os.getcwd()
def getVid(url):
    yt = YouTube(url)
    while True: 
        try:
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(filename = 'video')
            if (current_directory+'/video.mp4'):
                break
        except:
            print('video could not be found')

def recordWebCam():
    vid = cv2.VideoCapture(0)
    vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter("webcam.mp4", vid_cod, 20.0, (850,480))
    while(True):
      
    # Capture the video frame
    # by frame
        ret, frame = vid.read()
        frame = cv2.resize(frame, (850,480))
    # Display the resulting frame
        cv2.imshow('frame', frame)
        output.write(frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def convertToFrames(video):
    vidcap = cv2.VideoCapture(video+'.mp4')
    timestamps = [vidcap.get(cv2.CAP_PROP_POS_MSEC)]
    # print(timestamps)
    success,image = vidcap.read()
    count = 0
    
    while success:
        timestamp = vidcap.get((cv2.CAP_PROP_POS_MSEC))
        timestamps.append(timestamp)
        hours,minutes,seconds = convert(timestamp)
        print("Hours: " + hours + " Minutes: "+ minutes+" Seconds: "+ seconds)
        timeIndex = hours + minutes + seconds
        cv2.imwrite("folder/"+video+"frame%s.jpg" % timeIndex, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def convert(milliseconds):
    seconds = milliseconds / 1000 
    hours = seconds // 3600 
    seconds %= 3600
    mins = seconds // 60
    seconds %= 60
    seconds = round(seconds,2)
    return str(hours), str(mins), str(seconds)

def removeFromString(text1):
    text = text1
    arr = ["'","/",",","#","<",">","$","+","%","!","`","&",'*',"|","\\","{","}","?",'"',"=",":","@"," ","."]
    for i in arr:
        if i in text:
            text = text.replace(i,'_')
    return(text)

def countdown(t):
    
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
      
    yield "finished"

def ML(duration):
    
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
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    # countdown(duration)
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(video+'.mp4')
    vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter("converted.mp4", vid_cod, 20.0, (850,480))
    start_time = time.time()
    while(int(time.time() - start_time) < duration ):
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        
        # cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        frame = cv2.resize(frame, (850,480))   
        output.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(1)
        # duration -= 1
        # if duration == 0:
        #     break
        
            
    cap.release()
    cv2.destroyAllWindows()

# countdown(5)
ML(10)
# recordWebCam()
# convertToFrames('webcam')

# for i in tick(1.0):
#     print(i)

# def timeStamp():
#     cap = cv2.VideoCapture('video.mp4')
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
#     calc_timestamps = [0.0]

#     while(cap.isOpened()):
#         frame_exists, curr_frame = cap.read()
#         if frame_exists:
#             timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
#             # calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
#         else:
#             break

#     cap.release()
    # for i in timestamps:
    # print(convert(0))
    # for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    #     print('Frame %d difference:'%i, ts, cts)

# def tick(time_interval):
#     next_tick = time.time() + time_interval
#     while True:
#         time.sleep(0.2)     # Minimum delay to allow for catch up
#         while time.time() < next_tick:
#             time.sleep(0.2)

#         yield time.time()
#         next_tick += time_interval

# def getVideoDuration():
    # video = moviepy.editor.VideoFileClip(current_directory+"/video.mp4")
    # video_duration = int(video.duration)
    # print(video_duration)
    # hours, mins, secs = convert(video_duration)
    # print("Hours:", hours)
    # print("Minutes:", mins)
    # print("Seconds:", secs)