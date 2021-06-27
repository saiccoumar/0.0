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
def getVid(url,count):
    print("Started download")
    yt = YouTube(url)
    while True: 
        try:
            #yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first().download(filename = 'video' + str(count))
            yt.streams.filter(res="1080p").first().download(output_path = "../static", filename = 'video')
            if (current_directory+'/video.mp4'):
                break
        except:
            print('video could not be found')
            break

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

def convertSecs(seconds):
    # seconds = milliseconds / 1000 
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