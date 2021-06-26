import os

current_directory = os.getcwd()

for i in os.listdir(current_directory+"/folder"):
    if "frame" in i:
        os.remove(current_directory+"/folder/"+i)