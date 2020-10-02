import cv2
import os
def combine():

    framesList1 = []
    fps = 15

    for file in os.listdir(os.getcwd()):
        if file.endswith(".png"):  # image file type (need to change)
            framesList1.append(cv2.imread(file))
            name = file

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    sz = (480, 368) #video dimensions



    out1 = cv2.VideoWriter() # if a different file type is wanting to be used change code here
    out1.open(name[:-4]+".mp4" ,fourcc,fps,sz,True)

    for frame in framesList1: #writing video frames in list to video
        out1.write(frame)

    out1.release()


combine()



