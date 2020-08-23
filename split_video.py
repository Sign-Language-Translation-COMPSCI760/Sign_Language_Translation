import cv2
import os
def split_video(input): #only works with mp4 files
    video = cv2.VideoCapture(input)
    framesList1 = []
    framesList2 = []
    count = 0
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    sz = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) #video dimensions

    while True: #reads and save video frames to a list until the read fails(i.e. reaches end of the video)
        success, image = video.read()
        if not(success):
            break
        if (count%2) == 0:
            framesList1.append(image)
        else:
            framesList2.append(image)
        count += 1

    out1 = cv2.VideoWriter() # if a different file type is wanting to be used change code here
    out1.open(input[:-4]+"_1.mp4" ,fourcc,fps,sz,True)
    out2 = cv2.VideoWriter()
    out2.open(input[:-4]+"_2.mp4", fourcc, fps, sz, True)

    for frame in framesList1: #writing video frames in list to video
        out1.write(frame)

    for frame in framesList2:
        out2.write(frame)

    video.release()
    out1.release()
    out2.release()

# reads all video files of mp4 format in the current folder
count = 0
for file in os.listdir(os.getcwd()):
    if file.endswith(".mp4"):
        split_video(file)

