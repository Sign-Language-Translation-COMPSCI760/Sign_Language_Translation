# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:10:57 2020

@author: timhartill
Some OpenCV routines PLUS other useful utilities

subtract_imagenet_mean
image_resize
get_vid_frames     <- reads a video frame by frame and either writes it out as jpgs or returns it in a np array

"""

import cv2     # for capturing videos
import math
import os
import time
import numpy as np
import pickle
import json
import fnmatch




# Imagenet mean function 
def subtract_imagenet_mean(x):
    """ subtract imagenet mean
    """
    #x = x.astype("float32")
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.680
    return x


def resize_batch(batch, height=None, width=None, pad_type='L',
                 inter=cv2.INTER_AREA, 
                 BGRtoRGB=False, 
                 simplenormalize=True,
                 imagenetmeansubtract=False):
    """ Resize a batch of images. Params mostly same as image_resize except that
        batch is a np array [batch_size, h, w, c]
    """
    newbatch = np.zeros((batch.shape[0], height, width, batch.shape[3]), dtype=np.float32)
    for idx, img in enumerate(batch):
        newimg = image_resize(img, height=height, width=width, pad_type=pad_type,
                             inter=inter, 
                             BGRtoRGB=BGRtoRGB, 
                             addbatchdim=False,
                             simplenormalize=simplenormalize,
                             imagenetmeansubtract=imagenetmeansubtract,
                             returnasint=False)
        newbatch[idx] = newimg
    return newbatch


#https://stackoverflow.com/a/44659589/429476
# It is important to resize without losing the aspect ratio for good detection
def image_resize(image, height=None, width=None, pad_type='L',
                 inter=cv2.INTER_AREA, 
                 BGRtoRGB=True, 
                 addbatchdim=True,
                 simplenormalize=False,
                 imagenetmeansubtract=False,
                 returnasint=False):
    """ Resize image preserving aspect ratio and apply various standard transforms. 
    if width & height both specified, will resize to height, width with 0 padding
    images is np array [h,w,c]
    set pad_type to 'C' to center padded image
    Per cv2 documentation: To shrink an image, it will generally look best with CV_INTER_AREA interpolation, 
    whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
    
    """
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are specified, then resize with padding
    if (width is not None) and (height is not None):
        scale = min(width/w, height/h)
        if scale != 1:
            nw = int(w*scale)
            nh = int(h*scale)
            resized = cv2.resize(image, (nw,nh), interpolation = inter)
            if pad_type != 'C':
                #copyMakeBorder params order: top, bottom, left, right
                resized = cv2.copyMakeBorder(resized,0,height-nh,0,width-nw,cv2.BORDER_CONSTANT,value=(0,0,0))
            else:    
                resized = cv2.copyMakeBorder(resized,
                                             (height-nh)//2,(height-nh)//2,
                                             (width-nw)//2,(width-nw)//2,
                                             cv2.BORDER_CONSTANT,value=(0,0,0))
        else:
            resized = image
    elif height is not None:
        # calculate the ratio of the height
        r = height / float(h)
        dim = (int(w * r), height)
        resized = cv2.resize(image, dim, interpolation = inter)
    elif width is not None:
        # calculate the ratio of the width 
        r = width / float(w)
        dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)
    else:
        resized = image

    if BGRtoRGB:  #if image opened with cv2.imshow or video frame with cam.read() it will be BGR by default but most models need RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
    if imagenetmeansubtract:
        resized = subtract_imagenet_mean(resized.astype(np.float32))

    if simplenormalize:
        resized = resized / 255.0
    
    if addbatchdim:  #[h,w,c]->[1,h,w,c]
        resized = np.expand_dims(resized, axis=0)
        
    if returnasint: #typically do this if not normalising (simplenormalize) but resizing or imagenetmeansubtract turned the dtype to float: 
        resized = resized.astype(np.uint8)
    # return the resized image
    return resized


def crop_image(img, to_x1y1x2y2):
    """ Crop a single image or a batch of images
    """
    (x1, y1, x2, y2) = to_x1y1x2y2
    (h, w, c) = img.shape[-3:]
    if y2 > h: y2 = h
    if x2 > w: x2 = w 
    if len(img.shape) == 3:
        return img[y1:y2, x1:x2]
    return img[:, y1:y2, x1:x2, :]


def get_vid_frames(vid, indir, outdir='', writejpgs=True, writenpy=True, returnnp=True, verbose=False):
    """ Write video frames out as jpgs and or a np array in a npy file
    vid: input video file name excluding path
    indir, ourdir: directory to read vid from and directory to write jpgs and/or npy into
    writejpgs: true = write out jpgs 
    writenpy: true = write out .npy
    returnnp = return np array of video
    """
    count = 0
    basefile = os.path.splitext(vid)[0]
    
    cap = cv2.VideoCapture(os.path.join(indir, vid))   # capturing the video from the given path
    framerate = cap.get(cv2.CAP_PROP_FPS) #frame rate
    frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_np = np.zeros((framecount, frameheight, framewidth, 3), dtype = np.uint8)
    if verbose:
        print(f'{vid}: shape: {vid_np.shape}')
    while(cap.isOpened()):
        frameId = cap.get(cv2.CAP_PROP_POS_FRAMES) #current frame number
        ret, frame = cap.read()     #opencv reads channels as BGR
        if (ret != True):
            break
#        if (frameId % math.floor(framerate) == 0):
        if writejpgs:    
            filename = os.path.join(outdir, basefile + "__frame_" + str(count) + ".jpg")
            cv2.imwrite(filename, frame)  #assumes channels are BGR and writes channels as RGB
        if writenpy or returnnp:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #seap channels to RGB
            vid_np[count] = frame
        count += 1
        time.sleep(0.000001)
    cap.release()  #important to do this
    if writenpy:
        filename = os.path.join(outdir, basefile + ".npy")
        np.save(filename, vid_np)
    if returnnp:
        return vid_np
    return


####################################
#    MISC Utilities
####################################
    
def saveas_pickle(obj, file):
    """ Save python obj to file
    """
    with open(file,"wb") as fp:
        pickle.dump(obj, fp)
    return True


def loadas_pickle(file):
    """ Load python obj from pickle file
    """
    with open(file, 'rb') as fp:
        obj = pickle.load(fp)
    return obj     

def loadas_json(file):
    """ Load python obj from json file
    """
    obj = json.load(open(file))
    return obj

def saveas_json(obj, file, mode="w", indent=5, add_nl=False):
    """ Save python object as json to file
    default mode w = overwrite file
            mode a = append to file
    indent = None: all json on one line
                0: pretty print with newlines between keys
                1+: pretty print with that indent level
    add_nl = True: Add a newline before outputting json. ie if mode=a typically indent=None and add_nl=True   
    Example For outputting .jsonl (note first line doesn't add a newline before):
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=False)
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=True)
          
    """
    with open(file, mode) as fp:
        if add_nl:
            fp.write('\n')
        json.dump(obj, fp, indent=indent)
    return True    

def list_files_pattern(dirtolist, pattern='*'):
    """ Returns a list of files in a dictionary matching a pattern
    """
    return [file for file in os.listdir(dirtolist) if fnmatch.fnmatch(file, pattern)]

def list_files_multipatterns(dirtolist, patternlist = ['*']):
    """ Returns a list of files in a directory matching one of a list of patterns
    """
    files = []
    for pattern in patternlist:
        files += list_files_pattern(dirtolist, pattern)
    return files

####################################
#Test routine stuff below
####################################

"""    
vid_np = get_vid_frames('Liz_10.mov', 
                  'C:/Users/timha/OneDrive/Documents/uni/760 Data Mining and Machine Learning/GroupProj', 
                  'C:/tmp',
                  writejpgs=False,
                  writenpy=False,
                  returnnp=True)

print(vid_np.shape, vid_np.dtype)
"""


