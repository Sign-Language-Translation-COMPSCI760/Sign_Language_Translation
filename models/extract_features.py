#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:55:04 2020

@author: tim

Take a directory of videos, extract features using CNN, save features per video to a .pkl file in an output directory 

Usage: (Must run from models subdirectory and first edit the config_dirs.json file to point to your input and output directories)

python extract_features.py config_dirs.json config760.json   

"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

import numpy as np

import imgaug.augmenters as iaa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cs760    #opencv based utils for vid / image manipulation plus other utilities

def check_vid_sizes(C):
    """ check video dimensions and number of frames
    """
    crops_for_resolutions = {}
    example_vids = {}
    for subdir in ['NZ', 'US']:
        video_directory = "/home/tim/OneDrive/Documents/uni/760 Data Mining and Machine Learning/GroupProj/all signs/" + subdir  #C['dirs']['indir']
        vids = cs760.list_files_pattern(video_directory, C["vid_type"])
        frames_max = 0
        frames_min = 99999999
        frames_mean = 0
        vids_count = 0
        
        for i, vid in enumerate(vids):
            vid_np = cs760.get_vid_frames(vid, 
                            video_directory, 
                            writejpgs=False,
                            writenpy=False,
                            returnnp=True)
            (framecount, frameheight, framewidth, channels) = vid_np.shape    
            vids_count += 1
            if framecount > frames_max:
                frames_max = framecount
            if framecount < frames_min:
                frames_min = framecount
            frames_mean += framecount    
            if crops_for_resolutions.get( str(frameheight) + "-" + str(framewidth) ) is None:
                crops_for_resolutions[str(frameheight) + "-" + str(framewidth)] = [0,0,0,0]
                #print("NEW RESOLUTION:", str(frameheight) + "-" + str(framewidth))
                example_vids[str(frameheight) + "-" + str(framewidth)] = vid_np[4]#os.path.join(video_directory, vid)
                #plt.imshow(vid_np[4])
                #inp = input("Hit any key to continue..")
        frames_mean /= vids_count
        print('Stats for Video directory: ', video_directory)
        print(f"Vid Count:{vids_count}  Frames Max:{frames_max}  Min:{frames_min}  Mean:{frames_mean}")
        print("Resolutions found:", crops_for_resolutions)

        plt.imshow(example_vids['368-480'])  
        tst = cs760.crop_image(example_vids['368-480'], [60, 5, 420, 365])
        plt.imshow(tst)
        tst = cs760.image_resize(tst, height=600, width=600)
        plt.imshow(tst[0])

        plt.imshow(example_vids['360-640'])
        tst = cs760.crop_image(example_vids['360-640'], [145, 10, 495, 360])
        plt.imshow(tst)
        tst = cs760.image_resize(tst, height=600, width=600)
        plt.imshow(tst[0])

        plt.imshow(example_vids['240-320'])  
        tst = cs760.crop_image(example_vids['240-320'], [0, 0, 320, 240])
        plt.imshow(tst)
        tst = cs760.image_resize(tst, height=600, width=600)
        plt.imshow(tst[0])

        plt.imshow(example_vids['480-640'])  
        tst = cs760.crop_image(example_vids['480-640'], [0, 0, 640, 480])
        plt.imshow(tst)
        tst = cs760.image_resize(tst, height=600, width=600)
        plt.imshow(tst[0])
        return
    


def extract(C, model, batch):
    """ Extract features for one vid
    batch = vid encoded and resized as np array
    """
    fullfeatures_tf = tf.zeros((0, C["cnn_feat_dim"]), dtype=tf.float32)
    frame_count_lessthan_batchsize = True
    for i in range(C["cnn_batch_size"], batch.shape[0], C["cnn_batch_size"]):
        frame_count_lessthan_batchsize = False
        #print(i-C["cnn_batch_size"], i)
        batch_tf = tf.constant(batch[i-C["cnn_batch_size"]:i], dtype=tf.float32)  # convert to tf
        features = model(batch_tf)
        #print(features[0], features[-1])
        fullfeatures_tf = tf.concat([fullfeatures_tf, features], axis=0)
    if frame_count_lessthan_batchsize:
        i = 0
    if batch.shape[0] - i > 0:
        #print(i, i + (batch.shape[0] - i))
        batch_tf = tf.constant(batch[i:i+(batch.shape[0]-i)], dtype=tf.float32)  # convert to tf
        features = model(batch_tf)
        fullfeatures_tf = tf.concat([fullfeatures_tf, features], axis=0)
    return fullfeatures_tf.numpy()


def main():

    try:
        config_dirs_file = sys.argv[1] # directories file
        config_file = sys.argv[2]      # main params file
    except:
        print("Config file names not specified, setting them to default namess")
        config_dirs_file = "config_dirs.json"
        config_file = "config760.json"
    print(f'USING CONFIG FILES: config dirs:{config_dirs_file}  main config:{config_file}')    
    
    #print(type(feature_directory))
    C = cs760.loadas_json('config760.json')
    print("Running with parameters:", C)
    
    Cdirs = cs760.loadas_json(config_dirs_file)
    print("Directories:", Cdirs)
    
    C['dirs'] = Cdirs
    video_directory = C['dirs']['indir']
    feature_directory = C['dirs']['outdir']
    
    print(f'Creating feature file Dir: {feature_directory}')
    os.makedirs(feature_directory, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory


    sometimes = lambda aug: iaa.Sometimes(C["augmentation_chance"][0], aug)
    sequential_list = [iaa.Sequential([sometimes(iaa.Fliplr(1.0))]), # horizontal flip
    iaa.Sequential([sometimes(iaa.Rotate(-5, 5))]), # rotate 5 degrees +/-
    iaa.Sequential([sometimes(iaa.CenterCropToAspectRatio(1.15))]),
    iaa.Sequential([sometimes(iaa.MultiplyBrightness((2.0, 2.0)))]), # increase brightness
    iaa.Sequential([sometimes(iaa.MultiplyHue((0.5, 1.5)))]), # change hue random
    iaa.Sequential([sometimes(iaa.RemoveSaturation(1.0))]), # effectively greyscale
    iaa.Sequential([sometimes(iaa.pillike.FilterContour())]), # edge detection
    iaa.Sequential([sometimes(iaa.AdditiveLaplaceNoise(scale=0.05*255, per_channel=True))]), # add colourful noise
    iaa.Sequential([sometimes(iaa.Invert(1))]) # invert colours
    ]


    print("Reading videos from " + video_directory)
    print("Outputting features to " + feature_directory)

    print("Loading pretrained CNN...")
    model = hub.KerasLayer(C["module_url"])  # can be used like any other kera layer including in other layers...
    print("Pretrained CNN Loaded OK")

    vids = cs760.list_files_pattern(video_directory, C["vid_type"])
    print(f'Processing {len(vids)} videos...')

    for i, vid in enumerate(vids):
        print(f'{i} Processing: {vid}')    
        vid_np = cs760.get_vid_frames(vid, 
                        video_directory, 
                        writejpgs=False,
                        writenpy=False,
                        returnnp=True)
        (framecount, frameheight, framewidth, channels) = vid_np.shape
        res_key = str(frameheight) + "-" + str(framewidth)
        #print(vid, vid_np.shape)
        outfile = os.path.splitext(vid)[0]
        
        print(f"Vid frames, h, w, c = {(framecount, frameheight, framewidth, channels)}")
        
        if C["crop_by_res"].get(res_key) is not None:
            vid_np_top = cs760.crop_image(vid_np, C["crop_by_res"][res_key])
            print(f"Cropped by resolution to {C['crop_by_res'][res_key]}")
        else:    
            vid_np_top = cs760.crop_image(vid_np, C["crop_top"])
            print(f"Cropped by default to {C['crop_top']}")

        outfile_top = outfile + "__TOP.pkl"

        for n in range((len(sequential_list) + 1)):
            if n != 0:
                vid_aug = sequential_list[n - 1](images=vid_np_top) # augments frames
                if type(vid_aug) is list:
                    vid_aug = np.asarray(vid_aug)
                batch = cs760.resize_batch(vid_aug, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                            inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                            simplenormalize=True,
                            imagenetmeansubtract=False)
                temp_outfile = outfile_top[:-4] + C["augmentation_type"][n - 1] + ".pkl"
                features = extract(C, model, batch)
                cs760.saveas_pickle(features, os.path.join(feature_directory, temp_outfile))
            else:
                batch = cs760.resize_batch(vid_np_top, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                                inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                                simplenormalize=True,
                                imagenetmeansubtract=False)
                features = extract(C, model, batch)
                cs760.saveas_pickle(features, os.path.join(feature_directory, outfile_top))
                print(f'Features output shape: {features.shape}')
                
        if C["crop_type"] == 'B':  # only for boston vids
            vid_np_bot = cs760.crop_image(vid_np, C["crop_bottom"])
            outfile_bot = outfile + "__BOT.pkl"  
            batch = cs760.resize_batch(vid_np_bot, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                        inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                        simplenormalize=True,
                        imagenetmeansubtract=False)
            features = extract(C, model, batch)
            cs760.saveas_pickle(features, os.path.join(feature_directory, outfile_bot))

    print('Finished outputting features!!')

main()