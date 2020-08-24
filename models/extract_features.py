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

def extract(C, model, batch):
    """ Extract features for one vid
    batch = vid encoded and resized as np array
    """
    fullfeatures_tf = tf.zeros((0, C["cnn_feat_dim"]), dtype=tf.float32)
    for i in range(C["cnn_batch_size"], batch.shape[0], C["cnn_batch_size"]):
        #print(i-C["cnn_batch_size"], i)
        batch_tf = tf.constant(batch[i-C["cnn_batch_size"]:i], dtype=tf.float32)  # convert to tf
        features = model(batch_tf)
        #print(features[0], features[-1])
        fullfeatures_tf = tf.concat([fullfeatures_tf, features], axis=0)
    if batch.shape[0] - i > 0:
        #print(i, i + (batch.shape[0] - i))
        batch_tf = tf.constant(batch[i:i+(batch.shape[0]-i)], dtype=tf.float32)  # convert to tf
        features = model(batch_tf)
        fullfeatures_tf = tf.concat([fullfeatures_tf, features], axis=0)
    return fullfeatures_tf.numpy()

def main():

#    try:
#        video_directory = sys.argv[1] # intput video files
#        feature_directory = sys.argv[2] # output feature files
#    except:
#        print("Video and feature directories not specified, setting them to default locations")
#        print("Video locations: ../dataset/videos")
#        print("Feature locations: ../features")
#        video_directory = "../dataset/videos"
#        feature_directory = "../features"

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

    vids = cs760.list_files_pattern(video_directory, '*.mov')
    print(f'Processing {len(vids)} videos...')

    for i, vid in enumerate(vids):
        print(f'{i} Processing: {vid}')    
        vid_np = cs760.get_vid_frames(vid, 
                        video_directory, 
                        writejpgs=False,
                        writenpy=False,
                        returnnp=True)

        #print(vid, vid_np.shape)
        outfile = os.path.splitext(vid)[0]

        vid_np_top = cs760.crop_image(vid_np, C["crop_top"])
        outfile_top = outfile + "__TOP.pkl"
        vid_np_bot = cs760.crop_image(vid_np, C["crop_bottom"])
        outfile_bot = outfile + "__BOT.pkl"  

        for n in range((len(sequential_list) + 1)):
            if n != 0:
                vid_aug = sequential_list[n - 1](images=vid_np_top) # augments frames
                if type(vid_aug) is list:
                    vid_aug = np.asarray(vid_aug)
                batch = cs760.resize_batch(vid_aug, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                            inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                            simplenormalize=False,
                            imagenetmeansubtract=False)
                temp_outfile = outfile_top[:-4] + C["augmentation_type"][n - 1] + ".pkl"
                features = extract(C, model, batch)
                cs760.saveas_pickle(features, os.path.join(feature_directory, temp_outfile))
            else:
                batch = cs760.resize_batch(vid_np_top, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                                inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                                simplenormalize=False,
                                imagenetmeansubtract=False)
                features = extract(C, model, batch)
                cs760.saveas_pickle(features, os.path.join(feature_directory, outfile_top))
        
        batch = cs760.resize_batch(vid_np_bot, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                    inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                    simplenormalize=False,
                    imagenetmeansubtract=False)
        features = extract(C, model, batch)
        cs760.saveas_pickle(features, os.path.join(feature_directory, outfile_bot))

    print('Finished outputting features!!')

main()