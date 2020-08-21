#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:55:04 2020

@author: tim

Take a directory of videos, extract features using CNN, save features per video to a .pkl file in an output directory 

Usage: (Must run from models subdirectory and first edit the config760.json file to point to your input and output directories)

python extract_features.py      

"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

from vidaug import augmentors as va

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

    C = cs760.loadas_json('config760.json')
    print("Running with parameters:", C)

    print(f'Reading videos from {C["indir"]}')
    print(f'Outputting features to {C["outdir"]}')

    print("Loading pretrained CNN...")
    model = hub.KerasLayer(C["module_url"])  # can be used like any other kera layer including in other layers...
    print("Pretrained CNN Loaded OK")



    vids = cs760.list_files_pattern(C["indir"], '*.mov')
    print(f'Processing {len(vids)} videos...')
    for i, vid in enumerate(vids):
        print(f'{i} Processing: {vid}')    
        vid_np = cs760.get_vid_frames(vid, 
                        C["indir"], 
                        writejpgs=False,
                        writenpy=False,
                        returnnp=True)
        #print(vid, vid_np.shape)
        outfile = os.path.splitext(vid)[0]
        if C["crop_type"] == "T":
            vid_np = cs760.crop_image(vid_np, C["crop_top"])
            outfile += "__TOP.pkl"
        elif  C["crop_type"] == "B":
            vid_np = cs760.crop_image(vid_np, C["crop_bottom"])
            outfile += "__BOT.pkl"
        else:
            outfile += "__NOCROP.pkl"        
        #print('Cropped shape: ', vid_np.shape)
            
        batch = cs760.resize_batch(vid_np, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                            inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                            simplenormalize=True,
                            imagenetmeansubtract=False)
        #print('Resized shape: ', batch.shape)
        
        sometimes = lambda aug: va.Sometimes(0.3, aug) # set augmentation 30% of the time
        seq = va.Sequential([ # define augmentation steps
            sometimes(va.HorizontalFlip()), # horizontally flips video
            sometimes(va.InvertColor()), # inverts video colours
            sometimes(va.VerticalFlip()) # vertically flips video
        ])
        video_aug = seq(batch) # augments frams
        new_batch = np.array(video_aug) # Converts the augmented video into supported batch format

        features = extract(C, model, new_batch)
        cs760.saveas_pickle(features, os.path.join(C["outdir"], outfile))

    print('Finished outputting features!!')

main()






