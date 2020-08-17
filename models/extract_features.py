#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:55:04 2020

@author: tim

Take a directory of videos, extract features  using CNN, save features per video to a file in an output directory 

"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

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
    outfile = os.path.splitext(vid)[0]
    if C["crop_type"] == "T":
        vid_np = cs760.crop_image(vid_np, C["crop_top"])
        outfile += "__TOP.pkl"
    elif  C["crop_type"] == "B":
        vid_np = cs760.crop_image(vid_np, C["crop_bottom"])
        outfile += "__BOT.pkl"
    else:
        outfile += "__NOCROP.pkl"        
        
    batch = cs760.resize_batch(vid_np, width=C["expect_img_size"], height=C["expect_img_size"], pad_type='L',
                           inter=cv2.INTER_CUBIC, BGRtoRGB=False, 
                           simplenormalize=True,
                           imagenetmeansubtract=False)
    features = extract(C, model, batch)
    cs760.saveas_pickle(features, os.path.join(C["outdir"], outfile))

print('Finished outputting features!!')

#tst = cs760.loadas_pickle(os.path.join(C["outdir"], os.path.splitext(vid)[0] + '.pkl'))






