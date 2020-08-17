#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:50:15 2020

@author: tim

tf hub test

# get started https://www.tensorflow.org/hub. MUST have tensorflow >= 2.2 for tf hub efficientnet model to work!
!pip install tensorflow_hub
!pip install tensorflow

Must run this file from the subdir ..../Sign_Language_Translation/models

"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

import cs760    #opencv based utils for vid / image manipulation


if sys.platform == 'win32':
    #sys.path.append('../tests')
    filedir = 'C:/Users/timha/OneDrive/Documents/uni/760 Data Mining and Machine Learning/GroupProj'
    outdir = 'C:/tmp'
else:
    #sys.path.append('/home/tim/OneDrive/gitrepos/tests')
    filedir = '/media/tim/dl3storage/Datasets/asllrp'    
    outdir = '/media/tim/dl3storage/tmp'

vid = 'Liz_10.mov'
bs = 40              #batch size
expect_img_size = 600  # EfficientNet was trained on images with resolultion 600x600 so resizing to that size
module_url = "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1"   #EfficientNet b7 model expects input 600x600
num_classes = 10

# quick tf test #######################################
a  = np.random.randn(100,200,40,3)

b = tf.constant(a, dtype=tf.float32)
b.device  #prints CPU but might be a bug and actually be on gpu?
#c = b.gpu()
#c.device
e = b*2
e.device # now should print a gpu device like '/job:localhost/replica:0/task:0/device:GPU:0'

del a
del b
del e  
# end tf test #########################################

# test vid ################################################

vid_np = cs760.get_vid_frames(vid, 
                  filedir, 
                  writejpgs=False,
                  writenpy=False,
                  returnnp=True)

print(vid_np.shape, vid_np.dtype)
plt.imshow(vid_np[4])
plt.imshow(cs760.crop_image(vid_np[4], (0, 15, 300, 300+15)))
plt.imshow(cs760.crop_image(vid_np[4], (0, 356, 300, 356+300)))

batch_cropped = cs760.crop_image(vid_np, [0, 15, 300, 300+15])
print(batch_cropped.shape)
plt.imshow(batch_cropped[4])

print(vid_np[4, 75, 135])

batch = cs760.resize_batch(vid_np, width=expect_img_size, height=expect_img_size, pad_type='L',
                           inter=cv2.INTER_AREA, BGRtoRGB=False, 
                           simplenormalize=True,
                           imagenetmeansubtract=False)
print(batch.shape, batch.dtype)
plt.imshow(batch[4])
print(batch[4, 75, 135])


# tf hub test ########################

model = hub.KerasLayer(module_url)  # can be used like any other kera layer including in other layers...

batch_tf = tf.constant(batch[:bs], dtype=tf.float32)  # convert to tf (got OOM when tried to run all 128 frames through at once. 40 works ok)

# NOTE to train end-to-end I think setting model.trainable = True will work
features = model(batch_tf)   # Returns features with shape [batch_size, num_features].
print(features.shape)  #(batch_size, 2560)


# run a whole vid though model
fullfeatures_tf = tf.zeros((0, 2560), dtype=tf.float32)
for i in range(bs, batch.shape[0], bs):
    print(i-bs, i)
    batch_tf = tf.constant(batch[i-bs:i], dtype=tf.float32)  # convert to tf
    features = model(batch_tf)
    print(features[0], features[-1])
    fullfeatures_tf = tf.concat([fullfeatures_tf, features], axis=0)
if batch.shape[0] - i > 0:
    print(i, i + (batch.shape[0] - i))
    batch_tf = tf.constant(batch[i:i+(batch.shape[0]-i)], dtype=tf.float32)  # convert to tf
    features = model(batch_tf)
    fullfeatures_tf = tf.concat([fullfeatures_tf, features], axis=0)

print(fullfeatures_tf.shape)

# test as part of a model
m = tf.keras.Sequential([
    hub.KerasLayer(module_url,
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(num_classes, activation='softmax')     #pretend last layer
])
m.build([None, expect_img_size, expect_img_size, 3])  # Batch input shape is param. Builds the model based on input shapes received so can do m.summary() etc.
m.summary()

"""
# generator/ sequence example
class CIFAR10Sequence(tf.keras.utils.Sequence):
    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
               for filename in batch_x]), np.array(batch_y)

sequence = CIFAR10Sequence(filenames, labels, batch_size)
model.fit(sequence, epochs=10)
"""

# callbacks
# metrics
#m.compile
#m.fit
#m.evaluate




