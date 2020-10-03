#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:18:54 2020

@author: tim
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import random

import cs760


gt = [19, 10, 12, 2, 43, 5, 55, 13, 57, 6, 1, 9, 66, 67, 33, 51, 39, 14, 35, 21, 53, 60, 24]


try:
    config_dirs_file = sys.argv[1] # directories file
    config_file = sys.argv[2]      # main params file
except:
    print("Config file names not specified, setting them to default namess")
    config_dirs_file = "config_dirs.json"
    config_file = "config760.json"
print(f'USING CONFIG FILES: config dirs:{config_dirs_file}  main config:{config_file}')

C = cs760.loadas_json(config_file)
assert C["s2_classifier_type"] in ['softmax', 'sigmoid'], f"ERROR Invalid s2_classifier_type {C['s2_classifier_type']}. Must be one of 'softmax' or 'sigmoid'."
if C["s2_classifier_type"] == 'sigmoid':
    pred_sign = sys.argv[3]     # sign to predict if binary classifier
    assert pred_sign in C["sign_classes"], "ERROR: Invalid sign to predict: {pred_sign}."
    C["curr_sign"] = pred_sign
    print(f"TRAINING BINARY CLASSIFIER for sign {C['curr_sign']}")
    
print("Running with parameters:", C)

Cdirs = cs760.loadas_json(config_dirs_file)
print("Directories:", Cdirs)

C['dirs'] = Cdirs

classes_dict = {}
for i, c in enumerate(C["sign_classes"]):
    classes_dict[c] = i                    # index into final output vector for each class
print(f"Class Indices: {classes_dict}")

C['sign_indices'] = classes_dict   
C['num_classes'] = len(C["sign_classes"])


ensemblefiles = ['best55.json',
                 'best_0.1-0.01-take2ndframe.json',
                 'best_0.3_111_55.json',
                 'best_0.3_111_55_1enc.json',
                 'best_0.3_111_55_1enctruly.json',
                 'best_0.3_111_55_1enctruly42.json',
                 'best_422.json',
                 'best_422_40.json',
                 'best_0.34_111_55_1enc2_42.json']

fulllist = []
for file in ensemblefiles:
    inlist = cs760.loadas_json(file)
    fulllist.append(inlist)
    
results = np.array(fulllist)  #(8, 23)

preds = []
for i in range(results.shape[1]):
    sampleresults = results[:,i]
    counts = np.bincount(sampleresults)
    pred = np.argmax(counts)
    preds.append(pred)
    
for i in range(len(gt)):
    if gt[i] == preds[i]:
        print(f'{i}: Correct: {preds[i]} {C["sign_classes"][preds[i]]}')
    












