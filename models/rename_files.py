#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:03:30 2020

@author: tim
"""

import os
import copy

indir = "/media/tim/dl3storage/Datasets/asllrp_features_final/test"


files_orig = os.listdir(indir)

files_new = []

#for file_orig in files_orig:
#    #idx = file_orig.find('_')
#    file_new = file_orig.replace('_','__', 1) 
#    files_new.append(file_new)


signs = set()

for i, file in enumerate(copy.deepcopy(files_orig)):
    sign = file[:file.find('_')]
    if sign == "STAND.UP":
        sign = "STAND-UP"
        idx = file.find('_')
        file = sign + file[idx:]
        print(file)
        
    signs.add(sign) 
    files_new.append(file)

print(signs)           

for i in range(len(files_new)):
    orig = indir + "/" + files_orig[i]
    newname = indir + "/" + files_new[i]
    os.rename(orig, newname)




