#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:08:58 2020

@author: tim

Create train / val/ test splits

NOTE: This script will assume the input feature files are in the directory specified in config760.json "outdir" key i.e. C['outdir']
      and it will create /train, /val, and /test subdirectories underneath the 'outdir' subdirectory
      
      It will NOT remove the existing contents of directories before use so before running manually delete /train, /val, and /test

"""

import sys
import os
import random
import copy
import shutil

import cs760    #opencv based utils for vid / image manipulation plus other utilities

C = cs760.loadas_json('config760.json')
print("Running with parameters:", C)

traindir = os.path.join(C["outdir"], "train")
valdir = os.path.join(C["outdir"], "val")
testdir = os.path.join(C["outdir"], "test")

print(f'Creating TRAIN Dir: {traindir}')
os.makedirs(traindir, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory

print(f'Creating VAL Dir: {valdir}')
os.makedirs(valdir, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory

print(f'Creating TEST Dir: {testdir}')
os.makedirs(testdir, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory

feat_files = cs760.list_files_pattern(C["outdir"], '*.pkl')
#print(f"Input Feature files: {feat_files}")

feat_files.sort()
random.seed(42)                     #this should make it reproducable


# build dict of [files] for each sign
curr_sign_dict = {}                 # dict['sign'] = [filename1, filename2,....] 

for feat_file in feat_files:
    end_idx = feat_file.find('__')  #'ADVISE.INFLUENCE__10__BOT.pkl' 16
    sign = feat_file[:end_idx]      # 'ADVISE.INFLUENCE'
    if curr_sign_dict.get(sign) is None:
        curr_sign_dict[sign] = []       # create new sign entry in dict
    curr_sign_dict[sign].append(feat_file)
        

# Copy all
for k in curr_sign_dict:
    file_list = curr_sign_dict[k]
    random.shuffle(file_list)
    found = False
    for i, feat_file in enumerate(file_list):
        var_idx = feat_file.rfind('__')  # 'ADVISE.INFLUENCE__10__BOT.pkl' 20
        var_type = feat_file[var_idx:]   # '__BOT.pkl'        
        #var_type = var_type.replace('.pkl', '')  # '__BOT'
        if var_type == '__TOP.pkl':
            found = True
            break
    curr_file_list = copy.deepcopy(file_list)  # to avoid errors when modifying a list that we are in the process of looping through        
    if not found:
        print(f'Strange. No TOP entry found for sign {file_list[0]}. NOT added to VAL OR TEST')
    else:         # 
        nontrain = file_list[i]              #'ADVISE.INFLUENCE__10__BOT.pkl'
        var_idx = nontrain.rfind('__')  # 20
        firstpart = nontrain[:var_idx]       # 'ADVISE.INFLUENCE__10'
        if random.choice(['test', 'val']) == 'val':
            outdir = valdir
        else:
            outdir = testdir
        for i, file in enumerate(file_list):
            var_idx = file.rfind('__')
            if file[:var_idx] == firstpart:             #copy all the varieties of that file to test/val to avoid contamination even though we'll only use the 'top' entry for actual val/test
                print(f'Copying {file} to {outdir}')
                shutil.copy(os.path.join(C["outdir"], file), outdir)
                curr_file_list[i] = '_DONE_'
    for file in curr_file_list:                         # copy all remaining files to traindir
        if file != '_DONE_':
            print(f'Copying {file} to {traindir}')
            shutil.copy(os.path.join(C["outdir"], file), traindir)
print('Finished! (yay)')





