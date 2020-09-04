#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:29:18 2020

@author: tim

Create "final" dataset val and test subdirectories

NOTE: AFTER Running the ASL vids to add to train will be in train_asl_dict: Manually copy them into train after checking

NOTE: Additional to this script, manually copy the augmented BOSTON dataset pkl files into a "train" subdirectory

usage: python create_final_dataset_val_test.py config_dirs.json config760.json


"""


import numpy as np
import sys
import shutil
import os

import numpy as np

import cs760    #opencv based utils for vid / image manipulation plus other utilities

def check_frame_counts(C):
    """ Gather stats on input and output frame counts
    """
    dirs = C['dirs']['test_set_dirs'] + C['dirs']['val_set_dirs']
    #dirs = [os.path.join(C['dirs']['outdir'], 'train')]    #{'max': 193, 'min': 106, 'mean': 136.8842105263158, 'count': 9405}
    stats_dir = {}
    for inp_dir in dirs:
        stats_dir[inp_dir] = {'max': 0, 'min': 9999999999, 'mean' : 0.0, 'count': 0} 
        curr_input_dir = os.path.join(C['dirs']['dict_pkls'], inp_dir)
        feature_files = cs760.list_files_pattern(curr_input_dir, '*.pkl')
        for feat_file in feature_files:
            sample = cs760.loadas_pickle(os.path.join(curr_input_dir, feat_file))
            (frame_count,feat_count) = sample.shape
            assert feat_count == 2560, f"ERROR: Invalid Feature Count: {feat_count}. Must be 2560."
            stats_dir[inp_dir]['count'] += 1
            stats_dir[inp_dir]['mean'] += frame_count
            if frame_count > stats_dir[inp_dir]['max']:  
                stats_dir[inp_dir]['max'] = frame_count
            if frame_count < stats_dir[inp_dir]['min']:  
                stats_dir[inp_dir]['min'] = frame_count
        stats_dir[inp_dir]['mean'] /= stats_dir[inp_dir]['count']
        print("****************************************************")
        print(f'Video Frame Stats for subdir: {inp_dir}')
        print(stats_dir[inp_dir])
    return stats_dir            


def check_files(C, inputdirs_key='test_set_dirs'):
    """ Check output file names and stats
    """
    sign_rejects = set()
    out_files = []
    in_files = []
    sign_counts = {}
    for inp_dir in C['dirs'][inputdirs_key]:    
        curr_input_dir = os.path.join(C['dirs']['dict_pkls'], inp_dir)
        feat_files = cs760.list_files_pattern(curr_input_dir, '*.pkl')
        feat_files.sort()
        for feat_file in feat_files:
            in_files.append(feat_file)
            end_idx = feat_file.find('_')  #'ADVISE.INFLUENCE_10__BOT.pkl' 16
            sign = feat_file[:end_idx]      # 'ADVISE.INFLUENCE'
            rest_of_file = feat_file[end_idx:]
            #end_idx = rest_of_file.find('__')
            #if end_idx == -1:  #__TOP wasn't added
            #    end_idx = end_idx = rest_of_file.find('.pkl')
            #    rest_of_file = rest_of_file[:end_idx] + '__TOP.pkl'
            rest_of_file = '_' + rest_of_file  #add extra underscore after sign
            if sign == 'STAND.UP':
                sign = 'STAND-UP'
            if sign == 'CANCEL.CRITISIZE':
                sign = 'CANCEL.CRITICIZE'
            if sign == 'ADVISE':
                sign = 'ADVISE.INFLUENCE'
            if sign == 'GOLD':
                sign = 'GOLD.ns-CALIFORNIA'
            if sign not in C["sign_classes"]:
                sign_rejects.add(sign)
            new_file = sign + rest_of_file    
            out_files.append(new_file)
            if sign_counts.get(sign) is None:
                sign_counts[sign] = 1
            else:
                sign_counts[sign] += 1
    print('STATS FOR ', inputdirs_key)        
    if len(sign_rejects) > 0:
        print(f"Number of Reject Signs: {len(sign_rejects)}")   # WAS {'STAND.UP', 'CANCEL.CRITISIZE'}
        print(f"Reject Signs:", sign_rejects)
    else:
        print("All signs OK")
    print('EXAMPLE OUTPUT FILE NAMES:')    
    print(out_files[:12])                        
    print(f'Number of signs: {len(sign_counts)}')
    print('Sign Counts:', [ (s, sign_counts[s]//10) for s in sign_counts])      #there are 1+9 augmentations = 10 files per sign vid
    return sign_counts, out_files, in_files, sign_rejects


def cp_files(C, in_files, out_files, in_dir, out_dir ):
    """ Copy files changing the filename in in_list to the corresponding one in out_list
    """
    for i in range(len(in_files)):
        in_file = os.path.join(in_dir, in_files[i])
        out_file = os.path.join(out_dir, out_files[i])
        print(f"Copying {in_file} to {out_file}")
        shutil.copy(in_file, out_file)
    return        


def split_asl(C, in_files, out_files):
    """ Split ASL dict vids with 2 signs into 1->val, 1->train
    """
    sign_dict = {}
    
    for i in range(len(out_files)):
        feat_file_in = in_files[i]
        feat_file_out = out_files[i]
        end_idx = feat_file_out.find('__')  #'ADVISE.INFLUENCE_10__BOT.pkl' 16
        sign = feat_file_out[:end_idx]      # 'ADVISE.INFLUENCE'
        rest_of_file = feat_file_out[end_idx+2:]
        end_idx = rest_of_file.find('__')
        vid_name = rest_of_file[:end_idx]
        
        if sign_dict.get(sign) is None:
            sign_dict[sign] = {}
            
        if sign_dict[sign].get(vid_name) is None:
            sign_dict[sign][vid_name] = {'in_files':[], 'out_files':[], 'val1_or_train0' :1}
            
        sign_dict[sign][vid_name]['in_files'].append(feat_file_in)
        sign_dict[sign][vid_name]['out_files'].append(feat_file_out)
        
    flipfirst = True
    for sign in sign_dict:
        print(f"SIGN: {sign} keys: {sign_dict[sign].keys()}")
        if len(sign_dict[sign].keys()) == 2:
            for i, vid_name in enumerate(sign_dict[sign]):
                if i == 0 and flipfirst:
                    sign_dict[sign][vid_name]['val1_or_train0'] = 0
                if i == 1 and (not flipfirst):    
                    sign_dict[sign][vid_name]['val1_or_train0'] = 0
                print(f"VID:{vid_name} Len:{len(sign_dict[sign].keys())} val1_or_train0:{sign_dict[sign][vid_name]['val1_or_train0']}")    
        flipfirst = not flipfirst    

    in_files_train = []
    in_files_val = []
    out_files_train = []
    out_files_val = []

    for sign in sign_dict:
        for i, vid_name in enumerate(sign_dict[sign]):
            if sign_dict[sign][vid_name]['val1_or_train0'] == 0:
                in_files_train += sign_dict[sign][vid_name]['in_files']
                out_files_train += sign_dict[sign][vid_name]['out_files']
            else:
                in_files_val += sign_dict[sign][vid_name]['in_files']
                out_files_val += sign_dict[sign][vid_name]['out_files']
    
    #traindir = os.path.join(C['dirs']["dict_pkls"], "train")
    valdir = os.path.join(C['dirs']["dict_pkls"], "val")
    tmpdir = os.path.join(C['dirs']["dict_pkls"], "train_asl_dict")
    os.makedirs(tmpdir, exist_ok=True)
    asl_dir = os.path.join(C['dirs']["dict_pkls"], C['dirs']['val_set_dirs'][0])
    
    print(f'Copying asl vids from {asl_dir} to {valdir}')
    cp_files(C, in_files=in_files_val, out_files=out_files_val, in_dir=asl_dir, out_dir=valdir)
    
    print(f'Copying asl vids from {asl_dir} to {tmpdir}')  
    cp_files(C, in_files=in_files_train, out_files=out_files_train, in_dir=asl_dir, out_dir=tmpdir)
    
    print('finished copying ASL vids!')
    
    return



if __name__ == '__main__':

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
    pkl_directory = C['dirs']['dict_pkls']
    #feature_directory = C['dirs']['dict_pkls']

    traindir = os.path.join(C['dirs']["dict_pkls"], "train")
    valdir = os.path.join(C['dirs']["dict_pkls"], "val")
    testdir = os.path.join(C['dirs']["dict_pkls"], "test")
    

    
    #print(f'Creating feature file Dir: {feature_directory}')
    #os.makedirs(feature_directory, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory

    print(f'Creating TRAIN Dir: {traindir}')
    os.makedirs(traindir, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory
    
    print(f'Creating VAL Dir: {valdir}')
    os.makedirs(valdir, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory
    
    print(f'Creating TEST Dir: {testdir}')
    os.makedirs(testdir, exist_ok=True)        #if dir already exists will continue and WILL NOT delete existing files in that directory
    
    # check .pkl np shapes
    stats_dir = check_frame_counts(C)    
    
    
    nz_sign_counts, nz_out_files, nz_in_files, nz_sign_rejects = check_files(C, inputdirs_key='test_set_dirs')
    asl_sign_counts, asl_out_files, asl_in_files, asl_sign_rejects = check_files(C, inputdirs_key='val_set_dirs')
    
    nz_signs = set(nz_sign_counts.keys())
    asl_signs = set(asl_sign_counts.keys())
    asl_nz_common = asl_signs.intersection(nz_signs)

    print(f"Number of NZ: {len(nz_signs)} ASL: {len(asl_signs)}  Intersecting signs: {len(asl_nz_common)}")  #Number of NZ: 23 ASL: 64  Intersecting ASL/NZ signs: 22

    asl_signs_2_vids = set([ s for s in asl_sign_counts if asl_sign_counts[s]//10 > 1])
    print(f'There are {len(asl_signs_2_vids)} ASL Signs with 2 vids. They are: ', asl_signs_2_vids)
    
    asl_2_vids_nzl = asl_signs_2_vids.intersection(nz_signs)
    print(f'There are {len(asl_2_vids_nzl)} ASL Signs with 2 vids that are also in the NZ Signs. They are:', asl_2_vids_nzl)
    
    nz_dir = os.path.join(C['dirs']["dict_pkls"], C['dirs']['test_set_dirs'][0])
    print(f"Copying NZSL feat files from {nz_dir} to {testdir}.")
    cp_files(C, in_files=nz_in_files, out_files=nz_out_files, in_dir=nz_dir, out_dir=testdir)
    
    split_asl(C, in_files=asl_in_files, out_files=asl_out_files)

    print('Finished creating val and test dirs!!')

