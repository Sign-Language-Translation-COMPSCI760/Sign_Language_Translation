#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:09:14 2020

@author: tim

Create dummy val set from boston train vids to test extract_features

"""


import numpy as np
import sys
import os
import shutil
import random

import numpy as np


import cs760    #opencv based utils for vid / image manipulation plus other utilities





if __name__ == '__main__':
    
    try:
        config_dirs_file = sys.argv[1] # directories file
        config_file = sys.argv[2]      # main params file
    except:
        print("Config file names not specified, setting them to default namess")
        config_dirs_file = "config_dirs.json"
        config_file = "config760.json"
    print(f'USING CONFIG FILES: config dirs:{config_dirs_file}  main config:{config_file}')
    
    C = cs760.loadas_json(config_file)
    print("Running with parameters:", C)
    
    Cdirs = cs760.loadas_json(config_dirs_file)
    print("Directories:", Cdirs)
    
    C['dirs'] = Cdirs
    
    feature_directory = C['dirs']['dict_pkls']
    print(f"Base dir: {feature_directory}")


    traindir = os.path.join(feature_directory, "train")
    valdir = os.path.join(feature_directory, "val")
    testdir = os.path.join(feature_directory, "test")

    os.makedirs(valdir, exist_ok=True)

    random.seed(42)
    for sign in C["sign_classes"]:
        print(f"Processing sign {sign} ...")
        files = cs760.list_files_pattern(traindir, sign + "__*")
        valchoice = random.randint(0, len(files)-1)
        file = files[valchoice]
        idx1 = file.find('__') + 2   
        rest = file[idx1:]
        idx2 = rest.find('__') + 2
        idx3 = idx1 + idx2
        valfilebase = file[:idx3]
        valfiles = [f for f in files if (f.find(valfilebase) != -1)]
        for valfile in valfiles:
            print(f"Moving {os.path.join(traindir, valfile)} to {os.path.join(valdir, valfile)}")
            shutil.move(os.path.join(traindir, valfile), os.path.join(valdir, valfile))
    print('Finished!')        





