#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:40:07 2020

@author: tim

Stage 2 Model

Usage: from models subdir:
    
    python stage2model.py config_dirs_yourname.json config760.json


"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import random
from sklearn.metrics import accuracy_score

import cs760




class Features_in(tf.keras.utils.Sequence):
    """ Return a batch of samples for model input. Called from .fit(...).
        There will be one instance of this class for each of train, val and test
        subdirs must be one of train, val or test
        Generally you will set shuffle = True for train and false for val and test..
    """
    def __init__(self, C, subdir = "train", shuffle = False):
        self.C = copy.deepcopy(C)
        self.input_dir = os.path.join(C["dirs"]["outdir"], subdir)
        
        if subdir == "train":
            filepattern = C["s2_traintypes"]
            self.batch_size = C["s2_batch_size_train"]
        elif subdir == "val":
            filepattern = C["s2_valtypes"]
            self.batch_size = 99999999  #C["s2_batch_size_val"]  Weirdly .fit() doesnt allow a generator for validation so read entire val set back into np arrays
        elif subdir == "test":
            filepattern = C["s2_testtypes"]
            self.batch_size = C["s2_batch_size_test"]
        else:
            assert True == False, f"FeaturesIn ERROR: Unknown subdir name {subdir}! Unable to proceed"
            
        patterns = ['*' + pattern + '.pkl' for pattern in filepattern]    
        self.filenames = cs760.list_files_multipatterns(self.input_dir, patterns)
        if shuffle:
            random.shuffle(self.filenames)
            
        labels = []            
        for i, filename in enumerate(self.filenames):
            try:
                labels.append( filename[:filename.index('__')] )
            except:
                print(f"ERROR Unable to detect label for file name: {self.filenames[i]}. File name format needs to be: labelname__vidname_or_num__FILETYPE.pkl")
                assert True == False, f"FeaturesIn ERROR: Invalid file name {self.filenames[i]} in {subdir}! Unable to proceed"            

        self.labels = []
        for l in labels:
            if l not in C["sign_classes"]:
                assert True==False, f"FeaturesIn ERROR: Label {l} is not in C['sign_classes']. Either fix the file name or update C['sign_classes']. Aborting."
            self.labels.append(C["sign_indices"][l])    #append indix for label, not the label itself
                
        assert len(self.labels) == len(self.filenames), "FeaturesIn ERROR: number of labels must equal number of input files. Unable to proceed."

        self.maxseqlen = C["s2_max_seq_len"]   # pad or truncate to this seq len
        self.num_classes = C["num_classes"]
        return
    

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_list = []        
        for file in batch_x:
            sample = cs760.loadas_pickle(os.path.join(self.input_dir, file))
            sample = sample[0::C["s2_take_frame"]]        # only take every nth frame
            if sample.shape[0] > self.maxseqlen:          #truncate features to maxseqlen
                sample = sample[0:self.maxseqlen]
            elif sample.shape[0] < self.maxseqlen:        #pad features to maxseqlen
                sample_padded = np.zeros((self.maxseqlen, 2560), dtype=np.float32)
                sample_padded[0:sample.shape[0]] = sample
                sample = sample_padded
            batch_list.append(sample)
        batch_np = np.array(batch_list, dtype = np.float32)                        
        batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size], dtype=np.float32)
        batch_y = tf.keras.utils.to_categorical(batch_y, self.num_classes)
        return batch_np, batch_y


def plots(model):
    """ Adapted from Reuben's code
    """
    plt.plot(model.history["loss"])
    plt.plot(model.history["val_loss"])
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train loss", "Val loss"])
    plt.show()

    plt.plot(model.history["accuracy"])
    plt.plot(model.history["val_accuracy"])
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train acc", "Val acc"])
    plt.show()

    plt.plot(model.history["lr"])
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()
    return


def get_fc_model(C):
    """ Simple fully connected model
    """    
    m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(C["s2_max_seq_len"], 2560)) ,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2560*C["s2_max_seq_len"] // 16, activation='relu'),
            tf.keras.layers.Dropout(C["s2_dropout"]),
            tf.keras.layers.Dense((2560*C["s2_max_seq_len"]) //16, activation='relu'),
            tf.keras.layers.Dropout(C["s2_dropout"]),
            tf.keras.layers.Dense((2560*C["s2_max_seq_len"]) // 16, activation='relu'),
            tf.keras.layers.Dense(C["num_classes"], activation='softmax')     
    ])  
    
    m.compile(  loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])
    return m
    


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
    
    classes_dict = {}
    for i, c in enumerate(C["sign_classes"]):
        classes_dict[c] = i                    # index into final output vector for each class
    print(f"Class Indices: {classes_dict}")

    C['sign_indices'] = classes_dict   
    C['num_classes'] = len(C["sign_classes"])


    traingen = Features_in(C, "train", shuffle=True)
    
    tstbatch = traingen.__getitem__(0)
     # tuple
    print(type(tstbatch), tstbatch[0].shape, tstbatch[0].dtype)
    print(tstbatch[1].shape, tstbatch[1].dtype)

    valgen = Features_in(C, "val", shuffle=False)
    valdata = valgen.__getitem__(0)   
    print("Val Input x", valdata[0].shape, valdata[0].dtype)
    print("Val Labels y", valdata[1].shape, valdata[1].dtype)

    testgen = Features_in(C, "test", shuffle=False)
    #tstbatch = testgen.__getitem__(0)
    #print("Input x", tstbatch[0].shape, tstbatch[0].dtype)
    #print("Labels y", tstbatch[1].shape, tstbatch[1].dtype)

    m = get_fc_model(C)
    
    # NOTE: to restore a previously trained model:
    #m = keras.models.load_model(a_checkpoint_file)
    
    #tst = m(tstbatch[0])

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=C["s2_monitor"], 
                                                     factor = C["s2_factor"], 
                                                     patience = C["s2_patience"], 
                                                     min_lr = C["s2_min_lr"],
                                                     verbose=1)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=C["s2_monitor"],
                                                      min_delta=C["s2_mindelta"], 
                                                      patience=C["s2_stop_patience"],
                                                      verbose=1,
                                                      restore_best_weights=True)
    
    #tensorbd = tf.keras.callbacks.TensorBoard(log_dir=C["dirs"]["tensorboard"]+"/{}",
    #                                            update_freq="epoch")  # How often to write logs ('batch' or 'epoch' or integer # batches.)
    
    # Path where to save the model
    # The two parameters below mean that we will overwrite
    # the current checkpoint if and only if
    # the `val_loss` score has improved.
    # The saved model name will include the current epoch.
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(C["dirs"]["ckpts"], "cs760model_loss-{loss:.2f}_{epoch}"),
    #                                                save_best_only=True,  # Only save a model if `val_loss` has improved.
    #                                                monitor="val_loss",
    #                                                verbose=1)
    
    
    history = m.fit(x = traingen, 
                    epochs = C["s2_max_epochs"],
                    initial_epoch = 0,
                    verbose = 1,
                    callbacks = [reduce_lr_on_plateau, early_stopping],
                    validation_data = valdata,
                    validation_batch_size = C["s2_batch_size_val"],
                    steps_per_epoch = traingen.__len__(),
                    validation_freq = 1,
                    max_queue_size = 10,
                    workers = 1,
                    use_multiprocessing = False)
    
    plots(history)
    
    best_epoch = np.argmax(history.history['val_accuracy'])
    print()
    print("#######################################################")
    print(f"Training Best Epoch: {best_epoch}  Train Acc: {history.history['accuracy'][best_epoch]} Val Acc:{history.history['val_accuracy'][best_epoch]}")
    print(f"Best Epoch: {best_epoch}")   
    print("#######################################################")
    print()
    
    evaluation = m.evaluate(x = testgen,
                            verbose = 1,
                            max_queue_size = 10,
                            workers = 1,
                            use_multiprocessing = False,
                            return_dict = True)
    print()
    print("#######################################################")
    print(f"Evaluation: {evaluation}")
    print("#######################################################")
    print()
    
    testpreds = m.predict(x = testgen,
                            verbose = 1,
                            max_queue_size = 10,
                            workers = 1,
                            use_multiprocessing = False)

    preds = testpreds.argmax(axis = 1)  # index of max value in each row is the predicted class
    gt = testgen.labels
    correct_per_class = np.zeros((len(C["sign_classes"])), dtype = np.int32)
    incorrect_per_class = np.zeros((len(C["sign_classes"])), dtype = np.int32)
    for i in range(len(gt)):
        if gt[i] == preds[i]:
            correct_per_class[gt[i]] += 1
        else:    
            incorrect_per_class[gt[i]] += 1
    print("Per-Class Predictions on Test Set")     
    for i in range(len(correct_per_class)):
        if correct_per_class[i] + incorrect_per_class[i] > 0:
            print(f'{i} {C["sign_classes"][i]}   Correct: {correct_per_class[i]}   Incorrect: {incorrect_per_class[i]}   % Correct: {(correct_per_class[i] / (correct_per_class[i] + incorrect_per_class[i]))*100}')



