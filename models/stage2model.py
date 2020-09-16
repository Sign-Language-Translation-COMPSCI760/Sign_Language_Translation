#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:40:07 2020

@author: tim

Stage 2 Model

Usage: Run from models subdirectory. 

    First take a copy of config_dirs.json as eg configdirs_yourname.json.
    Edit configdirs_yourname.json to set directories appropriately. 
    Particularly, set the "outdir" key to the root directory that train_val_test_split.py created /train /val and /test from. 
    
    To run transformer classifier:
        
    python stage2model.py config_dirs_yourname.json config760.json

    To run fully connected NN classifier:
        
    python stage2model.py config_dirs_yourname.json config760_fc1.json
    
    To run binary classifier to predict sign WORK-OUT:
     
    python stage2model.py config_dirs.json config760_binaryclassifier.json WORK-OUT    

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import random
from functools import partial

import cs760
import model_transformer



def rand_seed_all(seed=42):
    """ Attempt to make results reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return

def calc_frames(seqlen = [32, 50]):
    for seq in seqlen:
        print("For Max Seq Len:", seq)
        for num_frames in [13, 25, 32, 57, 69, 120, 136, 150, 180, 193]:
            #print(f"{num_frames}: max(num_frames//seq,1): {max(num_frames//seq, 1)} ({num_frames/max(num_frames//seq, 1)})")
            #print(f"{num_frames}: max(num_frames//seq+1,1): {max(num_frames//seq + 1, 1)} ({num_frames/(max(num_frames//seq, 1)+1)})")
            print(f"{num_frames}: max(round(num_frames/seq),1): {max(round(num_frames/seq), 1)} ({num_frames/(max(round(num_frames/seq), 1))})")
    

class Features_in(tf.keras.utils.Sequence):
    """ Return a batch of samples for model input. Called from .fit(...).
        There will be one instance of this class for each of train, val and test
        subdirs must be one of train, val or test
        Generally you will set shuffle = True for train and false for val and test..
    """
    def __init__(self, C, subdir = "train", shuffle = False):
        self.C = copy.deepcopy(C)
        if C["s2_classifier_type"] == 'softmax':
            self.classifier_type = 'softmax'
            self.curr_sign = ""
        else:
            self.classifier_type = 'sigmoid'
            self.curr_sign = C["curr_sign"]

        self.input_dir = os.path.join(C["dirs"]["dict_pkls"], subdir)
        self.subdir = subdir
        
        if subdir == "train":
            filepattern = C["s2_traintypes"]
            self.batch_size = C["s2_batch_size_train"]
            self.take_frame = C["s2_train_take_frame"]   # -1 for calculate dynamically based on number of frames
            self.restrictto = C["s2_train_restrictto"]
        elif subdir == "val":
            filepattern = C["s2_valtypes"]
            self.batch_size = 99999999  #C["s2_batch_size_val"]  Weirdly .fit() doesnt allow a generator for validation so read entire val set back into np arrays
            self.take_frame = C["s2_val_take_frame"]   # -1 for calculate dynamically based on number of frames
            self.restrictto = C["s2_val_restrictto"]
        elif subdir == "test":
            filepattern = C["s2_testtypes"]
            self.batch_size = C["s2_batch_size_test"]
            self.take_frame = C["s2_test_take_frame"]   # -1 for calculate dynamically based on number of frames
            self.restrictto = C["s2_test_restrictto"]
        else:
            assert True == False, f"FeaturesIn ERROR: Unknown subdir name {subdir}! Unable to proceed"
            
        patterns = ['*' + pattern + '.pkl' for pattern in filepattern]    
        self.filenames = cs760.list_files_multipatterns(self.input_dir, patterns)
        
        if C["s2_restrictto_ornot"]:
            if self.restrictto != "":  # eliminate filenames not matching restrictto eg "__US" 
                self.filenames = [f for f in self.filenames if f.upper().find(self.restrictto) != -1]
        else:        
            if self.restrictto != "":  # eliminate filenames matching restrictto eg "__US" 
                self.filenames = [f for f in self.filenames if f.upper().find(self.restrictto) == -1]
        
        if shuffle:
            rand_seed_all(C["s2_random_seed"])  #should now get same results when run with same random seed.
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
            if self.classifier_type == 'softmax':
                self.labels.append(C["sign_indices"][l])    #append indix for label, not the label itself
            else:                                           # binary classifier: append 1 for the target sign, 0 for all others
                if l == self.curr_sign:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        assert len(self.labels) == len(self.filenames), "FeaturesIn ERROR: number of labels must equal number of input files. Unable to proceed."
        
        if self.classifier_type != 'softmax' and self.subdir == 'train':       #build list of positive and negative samples for balanced sampling
            self.filenames_positive = []
            self.filenames_negative = []
            self.labels_positive = []
            self.labels_negative = []            
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    self.filenames_negative.append(self.filenames[i])
                    self.labels_negative.append(self.labels[i])
                else:
                    self.filenames_positive.append(self.filenames[i])
                    self.labels_positive.append(self.labels[i])
            self.takepositives = int(np.floor(self.batch_size * C["s2_positives_ratio"]))        
            self.takenegatives = self.batch_size - self.takepositives     #int(np.ceil(self.batch_size * (1-C["s2_positives_ratio"])))        
            assert self.takepositives + self.takenegatives == self.batch_size, f"ERROR: invalid s2_positives_ratio {C['s2_positives_ratio']} relative to batch size {self.batch_size}"        

        self.maxseqlen = C["s2_max_seq_len"]   # pad or truncate to this seq len

        if self.classifier_type == 'softmax':
            self.num_classes = C["num_classes"]
        else:
            self.num_classes = 1
        return
    

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.classifier_type != 'softmax' and self.subdir == 'train':
            batch_x = []
            batch_y = []
            num_positives = len(self.labels_positive)-1
            num_negatives = len(self.labels_negative) - 1
            for i in range(self.takepositives):
                j = random.randint(0, num_positives)
                batch_x.append(self.filenames_positive[j])
                batch_y.append(self.labels_positive[j])
            for i in range(self.takenegatives):
                j = random.randint(0, num_negatives)
                batch_x.append(self.filenames_negative[j])
                batch_y.append(self.labels_negative[j])
            batch_y = np.array(batch_y, dtype=np.float32)
        else:
            batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size], dtype=np.float32)

        batch_list = []        
        for file in batch_x:
            sample = cs760.loadas_pickle(os.path.join(self.input_dir, file))
            if self.take_frame == -1:
                take_frame = max(round(sample.shape[0] / self.maxseqlen), 1)
            else:
                take_frame = self.take_frame
            sample = sample[0::take_frame]                # only take every nth frame
            if sample.shape[0] > self.maxseqlen:          # truncate features to maxseqlen
                sample = sample[0:self.maxseqlen]
            elif sample.shape[0] < self.maxseqlen:        # pad features to maxseqlen
                sample_padded = np.zeros((self.maxseqlen, self.C["cnn_feat_dim"]), dtype=np.float32)
                sample_padded[0:sample.shape[0]] = sample
                sample = sample_padded
            if C["s2_model_type"] in ['tc2']:           # to make transformer work, need to pad input so conv layer outputs 2560
                sample_padded = np.zeros((self.maxseqlen, self.C["s2_pad_to"]), dtype=np.float32)
                sample_padded[:, 0:self.C["cnn_feat_dim"]] = sample
                sample = sample_padded
                    
            batch_list.append(sample)
        batch_np = np.array(batch_list, dtype = np.float32)    
        if C["s2_model_type"] in ['tc2']:           # to make conv work, need to add final dim
            batch_np = np.expand_dims(batch_np, -1)
                            
        if self.classifier_type == 'softmax':
            batch_y = tf.keras.utils.to_categorical(batch_y, self.num_classes)
        return batch_np, batch_y
    

def output_perclass(C, m, gen, verbose=0):
    """ Output predictions per class for 70 class classifier
    """
    preds = m.predict(  x = gen,
                        verbose = 1,
                        max_queue_size = 10,
                        workers = 1,
                        use_multiprocessing = False)

    preds = preds.argmax(axis = 1)  # index of max value in each row is the predicted class
    gt = gen.labels
    correct_per_class = np.zeros((len(C["sign_classes"])), dtype = np.int32)
    incorrect_per_class = np.zeros((len(C["sign_classes"])), dtype = np.int32)
    for i in range(len(gt)):
        if gt[i] == preds[i]:
            correct_per_class[gt[i]] += 1
        else:    
            incorrect_per_class[gt[i]] += 1

    print("Per-Class Predictions:")     
    if verbose > 1:
        for i in range(len(correct_per_class)):
            if correct_per_class[i] + incorrect_per_class[i] > 0:   # exclude outputs for classes that don't exist in this dataset
                print(f'{i} {C["sign_classes"][i]}   Correct: {correct_per_class[i]}   Incorrect: {incorrect_per_class[i]}   % Correct: {(correct_per_class[i] / (correct_per_class[i] + incorrect_per_class[i]))*100}')

    correct_list = []        
    incorrect_list = []    
    for i in range(len(correct_per_class)):
        if correct_per_class[i] + incorrect_per_class[i] > 0:   # exclude outputs for classes that don't exist in this dataset
            if correct_per_class[i] >= 1:
                correct_list.append(C["sign_classes"][i])
            if incorrect_per_class[i] >= 1:
                incorrect_list.append(C["sign_classes"][i])
    print("#############################################################")
    print(f"Classes with at least one INCORRECT: {incorrect_list}")
    print("#############################################################")
    print(f"Classes with at least one CORRECT: {correct_list}")
    print("#############################################################")

    return preds


def output_perclass_binary(C, m, gen, verbose=0):
    """ Output predictions per class for binary classifier
    """
    sign = C["curr_sign"]
    preds = m.predict(  x = gen,
                        verbose = 1,
                        max_queue_size = 10,
                        workers = 1,
                        use_multiprocessing = False)
    pred_scores = np.where(preds < C["s2_classifier_thresh"], 0, 1)
    preds = pred_scores.squeeze()
    gt = gen.labels
    correct_per_class = 0
    incorrect_per_class = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(gt)):
        if gt[i] == preds[i]:
            correct_per_class += 1
            if gt[i] == 1:
                TP += 1
            else:
                TN += 1
        else:    
            incorrect_per_class += 1
            if gt[i] == 1:
                FN += 1
            else:
                FP += 1

    print(f"{sign} Predictions:")
    print("#############################################################")
    print(f"{sign} TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}")
    print(f"{sign} Predictions CORRECT: {TP}")
    print("#############################################################")

    return preds, (TP, TN, FP, FN)



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
    
    if C["s2_classifier_type"] == 'softmax':
        num_classes = C["num_classes"]
    else:
        num_classes = 1
        
    if C["s2_regularizer"] > 0.0:
        regul = tf.keras.regularizers.L2(l2=C["s2_regularizer"])
    else:
        regul = None
        
    m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(C["s2_max_seq_len"], C["cnn_feat_dim"])) ,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(C["cnn_feat_dim"]*C["s2_max_seq_len"] // 16, activation='relu', kernel_regularizer=regul),
            tf.keras.layers.Dropout(C["s2_dropout"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense((C["cnn_feat_dim"]*C["s2_max_seq_len"]) // 16, activation='relu', kernel_regularizer=regul),
            tf.keras.layers.Dropout(C["s2_dropout"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense((C["cnn_feat_dim"]*C["s2_max_seq_len"]) // 16, activation='relu', kernel_regularizer=regul),
            tf.keras.layers.Dense(num_classes, activation=C["s2_classifier_type"])     
    ])  
    
    opt = tf.keras.optimizers.Adam(learning_rate=C["s2_lr"])  # adam default = 0.001
    
    if C["s2_classifier_type"] == 'softmax':
        m.compile(  loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'])
    else:
        m.compile(  loss="binary_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'])
        
    print(m.summary())
    return m


def get_transclassifier_model(C):
    """ Classifying Transformer model
    """    
    if C["s2_classifier_type"] == 'softmax':
        num_classes = C["num_classes"]
    else:
        num_classes = 1

    if C["s2_regularizer"] > 0.0:
        regul = tf.keras.regularizers.L2(l2=C["s2_regularizer"])
    else:
        regul = None

    if C["s2_model_type"] == 'tc1':   
        m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(C["s2_max_seq_len"], C["cnn_feat_dim"])),
                model_transformer.TransformerEncoder(encoder_count=C["s2_encoder_count"],
                                                     attention_head_count=8, 
                                                     d_model=C["cnn_feat_dim"], 
                                                     dropout_prob=C["s2_dropout"], 
                                                     add_pos_enc=C["s2_add_pos_enc"],
                                                     regul=regul),
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),   #less variation when this line is here but don't get the really good accuracies (0.5)
                tf.keras.layers.Dense(num_classes, activation=C["s2_classifier_type"])     
        ])  
    elif C["s2_model_type"] == 'tc2':   #tc2
        m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(C["s2_max_seq_len"], C["s2_pad_to"], 1)),
                tf.keras.layers.Conv2D(1, C["s2_kernel_size"], strides=C["s2_strides"], activation='relu', kernel_regularizer=regul),
                tf.keras.layers.Dropout(C["s2_dropout"]),
                model_transformer.TransformerEncoder(encoder_count=C["s2_encoder_count"],
                                                     attention_head_count=8, 
                                                     d_model=C["cnn_feat_dim"],       #C["cnn_feat_dim"], 
                                                     dropout_prob=C["s2_dropout"], 
                                                     add_pos_enc=C["s2_add_pos_enc"],
                                                     regul=regul),
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),   #less variation when this line is here but don't get the really good accuracies (0.5)
                tf.keras.layers.Dense(num_classes, activation=C["s2_classifier_type"])     
        ]) 
    else:  # tc3
        if C.get("s2_tc_activation") is None:
            C["s2_tc_activation"] = 'relu'
            print(f"USING DEFAULT RELU")
        if C["s2_tc_activation"] == 'relu':
            activ = tf.keras.activations.relu
        elif C["s2_tc_activation"] == 'selu':
            activ = tf.keras.activations.selu
        elif C["s2_tc_activation"] == 'leaky_relu':
            activ = partial(tf.keras.activations.relu, alpha=0.2)
        elif C["s2_tc_activation"] == 'elu':
            activ = tf.keras.activations.elu
        elif C["s2_tc_activation"] == 'swish':
            activ = tf.keras.activations.swish
        print(f"USING ACTIVATION FN {C['s2_tc_activation']}")    
        m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(C["s2_max_seq_len"], C["cnn_feat_dim"])),
                tf.keras.layers.Dense(C["s2_emb_dim"], activation=activ, kernel_regularizer=regul),  # pseudo embedding dim. Note inputting eg [bs=10, seqlen=32, 2560] into this dense layer will output [10, 32, emb_dim]
#                tf.keras.layers.Dropout(C["s2_dropout"]),
                model_transformer.TransformerEncoder(encoder_count=C["s2_encoder_count"],
                                                     attention_head_count=8, 
                                                     d_model=C["s2_emb_dim"], 
                                                     dropout_prob=C["s2_dropout"], 
                                                     add_pos_enc=C["s2_add_pos_enc"],
                                                     regul=regul, activ=C["s2_tc_activation"]),
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),   #less variation when this line is here but don't get the really good accuracies (0.5)
                tf.keras.layers.Dense(num_classes, activation=C["s2_classifier_type"])     
        ])  
            
            
                
    
    opt = tf.keras.optimizers.Adam(learning_rate=C["s2_lr"])  # adam default = 0.001
    
    if C["s2_classifier_type"] == 'softmax':
        m.compile(  loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'])
    else:
        m.compile(  loss="binary_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'])
    print(m.summary())
    return m


def train_eval_one_sign_binary(C):
    """ Train evaluate a single sign using a binary classifier
        The sign to train for is specified in C["curr_sign"] before calling
    """
    sign = C["curr_sign"]
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
    testbatch = testgen.__getitem__(0)
    print("Input x", testbatch[0].shape, testbatch[0].dtype)
    print("Labels y", testbatch[1].shape, testbatch[1].dtype)

    if C["s2_model_type"] == "fc1":
        m = get_fc_model(C)
    elif C["s2_model_type"] == "tc1":
        m = get_transclassifier_model(C)
    else:
        assert True==False, f"ERROR: Unknown s2_model_type in config file: {C['s2_model_type']}. Must be one of fc1 or tc1"
    
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
    #plots(history)   
    best_epoch = np.argmax(history.history['val_accuracy'])  # history is zero-based but keras screen output 1st epoch is epoch 1
    print()
    print("#######################################################")
    print(f"{sign} Training Best Epoch: {best_epoch+1}  Train Acc: {history.history['accuracy'][best_epoch]} Val Acc:{history.history['val_accuracy'][best_epoch]}")
    print(f"Best Epoch (1-based): {best_epoch+1}")   
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
    print(f"{sign} Evaluation: {evaluation}")
    print("#######################################################")
    print()
    print("PREDICTIONS ON VAL:")
    valpreds, valstats = output_perclass_binary(C, m, gen=valgen)
    print("PREDICTIONS ON TEST (NZSL):")
    testpreds, teststats = output_perclass_binary(C, m, gen=testgen)
    tf.keras.backend.clear_session()  #TODO doesnt work
    #del m
    return testpreds, teststats, valpreds, valstats
    

def train_eval_binary(C):
    """ Train one binary classifier per sign in test set
    """
    sign_dict_test = {}
    sign_dict_val = {}
    for sign in C["nzsl_signs"]:
        print(f'Training Binary classifier for sign {sign}')
        C["curr_sign"] = sign
        testpreds, (TP,TN,FP,FN), valpreds, (TP_val,TN_val,FP_val,FN_val) = train_eval_one_sign_binary(C)
        sign_dict_test[sign] = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
        sign_dict_val[sign] = {'TP':TP_val, 'TN':TN_val, 'FP':FP_val, 'FN':FN_val}

    print(f"Sign Binary Classification on VAL (BOSTON) Summary: {sign_dict_val}")    
    correct_signs = 0
    for sign in sign_dict_val:
        if sign_dict_val[sign]['TP'] >= 1:
            correct_signs += 1
    print(f"OVERALL ACCURACY ON VAL (BOSTON) USING BINARY CLASSIFICATION: {correct_signs/len(sign_dict_test)}  {correct_signs} of {len(sign_dict_test)} correctly predicted")

        
    print(f"Sign Binary Classification on TEST (NZSL) Summary: {sign_dict_test}")    
    correct_signs = 0
    for sign in sign_dict_test:
        if sign_dict_test[sign]['TP'] >= 1:
            correct_signs += 1
    print(f"OVERALL ACCURACY ON TEST (NZSL) USING BINARY CLASSIFICATION: {correct_signs/len(sign_dict_test)}  {correct_signs} of {len(sign_dict_test)} correctly predicted")
        
    return


def train_eval_softmax(C):
    """ Train a 70 class softmax classifier
    """
    
    traingen = Features_in(C, "train", shuffle=True)
    
    tstbatch = traingen.__getitem__(0)
     # tuple
    print(type(tstbatch), tstbatch[0].shape, tstbatch[0].dtype)
    print(tstbatch[1].shape, tstbatch[1].dtype)   # [bs, seq_len, feat_dim]

    valgen = Features_in(C, "val", shuffle=False)
    valdata = valgen.__getitem__(0)   
    print("Val Input x", valdata[0].shape, valdata[0].dtype)
    print("Val Labels y", valdata[1].shape, valdata[1].dtype)

    testgen = Features_in(C, "test", shuffle=False)
    #tstbatch = testgen.__getitem__(0)
    #print("Input x", tstbatch[0].shape, tstbatch[0].dtype)
    #print("Labels y", tstbatch[1].shape, tstbatch[1].dtype)

    if C["s2_model_type"] == "fc1":
        m = get_fc_model(C)
    elif C["s2_model_type"] in ["tc1","tc2", "tc3"]:
        m = get_transclassifier_model(C)
    else:
        assert True==False, f"ERROR: Unknown s2_model_type in config file: {C['s2_model_type']}. Must be one of fc1 or tc1"
    
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
    
    best_epoch = np.argmax(history.history['val_accuracy'])  # history is zero-based but keras screen output 1st epoch is epoch 1
    print()
    print("#######################################################")
    print(f"Training Best Epoch: {best_epoch+1}  Train Acc: {history.history['accuracy'][best_epoch]} Val Acc:{history.history['val_accuracy'][best_epoch]}")
    print(f"Best Epoch (1-based): {best_epoch+1}")   
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


    print("PER-CLASS PREDICTIONS ON TEST (NZSL):")
    testpreds = output_perclass(C, m, gen=testgen, verbose=C["s2_verbose"])    
    print()
    print("#######################################################")    
    print("PER-CLASS PREDICTIONS ON VAL (BOSTON):")
    valpreds = output_perclass(C, m, gen=valgen, verbose=C["s2_verbose"])
    return testpreds, valpreds    
    


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


    
    if C["s2_classifier_type"] == 'softmax':
        testpreds, valpreds = train_eval_softmax(C)
    else:
        #testpreds, valpreds = train_eval_binary(C)  #tf throws OOM when try to reload the model so have to do it one sign at a time
        testpreds, (TP,TN,FP,FN), valpreds, (TP_val,TN_val,FP_val,FN_val) = train_eval_one_sign_binary(C)
        datastr = C["curr_sign"] + "," + "test_TPTNFPFN," + str(TP) + "," + str(TN) + "," + str(FP) + "," + str(FN) + ",val_TPTNFPFN," + str(TP_val) + "," + str(TN_val) + "," + str(FP_val) + "," + str(FN_val) + '\n'
        with open('binary_classifier_results.txt', 'a') as f:
            charswritten = f.write(datastr)
    



