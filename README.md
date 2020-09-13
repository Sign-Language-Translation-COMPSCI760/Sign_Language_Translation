# Sign_Language_Translation

NOTE: Before pip install tensorflow you need to have the right version of cuda etc installed on your machine. Installation instructions start at: https://www.tensorflow.org/install but before actually installing read the gpu setup page and install the software prereqs for Windows or Linux: https://www.tensorflow.org/install/gpu

# Tensorflow setup (WINDOWS)

- First make sure your python install is 64 bit, tensorflow does not support 32 bit.
- Download and CLEAN install Nvidia driver 418.91 (https://www.nvidia.com/Download/driverResults.aspx/142736/en-us).
- Download and install CUDA 10.1 (https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork).
- Download cuDNN 7.6.0 (https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork), this will require a Nvidia developer account to download (https://developer.nvidia.com/).
- Move the files from cuDNN directories to the Nvidia CUDA directories of the same name (CUDA files located: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1")
- Edit path variables to include...
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin,
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64,
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include,
  C:\tools\cuda\bin
- Using what ever terminal you use to run python programs (VSCode, PowerShell, Pycharm etc), pip install tensorflow.

# Model requirements

Required packages (use pip install):

- opencv-python
- tensorflow (you must have tensorflow >= 2.2)
- tensorflow_hub
- numpy
- matplotlib
- scikit-image
- imgaug

NOTE To train models you just need to follow the instructions below to download the final dataset and run/view stage 2 model.

# To run CNN feature extractor:

1. cd to your models subdirectory
2. In config760.json, setting "crop_type": "T" (default) means crop vids to the top (front view) of each image and "B" crops to the bottom (side view). To get both you need to run the program twice, once for each crop type.
4. python extract_features.py video_directory feature_directory, the video and feature arguments are the location of the video and feature files. If one is missing or both are missing then the arguments will default to the self contained directories "../dataset/videos", "../features".
5. Ignore the various tensorflow messages, eventually the program will start processing videos. Each vid takes around 10 secs to process on my machine.


# To create train/val/test split  (initial dataset only):
1. cd to your models subdirectory
3. python train_val_test_split.py video_directory feature_directory, the video and feature arguments are the location of the video and feature files. If one is missing or both are missing then the arguments will default to the self contained directories "../dataset/videos", "../features".
4. The program will create subdirectories /train, /val, and /test under feature_directory and copy (not move) the feature files into the respective subdirectories.
5. You should end up with 56 files in test, 1570 files in train and 84 files in val assuming you have run the CNN Feature extractor with crop_type "B" as well as crop_type "T".


# To download/set up the final dataset on your machine:
I've created a subfolder on the shared drive called "datasets". Inside that are val.tar.gz, test.tar.gz and train.tar.gz. You need to download them and extract them into into a folder structure as follows:
c:\.....\your_dataset_subdir\val
c:\.....\your_dataset_subdir\test
c:\.....\your_dataset_subdir\train

Then take a copy of and edit the /models/config_dirs.json file and set the "dict_pkls" key to the root directory of your dataset (the directory that has subdirectories /val /test and /train)



# To run/view stage 2 model (fc1 takes arounf 45 mins to train with TOP + 1 other aug type. tc1 takes under 10 minutes to fully train and run evaluation):

Before doing anything, I suggest looking at the stage2model.py code in conjunction with looking at config760.json so you can see what it's doing. The majority of your questions around what the keras and tensorflow bits are doing can probably be answered by looking at https://www.tensorflow.org/guide/keras/train_and_evaluate and otherwise googling for other parts of the tensorflow documentation. The trickiest bit is probably the class Features_in(tf.keras.utils.Sequence) - googling tf.keras.utils.Sequence and/or tf.fit will likely give you the idea of how this works.

To actually run training/eval:

1. First take a copy of config_dirs.json as eg configdirs_yourname.json.
   Edit configdirs_yourname.json to set directories appropriately. 
   Particularly, set the "dict_pkls" key to the root directory of your dataset (the directory that has subdirectories /val /test and /train). 
   
2. Similarly, take a copy of the config760.json file as eg config760_yourname.json and edit the parameters starting with "s2" - more on this below*

3. Run from models subdirectory. 
4. To run  classifier with a reasonable set of parameters:
        
    python stage2model.py config_dirs_yourname.json config760.json

*config760.json parameters:

	"augmentation_type": ["__HFLIP", "__ROTAT", "__CROP", "__BRIGHT", "__HUE", "__SATUR", "__CONTOUR", "__NOISE", "__INVERT"],
	"s2_traintypes": ["__TOP", "__BOT", "__HFLIP"],   # Add remove augmentation types (see above list) that will be trained on
	"s2_valtypes": ["__TOP"],                         # aug types to validate on - generally keep to __TOP
	"s2_testtypes": ["__TOP"],                        # aug   "    "  test    "      "        "    "  "
	"s2_train_restrictto" : "__US",			  # if "s2_restrictto_ornot" : false then don't include files with __US in their name (ASL dict vids)
	"s2_val_restrictto" : "",
	"s2_test_restrictto" : "",
	"s2_restrictto_ornot" : false,
	"s2_model_type" : "fc1",                          # "fc1" for fully connected model or "tc1", "tc2" or "tc3" for transformer models 
	"s2_classifier_type" : "softmax",                 # 'softmax' for 70 class classifier, 'sigmoid' for binary classifier
	"s2_classifier_thresh" : 0.5,                     # binary classifier threshold value: > this predicts positive class, < predicts negative class 
	"s2_positives_ratio" : 0.6,  # ratio of positive class to negative class samples to select for training (without this, massive class imbalance to negs)
	"s2_batch_size_train": 10,                        # training batch size
	"s2_batch_size_val": 10,                          # val bs
	"s2_batch_size_test": 10,                         # test bs
	"s2_max_seq_len" : 32,                            # the max number of featurised video frames to take from the input .pkl file sample
	"s2_take_frame" : -1,           # if -1, will calculate how many frames to skip over in each input sample to cover the whole vid but with s2_max_seq_len frames.
                                  # if > 1 the fixed number of frames to skip over
	"s2_dropout" : 0.1,             # dropout percentage
	"s2_regularizer" : 0.005,       # l2 regularization amount for fc layers
	"s2_min_lr" : 0.000001,         # min learning rate - stops when hits this
	"s2_patience" : 6,              # the number of epochs to run without improvement before reducing the learning rate
	"s2_factor" : 0.5,              # the factor to reduce the lr by after s2_patience is exhausted
	"s2_monitor" : "val_accuracy",  # the metric to monitor for lr decay
	"s2_mindelta" : 0.0001,         # the minimum diff in lr before it's counted as an improvement
	"s2_stop_patience" : 13,        # the number of epochs to run without improvement before stopping
	"s2_lr" : 0.0005,               # initial learning rate
	"s2_max_epochs" : 500,          # max epochs to train before stopping if early stopping doesn't kick in earlier (it always does)
	"s2_random_seed" : 42,          # Running with this random seed (and same other parameters) should produce identical results. Only use 42 and 101
	"s2_add_pos_enc" : true,        # Transformer model only - add positional encoding (got much worse results with this off)
	"s2_encoder_count" : 2,         # Transformer only - number of Encoder layers

