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
- git+https://github.com/okankop/vidaug
- scikit-image

NOTE downloding data from videos:

You must create a directory called "vidoes" in the dataset directory or the videos will fail to download

To run CNN feature extractor:

1. cd to your models subdirectory
2. In config760.json, setting "crop_type": "T" (default) means crop vids to the top (front view) of each image and "B" crops to the bottom (side view). To get both you need to run the program twice, once for each crop type.
4. python extract_features.py video_directory feature_directory, the video and feature arguments are the location of the video and feature files. If one is missing or both are missing then the arguments will default to the self contained directories "../dataset/videos", "../features".
5. Ignore the various tensorflow messages, eventually the program will start processing videos. Each vid takes around 10 secs to process on my machine.


To create train/val/test split:
1. cd to your models subdirectory
2. Make sure the 'outdir' key in cs760.json points to the directory that the CNN feature extractor put the feature files into on your machine.
3. python train_val_test_split.py
4. The program will create subdirectories /train, /val, and /test under 'outdir' and copy (not move) the feature files into the respective subdirectories.
5. You should end up with 56 files in test, 1570 files in train and 84 files in val assuming you have run the CNN Feature extractor with crop_type "B" as well as crop_type "T".


