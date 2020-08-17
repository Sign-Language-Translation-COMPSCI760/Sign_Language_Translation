# Sign_Language_Translation

NOTE: Before pip install tensorflow you need to have the right version of cuda etc installed on your machine. Installation instructions start at: https://www.tensorflow.org/install but before actually installing read the gpu setup page and install the software prereqs for Windows or Linux: https://www.tensorflow.org/install/gpu


Required packages (use pip install):

- opencv-python
- tensorflow (you must have tensorflow >= 2.2)
- tensorflow_hub



NOTE downloding data from videos:

You must create a directory called "vidoes" in the dataset directory or the videos will fail to download

To run CNN feature extractor:

1. cd to your models subdirectory
2. Edit config760.json and set indir and outdir to the directory your videos are in and the dir you want to put the feature files into.
3. Also in config760.json, setting "crop_type": "T" (default) means crop vids to the top (front view) of each image and "B" crops to the bottom (side view). To get both you need to run the program twice, once for each crop type.
4. python extract_features.py
5. Ignore the various tensorflow messages, eventually the program will start processing videos. Each vid takes around 10 secs to process on my machine.



