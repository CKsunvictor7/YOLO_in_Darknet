YOLOv3 in Darknet
=================
this repository is based on https://github.com/AlexeyAB/darknet  
it contains all necessary codes to train good YOLOv3 models

## For setup Enviroment
refer to https://github.com/AlexeyAB/darknet  
this repository has slight modified part of codes to change the period of saving models

## In this repository  
* YOLO_DB_Maker.py and DB_tools.py show how to make specified DB for training  
* the .cfg in exps show how to define the network and training hyper-parameters  
* the .data in exps show the path of data 
* the .names in exps desribe the name of categories
* use log_visualizer.py to monitor the training progress  
* after training, use darknet_detector.py to do the detection  
* data_cleaner.py does data clean and reindexing for raw user data using YOLOv3 models
* bbox_visualizer.py visualizes detection results and save them.

