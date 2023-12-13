This is the repo for the course project of CSCI 5525: Advanced Machine Learning in Fall 2023. 

Overview

This project is a re-implementation of YOLOv3 algorithm for object detection. It takes an image, 
and predicts several bounding boxes that contain the objects detected in the image. Each bounding box
is attached with a label to indicate the object type. 

Files

CSCI5525_course_project
|
|_____mAP/ : this is the directory containing codes for evaluation with mAP metrics
|
|_____coco_dataset/: this is the directory containing codes for COCO dataset parser; 
|
|_____myYOLOv3.py: this is the file containing the implementation of my custom YOLOv3 network
|
|_____train_yolov3.py: this is the main file to train the network
|
|_____test_yolov3.py: this is the main file to test the  network and conver the output into the needed formats for evaluation
|
|_____test_yolov3_shape.py: this is the file to test the correctness of the implementation of my YOLOv3 network
|
|_____VOCDataset.py: this is the file containing the dataset loader to load data from PASCAL VOC dataset
|
|_____yolov3_helper.py: some helper functions to facilitate implementations
|
|_____checkpoint.pth.tar: the pre-stored model parameters for the trained model 
|_____(If you cannot see it, you need to run train_yolov3.py to generate one)
|_____(WARNING: if you re-train the model, this will be replaced with the latest model you have trained!)
|
|_____mAP.png: the sample mAP results from evaluation
|
|_____loss_curve.png: the loss curve from the training


Prequisite:

To run the codes, you need the following libraries:

pytorch
albumentations 
tqdm
matplotlib
numpy
PIL

You will also probably need "cuda" resource. 

For the used dataset, you need to go to  to download the PASCAL VOC dataset from this website:
 https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video

This is a small dataset used for this project to train and test the model. You need to download the 
file and uncompress it under THIS DIRECTORY so that its path can be read by the dataset loader. 
You should have a folder called "PASCAL_VOC" if everything goes fine

If you want to explore more on COCO dataset, you can go to the directory "coco_dataset". You then run
dowload_coco.sh and unzip_coco.py in sequence. You should see three folders: coco_[train/test/ann]2017
if everything goes fine. You could run "visual_coco_dataset.py" for visualization. Unfortunately, I have
failed in going further in loading data from COCO dataset, so currently you cannot see the usage of COCO.


Usage

After you have UNZIPPED the dataset, you can re-train the model on yourself with
"python3 train_yolov3.py"

Or, you can test the model directly with 
"python3 test_yolov3.py"
Pay attention that the file is reading model parameters directly from checkpoint.pth.tar.
So, if you re-train the model on yourself and have updated the model parameters stored in it, 
you will probably have a different result. 

If you hope to see the mAP results, go to the directory "mAP", and type
"python3 main.py -na" 

Clarification
You won't find any robotic control codes in this repo. There are only codes for model training, becuase the 
robotic control codes are related to the intellectual properties belonging to our lab, and cannot be shared without
proper permission. Please look at the demo videos. 
