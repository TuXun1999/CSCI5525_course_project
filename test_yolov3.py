import os
import torch 

import torch.optim as optim 
import torchvision

from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2 
  
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import os 
import numpy as np 
  
import shutil
  
#from COCOParser import COCOParser
#from COCODataset import COCODataset
from VOCDataset import VOCDataset
from yolov3_helper import convert_cells_to_bboxes, plot_image, load_checkpoint
from myYOLOv3 import myYOLOv3
# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
  
# Load and save model variable 
load_model = True
save_model = False

plot_test_image = False
# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"
  
# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]
  

# Learning rate for training (used in loading the model)
learning_rate = 1e-5

# Image size 
image_size = 416

# Defining the grid size and the scaled anchors 
GRID_SIZE = [13, 26, 52] 
scaled_anchors = torch.tensor(ANCHORS) / ( 
    1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
) 

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 


# Class labels 
class_labels = [ 
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

  
# Transform for testing 
test_transform = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=image_size), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Normalize the image 
        A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ],
    # Augmentation for bounding boxes  
    bbox_params=A.BboxParams( 
                    format="yolo",  
                    min_visibility=0.2,  
                    label_fields=[] 
                ) 
)


  
# Setting the load_model to True 
load_model = True
  
# Defining the model, optimizer, loss function and scaler 
model = myYOLOv3(num_classes=20).to(device) 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 

scaler = torch.cuda.amp.GradScaler() 
  
# Loading the checkpoint 
if load_model: 
    load_checkpoint(checkpoint_file, model, optimizer, learning_rate) 

'''
coco_annotations_file=  os.getcwd() + "/coco_ann2017/annotations/instances_val2017.json"
coco_images_dir= os.getcwd()+ "/coco_val2017/val2017"
coco= COCOParser(coco_annotations_file, coco_images_dir)
# Creating a dataset object 
test_dataset = COCODataset( 
    coco_annotations_file,
    coco_images_dir,
    coco_parser=coco,
    grid_sizes=GRID_SIZE, 
    anchors=ANCHORS, 
    transform=test_transform
) 
  
# Creating a dataloader object 
test_loader = torch.utils.data.DataLoader( 
    test_dataset, 
    batch_size = batch_size, 
    num_workers = 2, 
    shuffle = True, 
    pin_memory = True, 
) 
''' 
# Defining the test dataset and data loader 
test_dataset = VOCDataset( 
    csv_file="./PASCAL_VOC/test.csv", 
    image_dir="./PASCAL_VOC/images/", 
    label_dir="./PASCAL_VOC/labels/", 
    anchors=ANCHORS, 
    transform=test_transform,
    is_test = True
) 
test_loader = torch.utils.data.DataLoader( 
    test_dataset, 
    batch_size = 1, 
    num_workers = 0, 
    shuffle = True
) 

test_batch_size = 50

# Specify the target directory
det_dir = "mAP/input/detection-results"
gt_dir = "mAP/input/ground-truth"

if os.path.exists(det_dir):
    shutil.rmtree(det_dir)
os.mkdir(det_dir)
if os.path.exists(gt_dir):
    shutil.rmtree(gt_dir)
os.mkdir(gt_dir)


for i in range(test_batch_size):
    
    file_name = "image-" + str(i) + ".txt"
    
    # Getting a sample image from the test data loader 
    x, y = next(iter(test_loader)) 
    x = x.to(device) 

    gt_bboxes = list(y)
    gt_bboxes = torch.tensor(gt_bboxes)
    N = gt_bboxes.shape[0]
    gt_bboxes = torch.hstack((gt_bboxes[:, 4].view(-1, 1),  torch.ones(N, 1), gt_bboxes[:, 0:4]))
    model.eval() 
    with torch.no_grad(): 
        # Getting the model predictions 
        output = model(x)

        # Getting the bounding boxes from the predictions & gt targets
        bboxes = [[] for _ in range(x.shape[0])] 
        anchors = ( 
                torch.tensor(ANCHORS) 
                    * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
                ).to(device) 
    
        # Getting bounding boxes for each scale 
        for i in range(3): 
            batch_size, A, S, _, _ = output[i].shape 
            anchor = anchors[i] 
            boxes_scale_i = convert_cells_to_bboxes( 
                                output[i], anchor, s=S, is_predictions=True
                            )
        
            for idx, (box) in enumerate(boxes_scale_i): 
                bboxes[idx] += box
    
    # Plotting the image with bounding boxes for each image in the batch (one image in the batch)
    for i in range(batch_size): 
        threshold = 0.65
        filter_boxes = [box for box in bboxes[i] if box[1] > threshold] 
        try:
            boxes_old = torch.tensor(filter_boxes)[:, 2:]

            # Convert the format from (cx, cy, h, w) into (x, y, x, y)
            # so that the built-in NMS method of torchvision can be implemented
            boxes = torch.ones(boxes_old.shape).to(device=device)
            boxes[:, 0] = boxes_old[:, 0] - boxes_old[:, 2]/2
            boxes[:, 1] = boxes_old[:, 1] - boxes_old[:, 3]/2
            boxes[:, 2] = boxes_old[:, 0] + boxes_old[:, 2]/2
            boxes[:, 3] = boxes_old[:, 1] + boxes_old[:, 3]/2

            scores = torch.tensor(filter_boxes)[:, 1].to(device)
            # Applying non-max suppression to remove overlapping bounding boxes 
            keep_idx = torchvision.ops.nms(boxes, scores=scores, iou_threshold=0.3) 

            # Only consider the filtered boxes after NMS suppression
            nms_boxes = list(np.array(filter_boxes)[np.array(keep_idx.to('cpu'))])
            
            # Plotting the image with bounding boxes 
            if plot_test_image:
                plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
        except:
            print("No box with sufficiently high confidence score is found in this image...")
        
        if plot_test_image:
            plot_image(x[i].permute(1,2,0).detach().cpu(), gt_bboxes)


        # Write the detection & gt results to the mAP module for evaluation
        h = x[i].shape[1]
        w = x[i].shape[2]
        nms_boxes = torch.tensor(np.array(nms_boxes))

        # Save the image to the directory

        with open(os.path.join(det_dir, file_name), "w") as f_det, open(
                os.path.join(gt_dir, file_name), "w"
            ) as f_gt:
            # Convert our prediction results into the formats accpeted by 
            # the used mAP module
            pred_boxes = torch.zeros(nms_boxes.shape).to(device=device)
            pred_boxes[:, 0] = nms_boxes[:, 0]
            pred_boxes[:, 1] = nms_boxes[:, 1]
            pred_boxes[:, 2] = (nms_boxes[:, 2] - nms_boxes[:, 4]/2) * w
            pred_boxes[:, 3] = (nms_boxes[:, 3] - nms_boxes[:, 5]/2) * h
            pred_boxes[:, 4] = (nms_boxes[:, 2] + nms_boxes[:, 4]/2) * w
            pred_boxes[:, 5] = (nms_boxes[:, 3] + nms_boxes[:, 5]/2) * h

            target_boxes = torch.zeros(gt_bboxes.shape).to(device=device)
            target_boxes[:, 0] = gt_bboxes[:, 0]
            target_boxes[:, 1] = (gt_bboxes[:, 2] - gt_bboxes[:, 4]/2) * w
            target_boxes[:, 2] = (gt_bboxes[:, 3] - gt_bboxes[:, 5]/2) * h
            target_boxes[:, 3] = (gt_bboxes[:, 2] - gt_bboxes[:, 4]/2) * w
            target_boxes[:, 4] = (gt_bboxes[:, 3] + gt_bboxes[:, 5]/2) * h

            for b in target_boxes:
                f_gt.write(
                    f"{class_labels[int(b[0])]} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} {b[4]:.2f}\n"
                )
            for b in pred_boxes:
                f_det.write(
                    f"{class_labels[int(b[0])]} {b[1]:.6f} {b[2]:.2f} {b[3]:.2f} {b[4]:.2f} {b[5]:.2f}\n"
                )



