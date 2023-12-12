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
  
import matplotlib.pyplot as plt 
  
from tqdm import tqdm
#from COCOParser import COCOParser
#from COCODataset import COCODataset
from VOCDataset import VOCDataset
from yolov3_helper import save_checkpoint, load_checkpoint, convert_cells_to_bboxes, plot_image
from myYOLOv3 import myYOLOv3,myYOLOLoss
# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
  
# Load and save model variable 
load_model = False
save_model = True
  
# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"
  
# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 
  
# Batch size for training 
batch_size = 32
  
# Learning rate for training 
learning_rate = 1e-5
  
# Number of epochs for training 
epochs = 100
  
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

# Transform for training 
train_transform = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=image_size), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Random color jittering 
        A.ColorJitter( 
            brightness=0.5, contrast=0.5, 
            saturation=0.5, hue=0.5, p=0.5
        ), 
        # Flip the image horizontally 
        A.HorizontalFlip(p=0.5), 
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


'''
coco_annotations_file=  os.getcwd() + "/coco_ann2017/annotations/instances_train2017.json"
coco_images_dir= os.getcwd()+ "/coco_train2017/train2017"
coco= COCOParser(coco_annotations_file, coco_images_dir)
# Creating a dataset object 
train_dataset = COCODataset( 
    coco_annotations_file,
    coco_images_dir,
    coco_parser=coco,
    grid_sizes=GRID_SIZE, 
    anchors=ANCHORS, 
    transform=train_transform
) 
  
# Creating a dataloader object 
train_loader = torch.utils.data.DataLoader( 
    train_dataset, 
    batch_size = batch_size, 
    num_workers = 0, 
    shuffle = True, 
    pin_memory = True, 
) 
'''

# Defining the train dataset 
train_dataset = VOCDataset( 
    csv_file="./PASCAL_VOC/train.csv", 
    image_dir="./PASCAL_VOC/images/", 
    label_dir="./PASCAL_VOC/labels/", 
    anchors=ANCHORS, 
    transform=train_transform,
    is_test = False
) 
  
# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
    train_dataset, 
    batch_size = batch_size, 
    num_workers = 0, 
    shuffle = True, 
    pin_memory = True, 
)  

'''



# Getting a batch from the dataloader 
x, y = next(iter(train_loader)) 
  
# Getting the boxes coordinates from the labels 
# and converting them into bounding boxes without scaling 
boxes = [] 
for i in range(y[0].shape[1]): 
    anchor = scaled_anchors[i] 
    boxes += convert_cells_to_bboxes( 
               y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
             )[0] 
  
# Applying non-maximum suppression 
boxes = nms(boxes, iou_threshold=1, threshold=0.7) 
  
# Plotting the image with the bounding boxes 
plot_image(x[0].permute(1,2,0).to("cpu"), boxes, coco)
'''

# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors, loss_record): 
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True) 
  
    # Initializing a list to store the losses 
    losses = [] 
  
    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device)
        y0, y1, y2 = ( 
            y[0].to(device), 
            y[1].to(device), 
            y[2].to(device), 
        ) 
  
        with torch.cuda.amp.autocast(): 
            # Getting the model predictions 
            outputs = model(x) 
            # Calculating the loss at each scale 
            loss = ( 
                  loss_fn(outputs[0], y0, scaled_anchors[0]) 
                + loss_fn(outputs[1], y1, scaled_anchors[1]) 
                + loss_fn(outputs[2], y2, scaled_anchors[2]) 
            ) 
  
        # Add the loss to the list 
        losses.append(loss.item()) 
  
        # Reset gradients 
        optimizer.zero_grad() 
  
        # Backpropagate the loss 
        scaler.scale(loss).backward() 
  
        # Optimization step 
        scaler.step(optimizer) 
  
        # Update the scaler for next iteration 
        scaler.update() 
  
        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)
    
    
    loss_record.append(sum(losses) / len(losses))



# Creating the model from YOLOv3 class 
model = myYOLOv3(num_classes=20).to(device) 
  
# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
  
# Defining the loss function 
loss_fn = myYOLOLoss() 
  
# Defining the scaler for mixed precision training 
scaler = torch.cuda.amp.GradScaler() 
  

  
# Scaling the anchors 
scaled_anchors = ( 
    torch.tensor(ANCHORS) * 
    torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(device) 

loss_record = []
# Training the model 
for e in range(1, epochs+1): 
    print("Epoch:", e) 
    training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, loss_record) 
  
    # Saving the model 
    if save_model: 
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

plot_x = np.arange(0, len(loss_record))
plot_y = np.array(loss_record)
plt.plot(plot_x, plot_y)
plt.xlabel("Epoch")
plt.ylabel("Mean Loss over training data")
plt.show()

####
##Test the model after a long training process
####

# Taking a sample image and testing the model 
  
# Setting the load_model to True 
load_model = True
  
# Defining the model, optimizer, loss function and scaler 
model = myYOLOv3(num_classes=20).to(device) 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
scaler = torch.cuda.amp.GradScaler() 
  
# Loading the checkpoint 
if load_model: 
    load_checkpoint(checkpoint_file, model, optimizer, learning_rate)
    # Also reading the learning rate in case the user wants to continue training
    # from that checkpoint (I am not using this feature, though)

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
    transform=test_transform 
) 
test_loader = torch.utils.data.DataLoader( 
    test_dataset, 
    batch_size = 1, 
    num_workers = 2, 
    shuffle = True, 
) 
  
# Getting a sample image from the test data loader 
x, y = next(iter(test_loader)) 
x = x.to(device) 
  
model.eval() 
with torch.no_grad(): 
    # Getting the model predictions 
    output = model(x) 
    # Getting the bounding boxes from the predictions 
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
  
# Plotting the image with bounding boxes for each image in the batch 
for i in range(batch_size): 
    threshold = 0.65
    filter_boxes = [box for box in bboxes[i] if box[1] > threshold] 
    boxes_old = torch.tensor(filter_boxes)[:, 2:]
    boxes = torch.ones(boxes_old.shape).to(device=device)
    boxes[:, 0] = boxes_old[:, 0] - boxes_old[:, 2]/2
    boxes[:, 1] = boxes_old[:, 1] - boxes_old[:, 3]/2
    boxes[:, 2] = boxes_old[:, 0] + boxes_old[:, 2]/2
    boxes[:, 3] = boxes_old[:, 1] + boxes_old[:, 3]/2

    scores = torch.tensor(filter_boxes)[:, 1].to(device)
    # Applying non-max suppression to remove overlapping bounding boxes 
    keep_idx = torchvision.ops.nms(boxes, scores=scores, iou_threshold=0.5) 
    nms_boxes = list(np.array(filter_boxes)[np.array(keep_idx.to('cpu'))])
    
    # Plotting the image with bounding boxes 
    plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
