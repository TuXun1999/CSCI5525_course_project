import torch
from PIL import Image
import numpy as np
from yolov3_helper import *
# Create a dataset class to load the images and labels from the folder 
class COCODataset(torch.utils.data.Dataset): 
    def __init__( 
        self, coco_annotations_file, coco_images_dir, coco_parser, anchors,
        image_size=416, grid_sizes=[13, 26, 52], transform=None
    ): 
        # Store the coco parser, where image data are stored
        self.coco_annotations_file = coco_annotations_file
        self.coco_images_dir = coco_images_dir
        self.coco = coco_parser

        # Image size 
        self.image_size = image_size 
        # Transformations 
        self.transform = transform 
        # Grid sizes for each scale 
        self.grid_sizes = grid_sizes 
        # Anchor boxes 
        self.anchors = torch.tensor( 
            anchors[0] + anchors[1] + anchors[2]) 
        # Number of anchor boxes 
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale 
        self.num_anchors_per_scale = self.num_anchors // 3

        # Ignore IoU threshold 
        self.ignore_iou_thresh = 0.5

    def __len__(self): 
        return len(self.coco.get_imgIds())

    def __getitem__(self, idx): 
        image_ids = self.coco.get_imgIds()
        idx = image_ids[idx]
        # Getting the image
        image = np.array(Image.open(f"{self.coco_images_dir}/{str(idx).zfill(12)}.jpg"))
        
        # Getting the bboxes of the associated image
        ann_ids = self.coco.get_annIds(idx)
        annotations = self.coco.load_anns(ann_ids)
        bboxes = []
        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = [int(b)for b in bbox]
            x = x/image.shape[1] 
            y = y/image.shape[0] 
            w = w/image.shape[1]
            h = h/image.shape[0]

            if (x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1):
                continue
            class_id = ann["category_id"]
            bboxes.append([x, y, w, h, class_id])
            
        # Albumentations augmentations 
        if self.transform: 
            augs = self.transform(image=image, bboxes=bboxes) 
            image = augs["image"] 
            bboxes = augs["bboxes"] 

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
        # target : [probabilities, x, y, width, height, class_label] 
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
                for s in self.grid_sizes] 

        # Identify anchor box and cell for each bounding box 
        for box in bboxes: 
            # Calculate iou of bounding box with anchor boxes 
            iou_anchors = iou(torch.tensor(box[2:4]), 
                            self.anchors, 
                            is_pred=False) 
            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
            x, y, width, height, class_label = box 

            # At each scale, assigning the bounding box to the 
            # best matching anchor box 
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                
                # Identifying the grid size for the scale 
                s = self.grid_sizes[scale_idx] 
                
                # Identifying the cell to which the bounding box belongs 
                i, j = int(s * y), int(s * x) 
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                
                # Check if the anchor box is already assigned 
                if not anchor_taken and not has_anchor[scale_idx]: 

                    # Set the probability to 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative 
                    # to the cell 
                    x_cell, y_cell = s * x - j, s * y - i 

                    # Calculating the width and height of the bounding box 
                    # relative to the cell 
                    width_cell, height_cell = (width * s, height * s) 

                    # Identify the box coordinates 
                    box_coordinates = torch.tensor( 
                                        [x_cell, y_cell, width_cell, 
                                        height_cell] 
                                    ) 

                    # Assigning the box coordinates to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 

                    # Assigning the class label to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

                    # Set the anchor box as assigned for the scale 
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the 
                # IoU is greater than the threshold 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
                    # Set the probability to -1 to ignore the anchor box 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target 
        return image, tuple(targets)
