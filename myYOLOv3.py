from torch import nn
import torch
from yolov3_helper import *

'''
My YOLOv3 implementation based on the original repo
(The original codes were written in C++)
'''

class convolutional(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride_size, padding_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride_size, padding=padding_size),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.layers(x)

class residual(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels//2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_channels//2, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return x + self.layers(x)


class myYOLOv3(nn.Module):
    '''
    Custom YOLOv3 pipeline
    '''

    def __init__(self, input_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        # Part I: Darknet 53 network (constructed from the paper & source codes)

        # Consist of three parts, because some outputs from the hidden layers
        # are also concatenated in the following layers after Darknet 53
        self.Darknet_1 = nn.Sequential(
            convolutional(input_channels, 32, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(32, 64, kernel_size=3, stride_size=2, padding_size=1),

            # residual x 1
            residual(64),

            convolutional(64, 128, kernel_size=3, stride_size=2, padding_size=1),

            # residual x 2
            residual(128),
            residual(128),


            convolutional(128, 256, kernel_size=3, stride_size=2, padding_size=1),

            # residual x 8
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256)
        )
        self.Darknet_2 = nn.Sequential(
            convolutional(256,512, kernel_size=3, stride_size=2, padding_size=1),

            # residual x 8
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512)
        )
        self.Darknet_3 = nn.Sequential(
            convolutional(512, 1024, kernel_size=3, stride_size=2, padding_size=1),

            # residual x 4
            residual(1024),
            residual(1024),
            residual(1024),
            residual(1024)
        )

        # Part II: extra parts (not mentioned explicitly in the original paper, but in the source codes)
        self.detector_1 = nn.Sequential(
            convolutional(1024, 512, kernel_size=1, stride_size=1, padding_size=0),
            convolutional(512, 1024, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(1024, 512, kernel_size=1, stride_size=1, padding_size=0),
            convolutional(512, 1024, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(1024, 512, kernel_size=1, stride_size=1, padding_size=0)
        )

        # First output at that scale
        self.scale_predictor_1 = nn.Sequential(
            convolutional(512, 1024, kernel_size=3, stride_size=1, padding_size=1),
            nn.Conv2d(1024, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)
        )   

        self.upsampler_1 = nn.Sequential(
            convolutional(512, 256, kernel_size=1, stride_size=1, padding_size=0),
            nn.Upsample(scale_factor=2), 
        )

        self.detector_2 = nn.Sequential(
            convolutional(768, 256, kernel_size=1, stride_size=1, padding_size=0),
            convolutional(256, 512, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(512, 256, kernel_size=1, stride_size=1, padding_size=0),
            convolutional(256, 512, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(512, 256, kernel_size=1, stride_size=1, padding_size=0)
        )

        # The second scale prediction
        self.scale_predictor_2 = nn.Sequential(
            convolutional(256, 512, kernel_size=3, stride_size=1, padding_size=1),
            nn.Conv2d(512, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)
        )

        self.upsampler_2 = nn.Sequential(
            convolutional(256, 128, kernel_size=1, stride_size=1, padding_size=0),
            nn.Upsample(scale_factor=2), 
        )

        self.detector_3 = nn.Sequential(
            convolutional(384, 128, kernel_size=1, stride_size=1, padding_size=0),
            convolutional(128, 256, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(256, 128, kernel_size=1, stride_size=1, padding_size=0),
            convolutional(128, 256, kernel_size=3, stride_size=1, padding_size=1),
            convolutional(256, 128, kernel_size=1, stride_size=1, padding_size=0)
        )

        # The third scale predictor
        self.scale_predictor_3 = nn.Sequential(
            convolutional(128, 256, kernel_size=3, stride_size=1, padding_size=1),
            nn.Conv2d(256, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        outputs = [] 
        route_connections = [] 

        # Section I: go through the network Darknet 53
        x = self.Darknet_1(x)
        route_connections.append(x)

        x = self.Darknet_2(x)
        route_connections.append(x)

        x = self.Darknet_3(x)

        # Section II: generate the scale predictions
        x = self.detector_1(x)
        output1 = self.scale_predictor_1(x)
        output1 = output1.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output1 = output1.permute(0, 1, 3, 4, 2) 
        outputs.append(output1)

        x = self.upsampler_1(x)
        x = torch.cat([x, route_connections[-1]], dim=1) 
        route_connections.pop() 


        x = self.detector_2(x)

        output2 = self.scale_predictor_2(x)
        output2 = output2.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output2 = output2.permute(0, 1, 3, 4, 2) 
        outputs.append(output2)

        x = self.upsampler_2(x)
        x = torch.cat([x, route_connections[-1]], dim=1) 
        route_connections.pop() 


        x = self.detector_3(x)

        output3 = self.scale_predictor_3(x)
        output3 = output3.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output3 = output3.permute(0, 1, 3, 4, 2) 
        outputs.append(output3)

        return outputs





# Defining YOLO loss class (copied from the blog codes)
class myYOLOLoss(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.mse = nn.MSELoss() 
        self.bce = nn.BCEWithLogitsLoss() 
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.sigmoid = nn.Sigmoid() 
      
    def forward(self, pred, target, anchors): 
        # Identifying which cells in target have objects  
        # and which have no objects 
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0
  
        # Calculating No object loss 
        no_object_loss = self.bce( 
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), 
        ) 
  
          
        # Reshaping anchors to match predictions 
        anchors = anchors.reshape(1, 3, 1, 1, 2) 
        # Box prediction confidence 
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), 
                               torch.exp(pred[..., 3:5]) * anchors 
                            ],dim=-1) 
        # Calculating intersection over union for prediction and target 
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach() 
        # Calculating Object loss 
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), 
                               ious * target[..., 0:1][obj]) 
  
          
        # Predicted box coordinates 
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) 
        # Target box coordinates 
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors) 
        # Calculating box coordinate loss 
        box_loss = self.mse(pred[..., 1:5][obj], 
                            target[..., 1:5][obj]) 
  
          
        # Claculating class loss 
        class_loss = self.cross_entropy((pred[..., 5:][obj]), 
                                   target[..., 5][obj].long()) 
  
        # Total loss 
        return ( 
            box_loss 
            + object_loss 
            + no_object_loss 
            + class_loss 
        )

    

