#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std=[0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
        transforms.RandomPerspective(),
        transforms.RandomResizedCrop(size=(80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])  
    
    if mode == 'train':
        return transform_train
    elif mode == 'test':
        return transform_val


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(in_channels)
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        bn = self.bn(x) 
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        
        conv4 = self.relu(self.conv4(c3_dense)) 
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
        
        return c5_dense
    
class Transition_Layer(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__() 
        self.relu = nn.ReLU(inplace = True) 
        self.bn = nn.BatchNorm2d(num_features = out_channels) 
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0) 
        self.dropout1=nn.Dropout(0.25)
        
        
    def forward(self, x): 
        bn = self.bn(self.relu(self.conv(x))) 
        out = self.avg_pool(bn) 
        out=self.dropout1(out)
        return out 

class Network(nn.Module): 
    def __init__(self): 
        super(Network, self).__init__() 

        self.lowconv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3, bias = False) 
        self.relu = nn.ReLU()

        # Make Dense Blocks 
        self.denseblock1 = self._make_dense_block(64) 
        self.denseblock2 = self._make_dense_block(128)
        self.denseblock3 = self._make_dense_block(128)
        # Make transition Layers 
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)
        # Classifier 
        self.bn = nn.BatchNorm2d(num_features = 64) 
        self.pre_classifier = nn.Linear(6400, 512) 
        self.classifier = nn.Linear(512, 8)

    def _make_dense_block(self, in_channels): 
        layers = [] 
        layers.append(Dense_Block(in_channels)) 
        return nn.Sequential(*layers) 
    
    def _make_transition_layer(self, layer, in_channels, out_channels): 
        modules = [] 
        modules.append(layer(in_channels, out_channels)) 
        return nn.Sequential(*modules) 

    def forward(self, x): 
        out = self.relu(self.lowconv(x)) 
        out = self.denseblock1(out) 
        out = self.transitionLayer1(out) 
        out = self.denseblock2(out) 
        out = self.transitionLayer2(out) 

        out = self.denseblock3(out) 
        out = self.transitionLayer3(out) 

        out = self.bn(out) 
        
        
        out = out.view(out.size(0), -1)
        
        #print(out.size(1))
        
        
        out = self.pre_classifier(out) 
        out = self.classifier(out)
        out = F.log_softmax(out,dim=1)
        return out


net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################

lr=0.001
optimizer = torch.optim.Adam(net.parameters(),lr=lr,
                                 weight_decay=0.00001)

loss_func = nn.NLLLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 128
epochs = 5000
