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

a. choice of architecture, algorithms and enhancements (if any)
    The basic architecture is convolutional neural network with no doubt because CNN is designed to analyze visual imagery and perform
the best in image classification. From that, we have tried a few CNN based architecture like traditional CNN, ResNet and denseNet.
Among these, denseNet performs the most stable and accurate under the same condition. It has some draw backs like slow training and require 
more epoch to see result, but it is not of our concern. 
    Therefore our final choice is denseNet with 3 dense block(each contain 4 densely connected convolution layer), 3 transition layer and 2 fully connected layer.

b.choice of loss function and optimizer: 
    For loss function, we have the choice between Cross-Entropy Loss and Negative Log-Likelihood Loss. They are useful in classification problems. 
We suppose that the difference between them is NLLLoss requires Log SoftMax layer as the last layer,
however, the Pytorch Cross-Entropy loss function has these computed inside. Finally, we decided to use NLLLoss.
    For optimizer, Adam is used instead of SGD since Adam is less sensitive to learning rate. We tried lr_scheduler (CosineAnnealingLR), 
while it will make the accuracy fluctuate a lot (5%). Then we have it removed.

c.choice of image transformations: 
     The data we have is relatively small, therefore some data augmentations are applied. we decided to keep the 80*80 size and make other changes like
crop, rotation, and color adjustment. They have efficiently reduced overfitting and increase accuracy. Some of them is commonly used on picture classification.
The ColorJitter function improves approximately 2% accuracy (on our DenseNet).

d.tuning of metaparameters
    We changed the number of epochs to be 5000 cause our model require around 500 epochs to see the results. Finish all these 5000 epochs take a 
lot time so we didn't expect to finish all, instead we add feature that will saves the model to a folder and the output accuracy to a log file. 
In this way, we can take a generally long training. 
    We start at a small batch_size and keep increasing it, and we found that small batch size speed up the training but result in a unstable model.
And large batch size slow the training but converge to a more stable model. Therefore we choose batch_size = 128 as a compromise.

e. use of validation set, and any other steps taken to improve generalization and avoid overfitting
    Though it is suggest that we use the variable train_val_split to make design decisions aimed to avoid overfitting to the training data,
we didn't really change this variable at the early satge. In early stage, we mainly focus on decide the architecture of our model, change 
train_val_split won't help with that and we agreed on 0.8 as a reasonable value. Later on, after we decided to use denseNet and some transformation,
we slightly raise this value to prevent overfitting. 
    Since 20 layer of denseNet is powerful enough and will quickly become overfitting in our previous trails, we applyed sevral method to avoid overfitting.
Add dropout layer is one method that helps a lot, we start by adding a couple around fully connected layers, and add dropout to every transition layer after
find it helps with resolve overfitting. Apply more transformation also helps and has been disscussed in previous answer. Reduce the complixity of the model 
also helps to avoid overfitting. The number of layers and independent paramters is an important variable that we tuning a lot, during our pratice, 
we have used it on all the different architectures to find a suitable capacity with not result in overfitting.




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
        transforms.RandomResizedCrop(80),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
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

# the model below is reference to this website 
# https://towardsdatascience.com/simple-implementation-of-densely-connected-convolutional-networks-in-pytorch-3846978f2f36

class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(in_channels)
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

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
    
        return c4_dense
    
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
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 128, out_channels = 128) 
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 128, out_channels = 128) 
        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 128, out_channels = 64)
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
train_val_split = 0.95
batch_size = 128
epochs = 5000
