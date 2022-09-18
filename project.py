# **Introduction**

# In this project, you will implement a neural network to classify Japanese calligraphy characters, and use it to perform as triggers to move around the museum with RRT path planning.
# You will use the generated paths on a differential drive robot to move around. 

# Part 1: Object Detection with Deep Learning

import os
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

!pip install torchsummaryX --quiet
from torchsummaryX import summary as summaryX
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## k-MNIST Dataset

# We will teach our classifier to recognize Japanese characters! The [k-MNIST](https://github.com/rois-codh/kmnist) contains different Japanese hiragana characters written in different forms, and it is ready-to-load in PyTorch with the following code.


# download dataset
train_dataset = datasets.KMNIST(root='dataset/', train=True, transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)
test_dataset = datasets.KMNIST(root='dataset/', train=False, transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)

# You can explore the images in the dataset by running the cell below. Familiarize yourself with the data structure of the dataset.  
# The dataset (both train and test) are a list of tuples, with each tuple containing the image in pytorch tensor and the label. 


# Visualizing one of the characters in the dataset
img_np = test_dataset[0][0] # An image in the dataset
print(test_dataset[0][1]) # Corresponding label of the image
plt.imshow(img_np[0], cmap='Greys_r')
print(img_np.shape)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        '''
        Init function to define the layersl
        '''
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 5, 
                              kernel_size = 5, stride = 3, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 5, out_channels = 48, 
                              kernel_size = 5, stride = 1, padding = 0)
        self.linear1 = nn.Linear(48, 10)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        ''' Performs forward pass with the network.

        Args:
            x: input image of shape (N,C,H,W)
               - N: size of the batch, e.g. number of images
               - C: number of channels, 1 for grayscale images
               - H: height of image
               - W: width of image
        Returns:
            model_output: the output (raw scores) of shape (N,10)

        Note:
            Gradients will be handled automatically with PyTorch.
        '''
        # construct forward pass
        # x: input matrix
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        return x

# construct the model(e.g. neural network)
model = SimpleNet().to(device)
model
# model summary
summary(model, (1,32,32))

# Hyperparameters
learning_rate = 0.01
num_epochs = 5
batch_size = 64

# set dataset loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}

# construct model
model = SimpleNet().to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Below are code to train and test the network on the provided dataset. We will use hyperparameters, model, loss criterion and optimizer defined in the code above. Do not change any parameters above. """

from IPython.display import HTML, display
class ProgressMonitor(object):   
    tmpl = """
        <p>Loss: {loss:0.4f}   {value} / {length}</p>
        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>
    """
 
    def __init__(self, length):
        self.length = length
        self.count = 0
        self.display = display(self.html(0, 0), display_id=True)
        
    def html(self, count, loss):
        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))
        
    def update(self, count, loss):
        self.count += count
        self.display.update(self.html(self.count, loss))

def train_iter(model, train_dataloader):
    # One iteration of training through the entire dataset.

    running_loss = 0.0

    # set model to training mode
    model.train()

    # create a progress bar
    progress = ProgressMonitor(length=dataset_sizes["train"])

    for data in train_dataloader:
        inputs, labels  = data
        batch_size = inputs.shape[0]
      
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))

        # zero out previous gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
                    
        running_loss += loss.data * batch_size
        # update progress bar
        progress.update(batch_size, running_loss)
    
    return running_loss


def test_iter(model, test_dataloader):
    # Test the current model on the test dataset.

    # set model to evaluation mode
    model.eval()
    # Do not need gradients for testing
    test_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels  = data
            batch_size = inputs.shape[0]

            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            outputs = model(inputs)

            # calculate the loss
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
    
            # update running loss value
            test_loss += loss.data * batch_size
    return test_loss

def train_and_test(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, first_epoch=1):
    # Train and test the neural network on the specified dataset for several iterations.

    best_epoch = -1
    last_train_loss = -1
    plot_train_loss = []
    plot_test_loss = []
  
    for epoch in range(first_epoch, first_epoch + num_epochs):
        print('\nEpoch', epoch)
        
        # training
        running_loss = train_iter(model, dataloaders[0])
  
        epoch_loss = running_loss / dataset_sizes["train"]
        print('Training loss:', epoch_loss.item())
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        plot_train_loss.append(epoch_loss.cpu())
  
        # testing
        test_loss = test_iter(model, dataloaders[-1])
  
        epoch_test_loss = test_loss / dataset_sizes["test"]
        print('Testing loss:', epoch_test_loss.item())
        plot_test_loss.append(epoch_test_loss.cpu())
        writer.add_scalar('Testing Loss', epoch_test_loss, epoch)

    return plot_train_loss, plot_test_loss, model

# The below code trains SimpleNet on the train set and performs testing on the test dataset.  
# Both stages return a list of loss values at each epoch.  
# Training will take under 2 minutes using the Colab GPU runtime. 


train_losses, test_losses, model = train_and_test(model=model, criterion=criterion, optimizer=optimizer,
                                              num_epochs=num_epochs, dataloaders=[train_loader, test_loader], 
                                              dataset_sizes=dataset_sizes)

## Visualizing Losses


def plot_losses(train_losses, test_losses):

    # Plots train and test loss curves.
    # Note: use matplotlib or any library to plot the graph
    #       Include title, x/y labels, and legend in the plot
    # Args:
    #     train_losses: list of train losses by epoch
    #     test_losses: list of test losses by epoch

    plt.title('Training & Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(test_losses, label='test losses')
    plt.plot(train_losses, label='train losses')
    plt.legend(loc='upper right')
    plt.show()
    # raise NotImplementedError('plot_losses not implemented')


# plot loss graph for SimpleNet
plot_losses(train_losses, test_losses)

# This code prints the prediction accuracy of the model on the datasets. You will get testing accuracy of around 87% on SimpleNet. This is not enough for a reliable classification. The LeNet you will implement will improve in accuracy. """

def accuracy(loader, model, train=True):

    # Calculates prediction accuracy of the model on data
    
    # Args:
    #     loader: (Dataloader) contains data and labels
    #     model: neural network model
    #     train (boolean): True if training, False if testing

    num_correct = num_samples = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels  = data
            batch_size = inputs.shape[0]
            
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            outputs = model(inputs)
            # prediction with the top score
            _, preds = outputs.max(1)
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
    accuracy = (num_correct.item()/num_samples)*100
    
    if train:
        print("Model Predicted {} correctly out of {} from training dataset, Accuracy : {:.2f}".format(num_correct.item(), num_samples, accuracy))
    else:
        print("Model Predicted {} correctly out of {} from testing dataset, Accuracy : {:.2f}".format(num_correct.item(), num_samples, accuracy))



# Calculate accuracy
accuracy(train_loader, model)
accuracy(test_loader, model, train=False)


## Task 1: Implementing LeNet


### LeNet Architecture
# S.No | Layers | Output Shape (Height, Width, Channels)
# --- | --- | ---
# 1 | Input Layer | 32 x 32 x 1
# 2 | Conv2d [6 Filters of size = 5x5, stride = 1, padding = 0 ] | 28 x 28 x 6
# 3 | Average Pooling [stride = 2, padding = 0] | 14 x 14 x 6
# 4 | Conv2d [16 Filters of size = 5x5, stride = 1, padding = 0 ] | 10 x 10 x 16
# 5 | Average Pooling [stride = 2, padding = 0] | 5 x 5 x 16
# 6 | Conv2d [120 Filters of size = 5x5, stride = 1, padding = 0 ] | 1 x 1 x 120
# 7 | Flatten vector| 120 
# 8 | Linear2 Layer | 84 
# 9 | Final Linear Layer | 10

# Referring to the network visualization and table above, fill in the `__init__` and `forward` in `class LeNet`. Since PyTorch takes care of the backward pass part, you will only have to code the forward pass in `forward`. You can take a look at the official documentation for the modules you will use:
# 1. [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
# 2. [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
# 3. [nn.AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
# 4. [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)


class LeNet(nn.Module):
    def __init__(self):

        # Init function to define the layers

        super(LeNet, self).__init__()
        # TODO 2.1
        ###########################################################################
        # Student code begin
        ###########################################################################
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, 
                           kernel_size = 5, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, 
                              kernel_size = 5, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, 
                              kernel_size = 5, stride = 1, padding = 0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        ###########################################################################
        # Student code end
        ###########################################################################

    def forward(self, x):
        '''
        Performs forward pass with the network

        Args:
            x: input image of shape (N,C,H,W)
        Returns:
            model_output: the output (raw scores) of shape (N,10)
        '''
        model_output = None
        # TODO 2.2
        ###########################################################################
        # Student code begin
        ###########################################################################
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        model_output = self.linear2(x)
        ###########################################################################
        # Student code end
        ###########################################################################
        return model_output

model = LeNet().to(device)
model

# Run the unit test below to check if your network implementation is correct."""

import unittest

class TestLeNet(unittest.TestCase):
  def test_layers(self):
    convcnt, lincnt, relucnt, poolcnt = 0,0,0,0
    for name, layer in model.named_modules():
      if isinstance(layer, nn.Conv2d):
        convcnt += 1
      if isinstance(layer, nn.Linear):
        lincnt += 1
      if isinstance(layer, nn.ReLU):
        relucnt += 1
      if isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
        poolcnt += 1
    assert convcnt >= 3
    assert lincnt >= 2
    assert relucnt >= 1
    assert poolcnt >= 1

  def test_output(self):
    inputs = torch.randn(5,1,32,32).to(device)
    outputs = model(inputs)
    assert (inputs.shape[0], 10) == outputs.shape

suite = unittest.TestSuite()
suite.addTest(TestLeNet('test_layers'))
suite.addTest(TestLeNet('test_output'))
unittest.TextTestRunner().run(suite)

# Now that you have the LeNet model ready, you can use the first code block in Task2 for training and testing.

## Task 2: Tuning the network

# In this section you will tune the hyperparameters to improve network performance.  
# Initially, train the network with the default parameters provided below.  
# After that, play with the hyperparameters (check below in the Reflection Question section for possible values) to improve the accuracy of the network. Training this network should usually take under 5 minutes on Colab with the GPU runtime.

# You will have to achieve over 93% test accuracy in order to get full points.

### Train and Test


# Hyperparameters
# Only edit the values! The autograder will not work if you don't.  
learning_rate = 0.005
num_epochs = 5
batch_size = 128
#######################

# set dataset loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}

# construct model
model = LeNet().to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# perform training and testing
train_losses, test_losses, model = train_and_test(model=model, criterion=criterion, optimizer=optimizer,
                                              num_epochs=num_epochs, dataloaders=[train_loader,test_loader], 
                                              dataset_sizes=dataset_sizes)

### Save and Load Model

# Use the code below to save the model weights or load the pre-saved model that contains the trained model weights.  
# You can use this to avoid training the network every time you re-load the Colab environment. 
# Make sure you download the `checkpoint.pt` file locally as it will be deleted when the environment is reloaded. 


# define the path to save/load the model
model_dir = 'checkpoints/'

def save_model(model_dir):
    '''
    Saves the model state and optimizer state on the dict
    Change model_dir to the to save different model weights
    ex) 'SimpleNet_checkpoints', 'LeNet_checkpoints'
    '''
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(model_dir, 'checkpoint.pt'))
    print('model saved at', model_dir + 'checkpoint.pt')

def load_model(model_dir, load_optimizer=False):
    '''
    Load the model from the disk if it exists, skip if you don't need this part
    '''
    if os.path.exists(model_dir):
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loaded model from saved checkpoint')

# Uncomment as needed
save_model(model_dir)
# load_model(model_dir)

# Plot the loss curves using the `plot_losses` function you implemented earlier.  
# You will paste the before/after tuning loss plots in the reflection powerpoint.
# 

# Plot your train and test losses
plot_losses(train_losses, test_losses)

# Print train/test accuracy using the accuracy function above. 


accuracy(train_loader, model)
accuracy(test_loader, model, train=False)

### Inference on Custom Image

# In this section you will see if your trained network actually works in classifying hiragana characters. 
# The `inference` function takes in a 32x32 image and returns the predicted character.  
# Below is a classmap of the 10 labels for better understanding.  
# For your curiosity, you can find the pronunciation of each of the characters and more [here](https://www.thoughtco.com/how-to-pronounce-hiragana-japanese-hiragana-with-audio-files-4077351)

# index | unicode | char | pronunciation
# :-: | :-: | :-: | :-:
# 0 | U+304A | お | a
# 1 | U+304D | き | ki
# 2 | U+3059 | す | su
# 3 | U+3064 | つ | tsu
# 4 | U+306A | な | na
# 5 | U+306F | は | ha
# 6 | U+307E | ま | ma
# 7 | U+3084 | や | ya
# 8 | U+308C | れ | re
# 9 | U+3092 | を | o

# The `imageClassifier` class is used to read the image containing a hiragana character and provide the predicted label of the hiragana character.  
# You will implement the rest of `classify_image()` function.

class ImageClassifier:
    """
    Classify the image and provide the label of the character.
    """
    def __init__(self, model):
        self.model = model
    
        # index and corresponding character unicode
        self.labels_dict = dict([(0, u"\u304A"), (1, u"\u304D"), (2, u"\u3059"), 
                                 (3, u"\u3064"), (4, u"\u306A"), (5, u"\u306F"), 
                                 (6, u"\u307E"), (7, u"\u3084"), (8, u"\u308C"), 
                                 (9, u"\u3093")])

    def classify_image(self, image):
        """ Classify the image of calligraphy charater.

        Args:
            image (torch.Tensor) image tensor of size (1,32,32)

        Returns:
            int: classified label of the image
        
        Note:
            You can refer to accuracy() function to see how to forward pass
            the image input through the model and get the output
        """
        image_tensor = image.clone()
        image_tensor.unsqueeze_(0)
        ###############################################################################

        # hint: you will need to change the device attached to the data from 
        # the gpu(cuda) to cpu. Using the command {data}.cpu() will do this.
        input = Variable(image_tensor).to(device)
        output = self.model(input)
        index = output.cpu().data.numpy().argmax()
        # raise NotImplementedError('classify_image not implemented')

        ###############################################################################


        return index

    def character(self, image):
        return self.labels_dict[self.classify_image(image)]

image_classifier = ImageClassifier(model)

# test the classifier on an image
img_index = np.random.randint(10000)
image = test_dataset[img_index][0]
label = test_dataset[img_index][1]
plt.imshow(image[0], cmap='Greys_r')
print('Prediction: ', image_classifier.character(image))
print('Target: ', image_classifier.labels_dict[label])

#Part 2: RRT with Object Detection

# In this section you will implement Rapidly-exploring Random Tree (RRT) to make the robot navigate in the map. A visualization of a museum with obstacles is provided to help you better understand the tree formation and path planning process.

## Setup

# importing necessary modules for factor graph optimization and visualization
!pip install -q plotly gtsam
import gtsam
import numpy as np
import plotly.graph_objects as go
from tqdm import trange
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import math

########################## load utils ##########################
!pip install --upgrade --no-cache-dir gdown &> /dev/null
!gdown 1tSpeMiBsXemhSWv0xMMxLP9F0O2ChZQr
from utils import *

########################## load dataset ##########################
test_dataset = datasets.KMNIST(root='dataset/', train=False, transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)
image_classifier = ImageClassifier(model)

########################## environment helpers ##########################

def rect(x_m, y_m, xsize_m, ysize_m, rgba=None):
    return dict(
        type='rect',
        x0=x_m, y0=y_m,
        x1=x_m + xsize_m, y1=y_m + ysize_m,
        fillcolor=f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})" if rgba else "rgb(255, 255, 255)",
        line_width=0,
        layer="below"
    )


class Image:

    # Class for images in the KMNIST dataset

    def __init__(self, location, facing, image_label):
 
        # location: [x,y] coordinate of the image location
        # facing: (w/e/s/n) orientation of the image
        # image_label: (0~9) label of the image

        self.facing = facing
        if facing == 'w':
            theta = np.pi
        elif facing == 'e':
            theta = 0
        elif facing == 's':
            theta = -np.pi/2
        elif facing == 'n':
            theta = np.pi/2
        self.wTi = gtsam.Pose2(location[0], location[1], theta)
        self.image_label = image_label
        self.location = location

    def sample(self):

        # Samples a character corresponding to the image label from the dataset.
        # Return: image tensor

        test_size = len(test_dataset)
        index = np.random.randint(0, test_size)
        while not test_dataset[index][1] == self.image_label:
            index = (index + 1) % test_size
        return test_dataset[index][0]

    def get_rect(self, color):
        if self.facing == 'w':
            xm = self.wTi.x() - 0.25
            ym = self.wTi.y() - 0.5
            xsize = 0.25
            ysize = 1
        elif self.facing == 'e':
            xm = self.wTi.x()
            ym = self.wTi.y() - 0.5
            xsize = 0.25
            ysize = 1
        elif self.facing == 's':
            xm = self.wTi.x() - 0.5
            ym = self.wTi.y() - 0.25
            xsize = 1
            ysize = 0.25
        elif self.facing == 'n':
            xm = self.wTi.x() - 0.5
            ym = self.wTi.y()
            xsize = 1
            ysize = 0.25

        return rect(xm, ym, xsize, ysize, color)


class Wall:

    # Class for walls in the map.

    def __init__(self, start_point, end_point, color):

        # start_point: [x,y] coordinate of the lower left corner of the rectangular wall
        # goal_point: [x,y] coordinate of the top right corner of the wall

        self.ps = start_point
        self.pe = end_point
        self.color = color

    def get_rect(self):
        return rect(self.ps[0], self.ps[1], self.pe[0]-self.ps[0], self.pe[1]-self.ps[1], self.color)

def env_walls():

    # Returns a list of obstacles in the map. 
    # Each Wall class represents a rectangular obstacle in the map. 

    white = [255, 255, 255, 255]
    gray = [150, 150, 150, 255]
    wall_list = [Wall([0,-0.5], [16, 0], white),
                 Wall([0, 9], [16, 9.5], white),
                 Wall([-0.5, -0.5], [0, 9.5], white),
                 Wall([16, -0.5], [16.5, 9.5], white),
                 Wall([0, 4.25], [3, 4.75], gray),
                 Wall([5, 4.25], [11, 4.75], gray),
                 Wall([13, 4.25], [16, 4.75], gray),
                 Wall([7.75, 1.75], [8.25, 7.25], gray),
                 Wall([0, 0], [3, 1.75], gray),
                 Wall([13, 0], [16, 1.75], gray),
                 Wall([0, 7.25], [3, 9], gray),
                 Wall([13, 7.25], [16, 9], gray),
                 Wall([5, 7.25], [11, 7.75], gray),
                 Wall([5, 1.25], [11, 1.75], gray)]
    return wall_list

def env_images():

    # Assigns images to specified position and orientation in the map.
    # Returns a list of images

    images_list = [Image([0, 3], 'e', 0),
                   Image([7.75, 3], 'w', 1),
                   Image([0, 6], 'e', 2),
                   Image([7.75, 6], 'w', 3),
                   Image([4, 9], 's', 4),
                   Image([12, 9], 's', 5),
                   Image([8.25, 6], 'e', 6),
                   Image([16, 6], 'w', 7),
                   Image([8.25, 3], 'e', 8),
                   Image([16, 3], 'w', 9)]
    return images_list

## Environment

# Later, you will use the image classifier you trained to help the robot navigate in a museum. Run the following code block to see a map of the museum. There are calligraphy images hanging on the wall shown in different colors.

# Each of the ten images contain different kinds of japanese characters. 

# Notice that the same calligraphy image may appear different over time, but it should always be the same character.

class Environment:
#    Simulation environment class.
#     Initializes the room size, images, and obstacles. 

    def __init__(self):
        self.room_size = (16, 9)
        self.image_list = env_images()
        self.wall_list = env_walls()

env = Environment()

# visualize the environment
visualize(env)

## RRT Components

# Here you  implement various components needed to run RRT. Some functions are already filled out for you.  


class Node:
    def __init__(self, point, parent=None):
        super(Node, self).__init__()
        # RRT Node Class

        # Parameters
        # ----------
        # point: numpy array [x,y]
        # parent: parent Node class

        # Notes
        # -----
        # You must use this node representation to build the tree. 
 
        self.point = point
        self.parent = parent

    @property
    def x(self):
       return self.point[0]

    @property
    def y(self):
       return self.point[1]


def dist(node1, node2):
    # Returns the distance between two nodes
    # Parameters
    # ----------
    # node1, node2: two Node objects

    # Returns
    # -------
    # Euclidean distance between two nodes

    return np.linalg.norm(node2.point - node1.point) 


def inObstacle(node, env):
    # Returns True if the given node is within ANY obstacle
    # Parameters
    # ----------
    # node: Node object
    # env: Environment class
    
    # Returns
    # -------
    # (boolean) True if the node is within any obstacle

    # Notes
    # -----
    # Use env.wall_list to access the list of obstacles. You may loop through the
    # obstacles to check if the given node is within the boundary of any obstacle.

    ###############################################################################

    for i in env.wall_list:
       if i.ps[0] <= node.point[0] <= i.pe[0] and i.ps[1] <= node.point[1] <= i.pe[1]:
         return True 
    return False

    ###############################################################################




def randomSample(goal_pos, env):
    # Generates random point in the map. 
    # Generated point should not be inside an obstacle.
    # You should return the goal position with some probability.

    # Parameters
    # ----------
    # goal_pos: numpy array [x,y] of goal position
    # env: Environment class

    # Returns
    # -------
    # Node object containing randomly generated coordinates


    ###############################################################################

    if 0.1 > np.random.rand():
        return Node(np.array(goal_pos))
    else:
        while True:
            random_point = Node(gtsam.Point2(np.random.uniform(16), 
                                            np.random.uniform(9)))
            # if inObstacle(random_point, env):
            #    random_point = Node(gtsam.Point2(np.random.uniform(16), 
                                            # np.random.uniform(9)))
            if not inObstacle(random_point, env):
               break
        return random_point
    ###############################################################################

  

def getNearest(rrt_tree, random_point):
    # Find the nearest node in the tree to the given point
    
    # Parameters
    # ----------
    # rrt_tree: list of nodes in the RRT tree
    # random_point (Node): randomly sampled node

    # Returns
    # -------
    # nearest_node (Node): Existing node in the tree nearest to random_point
    

    ###############################################################################
    currDist = float('inf')
    currNode = rrt_tree[0]
    for i in rrt_tree:
        distance = dist(i, random_point)
        if distance < currDist:
            currNode = i
            currDist = distance
    nearest_node = currNode
    ###############################################################################

    return nearest_node

def stepTo(nearest_point, random_point, step=1.):
    # Returns a node in the direction from nearest_point to random_point with a length 
    # of 'step' from the nearest point. 
    # Return random_point if the distance between two points is less than 'step'.

    # Parameters
    # ----------
    # nearest_point (Node): nearest point inside tree to the random point
    # random_point (Node): randomly sampled point
    # step (float): maximum step size to the random point

    # Return
    # ------
    # return: Node object with coordinate along the line segment
    

    ###############################################################################

    if dist(nearest_point, random_point) < step:
        return random_point
    else:
        theta = np.arctan2(random_point.y - nearest_point.y, random_point.x - nearest_point.x)
        return Node(np.array([nearest_point.x + step * np.cos(theta), nearest_point.y + step * np.sin(theta)]))
        # n = (random_point.point - nearest_point.point) / dist(nearest_point, random_point)
        # return Node(nearest_point.point + step * n)
    ###############################################################################


def isCollision(line, env):
    # Check if the given vertex collides with any obstacles

    # Parameters
    # ----------
    # line: Tuple of two nodes
    # env: Environment class

    # Return
    # ------
    # (boolean) True if the line passes through any of the obstacles.

    # Note
    # -----
    # 1. Use `line_intersection` to check for line collisions.
    # You can find the docstring for line_intersection by typing 
    # help(line_intersection) in a code cell.
    # 2. Use env.wall_list to call in a list of walls.
    
    ###############################################################################

    # start, end = line
    point = (line[0].point, line[1].point)
    for wall in env.wall_list:
        sx, sy = wall.ps
        ex, ey = wall.pe
        obs_lines = [[wall.ps, [sx,ey]], [wall.ps, [ex,sy]], [[sx,ey], wall.pe], [[ex,sy], wall.pe]]
        for obs_line in obs_lines:
            if line_intersection(*point, obs_line[0], obs_line[1]):
               return True
    return False
    ###############################################################################

  

def withinTolerance(point, goal, tolerance=1.5):
    # Check if a node is within tolerance of the goal.
    # Parameters
    # ----------
    # point (Node)
    # goal (Node): goal node
    # tolerance (float): distance from goal point that is considered close enough

    # Return
    # ------
    # (boolean) True if point is within tolerance
    
    ###############################################################################

    if (dist(point, goal) <= tolerance): 
      return True
    else: 
      return False
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################


import unittest

class TestRRT(unittest.TestCase):
    def test_inObstacle(self):
        assert inObstacle(Node(np.array([4,3])), env) == False
        assert inObstacle(Node(np.array([14,1])), env) == True
        assert inObstacle(Node(np.array([8,3])), env) == True

    def test_randomSample(self):
        for _ in range(5):
            sample = randomSample(np.array([4,1]), env)
            assert 0 <= sample.x <= 16
            assert 0 <= sample.y <= 9
            assert inObstacle(sample, env) == False

    def test_getNearest(self):
        tree = [Node(np.array([4,3])), Node(np.array([6,2]))]
        nearest_node = getNearest(tree, Node(np.array([4,5])))
        assert (nearest_node.point == np.array([4,3])).all()

    def test_stepTo(self):
        node1 = Node(np.array([4,1]))
        node2 = Node(np.array([4,7]))
        node3 = Node(np.array([4,1.5]))

        assert dist(node1, stepTo(node1, node2)) == 1
        assert dist(node1, stepTo(node1, node3)) < 1


    def test_isCollision(self):
        start = Node(np.array([7,3]))
        end = Node(np.array([9,5]))
        assert isCollision((start,end), env)

        start = Node(np.array([14,3]))
        end = Node(np.array([15,4]))
        assert isCollision((start,end), env) == False

    def test_tolerance(self):
        assert withinTolerance(Node(np.array([0,0])), Node(np.array([0,2])), 3)
        assert withinTolerance(Node(np.array([0,0])), Node(np.array([0,2]))) == False

suite = unittest.TestSuite()
suite.addTest(TestRRT('test_inObstacle'))
suite.addTest(TestRRT('test_randomSample'))
suite.addTest(TestRRT('test_getNearest'))
suite.addTest(TestRRT('test_stepTo'))
suite.addTest(TestRRT('test_isCollision'))
suite.addTest(TestRRT('test_tolerance'))
unittest.TextTestRunner().run(suite)

## RRT

# Using the components you wrote above, complete the RRT algorithm.  
# RRT should  
# * sample a random point
# * find the nearest point in the tree
# * step towards the sampled point
# * add the point to the tree

def RRT(start_pos, goal_pos, env):
    # Implement RRT to find a path to the goal position. 
    # Parameters
    # ----------
    # start_pos (np.array): [x,y] start position
    # goal_pos (np.array): [x,y] goal coordinate
    # env: Environment class

    # Return
    # ------
    # list of nodes representing the path from source to goal node

    # Note
    # ----
    # To find the path in the tree, you will need to backtrack from the node closest
    # to the goal node to the root node. 
    
    root = Node(start_pos)
    goal_node = Node(goal_pos)
    rrt_tree = [root]
    soln_node = None
    ###############################################################################

    i = 0
    while soln_node is None:
        random = randomSample(goal_pos, env)
        nearest = getNearest(rrt_tree, random)
        step = stepTo(nearest, random)
        if not inObstacle(step, env) and not isCollision((nearest, step), env):
            step.parent = nearest
            rrt_tree.append(step)
            if withinTolerance(step, goal_node):
                soln_node = step
            i += 1
    
    path = []
    curr = soln_node
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    path.reverse()

    ###############################################################################

    return path, rrt_tree

##Visualize RRT and path

# Check if your RRT is running right by visualizing the RRT tree in the map.
# You can change the start and goal positions to test the tree generation in different locations. It may take a bit long to visualize a tree for a goal position far from the starting position. 


start_pos = np.array([4,1])
goal_pos = np.array([0,6])
path, rrt_tree = RRT(start_pos, goal_pos, env)

visualize_tree(env, rrt_tree)

# Now visualize the path to the goal you found from the RRT. 
# The code below visualizes and animates the found path frame by frame. 


visualize_path(env,path)

##Path smoothing (Extra Credit - 2 points)
# You will notice from the visualization above that the generated path is not always the shortest path to the goal, and the paths are in a jagged pattern.
# Here you will apply smoothing to connect non-consecutive vertices and form a new edge that does not collide with the obstacles. 


def smoothPath(path, env, iters=10):
    '''Returns a smoothed version of the given path.
    Parameters
    ----------
    path: list of nodes
    env: Environment class
    iters (int): number of iterations

    Return
    ------
    path: list of nodes 

    Note
    ----
    Remember to check if the new path collides with any obstacles
    '''
    # TODO 11
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################

    for i in range(iters):
        i = np.random.randint(len(path))
        j = np.random.randint(len(path))
        if not isCollision((path[i], path[j]), env):
          path = path[0:min(i,j) + 1] + path[max(i, j):]
    return path
    # raise NotImplementedError('smoothPath not implemented')

    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    
    # return path

# Visualize the new path after smoothing is applied. Compare to the generated path before smoothing. What changes to you observe?"""

visualize_path(env,smoothPath(path, env))


#Part 3: Differential Drive

# The robot used to move around is a differential drive robot; the robot has two wheels that can move at different speeds. Here you will implement some functions that generate speed commands for each wheel so the robot can move along the paths generated by RRT.


#rotate(): In this function you will calculate the wheel velocities of the robot (using the ddr_ik() function. [Link to Book](https://www.roboticsbook.org/S52_diffdrive_actions.html#kinematics-in-code)) and also the time duration for the robot to perform pure rotation to orient itself from its current heading to the heading of the next point. You might find [np.arctan2()](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html) useful to find the angle required to rotate. The angle should be between 0 and $2\pi$, so might need to make some adjustments.

#forward(): In this function you will calculate the wheel velocities of the robot (using the ddr_ik() function. [Link to Book](https://www.roboticsbook.org/S52_diffdrive_actions.html#kinematics-in-code)) and also the time duration for the robot to perform pure translation to reach from current position to the next point.


def ddr_ik(v_x, omega, L=0.5, r=0.1):
    # DDR inverse kinematics: calculate wheels speeds from desired velocity. You find the function in book's section 5.2
    
    # v_x (float): Translational velocity of the robot
    # omega (float): Angular velocity of the robot
    # L (float): Distance between robot wheels
    # r (float): Wheel radius of robot    
    

    return (v_x - (L/2)*omega)/r, (v_x + (L/2)*omega)/r


def rotate(init_pose, point, omega=2):


    # Function to perform pure rotation of robot to correct heading based on init_pose and point
    # Hint: (i) What should v_x be in ddr_ik() for pure rotation?
    #       (ii) You might find np.arctan2() usefull in finding the angle to rotate


    # init_pose (gtsam.Pose2): Initial Pose from where the robot starts. Given as (x,y,theta)
    #                           You can get x as init_pose.x().
    # point [float,float]: Destination Point of the robot. Given as [x,y]
    # omega: angular velocity in radians per second #DO NOT CHANGE

    # Return:
    # v_L (float): Left wheel velocity
    # v_R (float): Right wheel velocity
    # t (float): time duration for which the robot should use these velocities


    new_orientation = np.arctan2((point[1] - init_pose.y()), (point[0] - init_pose.x()))
    theta_diff = new_orientation if new_orientation >= 0 else new_orientation + 2*np.pi
    t = abs(theta_diff / omega)
    v_x = 0
    v_L, v_R = ddr_ik(v_x, omega)

    return v_L,v_R, t

  
def rotateRobot(init_pose, v_L, v_R, t, L=0.5, r=0.1):

    # Generates intermediate robot poses based on wheel velocities, useful for visualization

    # init_pose (gtsam.Pose2): Initial Pose from where the robot starts. Given as (x,y,theta)
    # v_L (float): Left wheel velocity
    # v_R (float): Right wheel velocity
    # t (float): time duration for which the robot should use these velocities
    # L (float): Distance between robot wheels
    # r (float): Wheel radius of robot



    #angle_turned = (v_R-v_L)*t*r/L
    angle_turned = 0
    poses = []
    N=5
    for _ in range(N):
        angle_turned += (v_R-v_L)*t*r/(L*N)
        poses.append(gtsam.Pose2(init_pose.x(),init_pose.y(), angle_turned))

    return poses


def forward(init_pose, point, V=1):


    # Function to perform pure translation of robot to point
    # Hint: What should omega in ddr_ik() be in case of pure translation?

    # init_pose (gtsam.Pose2): Initial Pose from where the robot starts as (x,y,theta)
    #                           You can get x as init_pose.x().
    # point [float,float]: Destination Point of the robot. Given as [x,y]
    # V: Translational Velocity of robot #DO NOT CHANGE

    # Return:
    # v_L (float): Left wheel velocity
    # v_R (float): Right wheel velocity
    # t (float): time duration for which the robot should use these velocities


    next_x, next_y = point
    dist = ((init_pose.x() - next_x) **2 + (init_pose.y() - next_y) **2) **(0.5)
    t = dist/V
    v_L, v_R = ddr_ik(V,0)


    return v_L, v_R, t


def forwardRobot(init_pose, v_L, v_R, t, L=0.5, r=0.1):

    # Generates intermediate robot poses based on wheel velocities, useful for visualization
    
    # v_L (float): Left wheel velocity
    # v_R (float): Right wheel velocity
    # t (float): time duration for which the robot should use these velocities
    # L (float): Distance between robot wheels
    # r (float): Wheel radius of robot

    x = init_pose.x()
    y = init_pose.y()
    poses = []
    N = 5

    for i in range(N):
        x += v_L * math.cos(init_pose.theta())*t*r/N
        y += v_L * math.sin(init_pose.theta())*t*r/N

        poses.append(gtsam.Pose2(x,y,init_pose.theta()))

    return poses
    

def moveRobot(init_pose: gtsam.Pose2, path):

    # move robot along given path
    # init_pose (gtsam.Pose2): Starting pose (x,y,theta) of the robot
    # path: path found from RRT

    # return: list of poses along the given path

    poses = [init_pose]
    
    for i in range(1, len(path)):
        curr_pose = poses[-1]
        next_x, next_y = path[i].point
        v_L, v_R, t = rotate(curr_pose, [next_x, next_y])
        poses += rotateRobot(curr_pose, v_L, v_R, t)
        v_L,v_R, t = forward(poses[-1], [next_x, next_y])
        poses += forwardRobot(poses[-1], v_L,v_R, t)

    return poses

starting_pose = gtsam.Pose2(4, 1, np.pi/2)
path, rrt_tree = RRT(np.array([starting_pose.x(), starting_pose.y()]), np.array([0,6]), env)
poses = moveRobot(starting_pose, path)
visualize(env, poses)

import unittest

class TestDiffDrive(unittest.TestCase):
    def test_rotate(self):
        starting_pose = gtsam.Pose2(4,1,np.pi/2)
        points = [[3.37530495, 1.78086881],
                  [2.7506099,  2.56173762]
                  ]
        v_list = []
        for p in points:
          v_list.append(list(rotate(starting_pose,p)))

        v_list_np = np.array(v_list)
        desired_v_np = np.array([[-5.0, 5.0, 1.1227686352900934], 
                      [-5.0, 5.0, 1.1227686352900934], 
                    ])
        np.testing.assert_almost_equal(v_list_np,desired_v_np)

    def test_forward(self):
      starting_pose = gtsam.Pose2(4,1,np.pi/2)

      points = [[3.37530495, 1.78086881],
                  [2.7506099,  2.56173762]
                  ]
      v_list = []
      for p in points:
          v_list.append(list(forward(starting_pose,p)))
      v_list_np = np.array(v_list)
      desired_v_np = np.array([[10.0, 10.0, 1.0000000019626594], 
                                [10.0, 10.0, 2.0000000039253187], 
                                ])
      np.testing.assert_almost_equal(v_list_np,desired_v_np)

        


suite = unittest.TestSuite()
suite.addTest(TestDiffDrive('test_rotate'))
suite.addTest(TestDiffDrive('test_forward'))

unittest.TextTestRunner().run(suite)

##Integration with Object Detection 


# Code to run RRT multiple times based on the coordinates obtained from the detected images. 

# Given
# -----
# saved_pos: (dict) Contains image labels as keys and the coordinates of the next image position as values.
# starting_pose: (gtsam.Pose2) Robot initial pose in coordinate [4,1]
# goal_node: (np.array) coordinate of the initial goal position of image 0.

# TODO
# ----
# 1. Navigate to the goal position using RRT
# 2. When the robot is close enough to the image, (i.e. final node in the found path) 
# sample the image and use the image classifier to get the predicted label. 
# 3. Get the new goal position from saved_pos using the predicted label. 
# 4. Repeat 1-3 until the robot moves though all images. 

# Dictionary of image labels as keys and the next image coordinates as values
saved_pos = dict([(0, [7.75,3]), (1, [0,6]), (2, [7.75,6]), (3, [4,9]), 
                  (4, [12,9]), (5, [8.25,6]), (6, [16,6]), (7, [8.25,3]), 
                  (8, [16,3]), (9, [4,1])])

starting_pose = gtsam.Pose2(4, 1, np.pi/2)
goal_node = np.array([0,3])

path, tree = RRT(np.array([starting_pose.x(), starting_pose.y()]), goal_node, env)
position = moveRobot(starting_pose, path)

for i in range(10):
  for image in env.image_list:
    if np.array_equal(goal_node, image.location):
      curr_image = image
      break
  image_tensor = curr_image.sample()
  index = image_classifier.classify_image(image_tensor)

  curr_pose = poses[-1]
  goal_node = saved_pos[index]

  path, tree = RRT(np.array([curr_pose.x(), curr_pose.y()]), goal_node, env)
  position += moveRobot(curr_pose, path)
# raise NotImplementedError('final integration not implemented')

# visualize all paths
visualize(env, poses)