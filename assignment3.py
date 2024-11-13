# Packages for getting the data

import gzip
import numpy as np
from io import BytesIO
from requests import get
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Package for checking data is formatted correctly
import plotly.express as px

#Packages for making model
import torch.nn as nn
import torch.nn.functional as F


# Make Dataloader (Based on week 7 slides)

class FashionMNIST(Dataset):
  def __init__(self, image_url, label_url):
    image_response = get(image_url)
    with gzip.GzipFile(fileobj=BytesIO(image_response.content), mode = "rb") as file:
      image_data = np.frombuffer(file.read(), dtype = np.uint8)
      self.image_data = torch.tensor(image_data[16:].reshape(-1, 28, 28),dtype=torch.float32)
    label_response = get(label_url)
    with gzip.GzipFile(fileobj=BytesIO(label_response.content), mode = "rb") as file:
      self.label_data = np.frombuffer(file.read(), dtype = np.uint8)
    self.length = len(self.image_data)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    return self.image_data[index], self.label_data[index]

train_data = FashionMNIST("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz", "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz" )
test_data = FashionMNIST("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz","https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz" )

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)



#Sanity check for training data (the pink elephant in the corner told me I should do one of those)
title = train_data.label_data[10]
px.imshow(train_data.image_data[10],title=str(title))

#Sanity check for training data
title = test_data.label_data[2]
px.imshow(test_data.image_data[2],title=str(title))

# That image data is looking image-y and the labels match


# Make a model
class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FashionNet()

#Slightly modified LeNet-5, from this pytorch tutorial: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#loss-function
#Cutting edge technology straight from the 1990's. Someone probably looked up info on this with Netscape Navigator while wearing parachute pants and a backwards hat
#Originally tried to follow the structure of 1998 MNIST paper (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), but they were using 32x32 input for the MNIST? And when I tried to pad the Fashion MNIST 28x28 input up to 32x32 everything broke




#import script to evaluate model

PATH = "model.pt"

# Blank model to load weights into
model = FashionNet()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
EPOCH = checkpoint['epoch']

