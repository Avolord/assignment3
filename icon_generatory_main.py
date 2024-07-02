import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import math
from utility import train
# Import the IconDataset class
from icon_dataset import IconDataset 

# Import the IconGenerator class
from icon_generator import IconGenerator

# Define constants
eps = 0.00001
device = "cuda"
dataset_file_name = "ImagerIcon_subset.hdf5"
num_points = 10  # Adjust as necessary

# Load and split the dataset
icon = IconDataset(dataset_file_name)
print("Number of examples in dataset: {}".format(len(icon)))

val_ratio = 0.05
val_size = int(len(icon) * val_ratio)
indices_val = list(range(0, val_size))
indices_train = list(range(val_size, len(icon)))

val_set = torch.utils.data.Subset(icon, indices_val)
train_set = torch.utils.data.Subset(icon, indices_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle=True, drop_last=True)
loss_fn = nn.MSELoss()
network_gen = IconGenerator(num_points).cuda()
train(train_loader, network_gen, loss_fn, num_training_epochs=50)