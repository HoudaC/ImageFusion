from utils import read_data, define_stride
import numpy as np
from generate_dataloader import sr_dataloader
#model definition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from SRCNN_model import SRCNN
from training import train
from testing import test
from preprocessing import down_up_sentinel_4, set_seed

# Define the SRCNN model in PyTorch
import torch
import torch.nn as nn

set_seed(42)

data_train_dir = "./Dataset/train"
data_val_dir = "./Dataset/val"
img_size=32
train_dataloader_batch_size = 2
patch = (img_size, img_size)

img_train_all, _ = read_data(data_train_dir,500)
img_val_all, _ = read_data(data_val_dir,100)
img_train_all = np.stack(img_train_all, axis=0)
img_val_all = np.stack(img_val_all, axis=0)
strides = define_stride(img_train_all[0].shape[1], img_size)[0], define_stride(img_train_all[0].shape[2], img_size)[0]
train_dataloader = sr_dataloader(img_train_all, patch, strides, train_dataloader_batch_size,
                                               applytransform=True)
val_dataloader = sr_dataloader(img_val_all, patch, strides, train_dataloader_batch_size)


# Initialize the model
model = SRCNN()  # Assuming SRCNN is defined elsewhere

# Loss function: Mean Squared Error Loss
criterion = nn.MSELoss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Initialize the learning rate scheduler (StepLR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Decrease lr by half every 50 epochs


# Train the model for 10 epochs and validate every 2 epochs
# train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=250, scheduler=scheduler)
train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=1000, save_path="weights/best_model.pth",scheduler=scheduler)


# Test the model on a sample image


# test(model, test_img_lr, patch, strides)