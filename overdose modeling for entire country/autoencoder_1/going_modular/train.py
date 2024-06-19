"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, autoencoder
from torchvision import transforms
from timeit import default_timer as timer 
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import get_dataloaders_mnist
from helper_train import train_autoencoder_v1
from utils import set_deterministic, set_all_seeds
from helper_plotting import plot_training_loss
from helper_plotting import plot_generated_images
from helper_plotting import plot_latent_space_with_labels

# Setup hyperparameters
RANDOM_SEED = 123
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

set_deterministic
set_all_seeds(RANDOM_SEED)

# Setup directories
root_dir = "/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy_combined_features" # has 5 classes
annotation_file_path = "/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/annotation 2018/annotation.csv"


def train_autoencoder(NUM_EPOCHS=NUM_EPOCHS, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, ROOT_DIR=root_dir, ANNOTATION_FILE_PATH=annotation_file_path):
    
    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Create transforms
    # data_transform = transforms.Compose([
    # transforms.ToTensor()
    # ])

    # # Create DataLoaders with help from data_setup.py
    # train_dataloader = data_setup.create_dataloaders(
    #     annotation_file_path=ANNOTATION_FILE_PATH,
    #     root_dir=ROOT_DIR,
    #     transform=data_transform,
    #     batch_size=BATCH_SIZE
    # )

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
      batch_size=BATCH_SIZE, 
      num_workers=2, 
      validation_fraction=0.)
    
    # Checking the dataset
    print('Training Set:\n')
    for images, labels in train_loader:  
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
        
    # Checking the dataset
    print('\nValidation Set:')
    for images, labels in valid_loader:  
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break

    # Checking the dataset
    print('\nTesting Set:')
    for images, labels in test_loader:  
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    

    set_all_seeds(RANDOM_SEED)

    # Create model
    model = autoencoder.auto_encoder().to(device)

    # Create loss function and optimizer
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    log_dict = train_autoencoder_v1(num_epochs=NUM_EPOCHS, model=model, 
                                optimizer=optimizer, device=device, 
                                train_loader=train_loader,
                                skip_epoch_stats=True,
                                logging_interval=250)
    
    plot_training_loss(log_dict['train_loss_per_batch'], NUM_EPOCHS)

    plot_generated_images(data_loader=train_loader, model=model, device=device)           




if __name__ == "__main__":
    train_autoencoder()