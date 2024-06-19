"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import torch
import numpy as np
import data_loader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NUM_WORKERS = os.cpu_count()

NUM_WORKERS =0

def create_dataloaders(
    annotation_file_path: str, 
    root_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training a DataLoader for Auto Encoder.


  """

  #---
  data_set = data_loader.data_loader_persistence_img(annotation_file_path=annotation_file_path, root_dir=root_dir, transform=transform)

  #---
  # Set the random seed
  torch.manual_seed(42)
  np.random.seed(42)

  # Turn images into data loaders
  train_dataloader = DataLoader(
      data_set,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )



  return train_dataloader