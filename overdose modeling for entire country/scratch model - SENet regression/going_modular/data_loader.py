
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import os
from skimage import io
import geopandas as gpd
import matplotlib.pyplot as plt

import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision.transforms import ToPILImage


class data_loader_persistence_img(Dataset):

    def __init__(self,annotation_file_path,root_dir,transform=None):
        self.dtype = {'STCNTY': str}
        self.annotations = pd.read_csv(annotation_file_path,dtype=self.dtype)
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(self.annotations['percentile'].unique())
        self.to_pil = ToPILImage()  # Initialize ToPILImage transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):

        npy_file_path = os.path.join(self.root_dir, str(self.annotations.iloc[index,0]) + '.npy')

        img = np.load(npy_file_path).astype(np.float32)
        # img = self.to_pil(img)

        y_label = torch.tensor((self.annotations.iloc[index]['NOD']).astype(np.float32))

        if self.transform:
            img = self.transform(img)
        return (img, y_label)

    def get_class_names(self):
        return self.class_names




# root_dir = "/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy_combined_features" # has 5 classes
# annotation_file_path = "/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/annotation 2018/annotation.csv"

# dataset = data_loader_persistence_img(annotation_file_path=annotation_file_path,root_dir=root_dir,transform=transforms.ToTensor())

# print(len(dataset))

# print(len(dataset[0]))

# print(dataset[0][1])
# print(dataset[0][0].shape)

# train_set, test_set = torch.utils.data.random_split(dataset, [70, 25])


# #---
# train_data = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
# test_data = DataLoader(dataset=test_set, batch_size=16, shuffle=False)

# class_names = dataset.get_class_names()
# print(class_names)







