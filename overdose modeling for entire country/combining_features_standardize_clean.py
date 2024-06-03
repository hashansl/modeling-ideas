# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Constants
DATA_DIR = '/home/h6x/Projects/data_processing/data/processed_data/persistence_images/below_90th/h0h1/npy_all_variables'
DISABL_DIR = f'{DATA_DIR}/EP_DISABL'
COMBINED_FEATURES_DIR = '/home/h6x/Projects/data_processing/data/processed_data/persistence_images/below_90th/h0h1/npy_combined_features_standardized'

# Get all the file names to a list without .npy
def get_files(location):
    return [name.split('.')[0] for name in os.listdir(location) if name.endswith('.npy')]

# Get the FIPS codes
fips_codes = get_files(DISABL_DIR)

# Get the folder names to a list without the folders that start with '.'
variables = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]

# Initialize a dictionary to hold persistence images for each variable
channels_dic = {variable: [] for variable in variables}

# Process each FIPS code
for fips_code in tqdm(fips_codes, desc='Processing FIPS Codes to get mean and std'):
    print(f'Processing {fips_code}')
    
    # List to store persistence images for each variable
    persistence_images = []

    # Load the data for each variable
    for variable in variables:
        file_path = f'{DATA_DIR}/{variable}/{fips_code}.npy'
        
        # Check if the file exists before loading
        if os.path.exists(file_path):
            persistence_image = np.load(file_path)
            channels_dic[variable].append(persistence_image)
        else:
            print(f'File not found: {file_path}')

# Get the mean and standard deviation of each variable
mean_std_dic = {variable: [] for variable in variables}

for variable in variables:
    mean = np.mean(channels_dic[variable])
    std = np.std(channels_dic[variable])
    mean_std_dic[variable].append(mean)
    mean_std_dic[variable].append(std)

# Process each FIPS code again to standardize the persistence images
for fips_code in tqdm(fips_codes, desc='Processing FIPS Codes to standardize'):
    print(f'Processing {fips_code}')
    
    # List to store persistence images for each variable
    persistence_images = []

    # Load the data for each variable
    for variable in variables:
        file_path = f'{DATA_DIR}/{variable}/{fips_code}.npy'
        
        # Check if the file exists before loading
        if os.path.exists(file_path):
            persistence_image = np.load(file_path)

            # Standardize the persistence images
            mean = mean_std_dic[variable][0]
            std = mean_std_dic[variable][1]
            persistence_image = (persistence_image - mean) / std

            persistence_images.append(persistence_image)
        else:
            print(f'File not found: {file_path}')
    
    # Concatenate the persistence images along the last axis
    combined_matrix = np.stack(persistence_images, axis=-1)

    fips_code_int = int(fips_code)

    # Save the combined persistence image
    output_path = f'{COMBINED_FEATURES_DIR}/{fips_code_int}.npy'
    np.save(output_path, combined_matrix)

print('Done processing FIPS codes')
