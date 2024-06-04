# Import libraries
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Constants
DATA_DIR = '/home/h6x/Projects/data_processing/data/processed_data/persistence_images/below_90th/h0h1/npy_all_variables'
DISABL_DIR = f'{DATA_DIR}/EP_DISABL'
COMBINED_FEATURES_DIR = '/home/h6x/Projects/data_processing/data/processed_data/persistence_images/below_90th/h0h1/npy_combined_features'

# Get all the file names to a list without .npy
def get_files(location):
    return [name.split('.')[0] for name in os.listdir(location) if name.endswith('.npy')]

# Get the FIPS codes
fips_codes = get_files(DISABL_DIR)

# Get the folder names to a list without the folders that start with '.'
variables = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]

# Process each FIPS code
for fips_code in tqdm(fips_codes, desc='Processing FIPS Codes'):
    print(f'Processing {fips_code}')
    
    # List to store persistence images for each variable
    persistence_images = []

    # Load the data for each variable
    for variable in variables:
        file_path = f'{DATA_DIR}/{variable}/{fips_code}.npy'
        
        # Check if the file exists before loading
        if os.path.exists(file_path):
            persistence_image = np.load(file_path)
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
