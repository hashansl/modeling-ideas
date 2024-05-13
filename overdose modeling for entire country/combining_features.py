import geopandas as gpd
import pandas as pd
import numpy as np
import os

# get all the file naems toa a list without .npy
def get_files(location):
    return [name.split('.')[0] for name in os.listdir(location) if name.endswith('.npy')]

fips_codes = get_files('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy/EP_MOBILE/')

# get the folder names to a list without the folders that start with '.'
variables = [f for f in os.listdir('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy') if not f.startswith('.')]

# print(variables)
# print(len(fips_codes))

for fips_code in fips_codes:
    print(f'Processing {fips_code}')

    # Load the data
    peristence_image_1 = np.load(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy/{variables[0]}/{fips_code}.npy')
    # peristence_image_2 = np.load(f'./results/persistence images/percentiles/H0H1 np/{variables[1]}/{fips_code}.npy')
    # peristence_image_3 = np.load(f'./results/persistence images/percentiles/H0H1 np/{variables[2]}/{fips_code}.npy')
    peristence_image_4 = np.load(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy/{variables[3]}/{fips_code}.npy')
    peristence_image_5 = np.load(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy/{variables[4]}/{fips_code}.npy')

    # # print persistence image shapes 
    # print(peristence_image_1.shape)
    # print(peristence_image_2.shape)
    # print(peristence_image_3.shape)
    # print(peristence_image_4.shape)
    # print(peristence_image_5.shape)

    # Concatenate the persistence images
    #combined_matrix = np.stack((peristence_image_1, peristence_image_2, peristence_image_3, peristence_image_4, peristence_image_5), axis=-1)
    combined_matrix = np.stack((peristence_image_1, peristence_image_4, peristence_image_5), axis=-1)

    # save the persistence image
    np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy 3 channels/{fips_code}', combined_matrix)