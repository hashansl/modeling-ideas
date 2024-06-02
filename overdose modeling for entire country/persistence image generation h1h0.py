# import libraries
import numpy as np
import re
import matplotlib.pyplot as plt
import os

from ripser import Rips
from persim import PersistenceImager

import glob
import pickle
import geopandas as gpd
from tqdm import tqdm


# remove warnings
import warnings
warnings.filterwarnings("ignore")

# get the list of folders in a location
def get_folders(location):
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

states = get_folders('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables')

variables = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']

for variable in variables:
    # create a folder for each state if it does not exist
    os.makedirs(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy 16 channels/{variable}", exist_ok=True)
print('Done creating folders for each variable')

# print('Number of states:', len(states))
# print(states)

# loop through each state
for state in tqdm(states, desc="Processing states"):
    print('Processing:', state)

    try:
        # load the dictonary into a dictionary from the pkl files
        data = {}

        for file in glob.glob(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables/{state}/*.pkl"):
            with open(file, 'rb') as f:
                
                # select last 20 characters of the file name
                extracted_words = file[-20:]

                match = re.search(r'(\d+)', extracted_words)
                if match:
                    extracted_number = match.group(1)
                    # print(extracted_number)
                    # print(type(extracted_number))

                    data[extracted_number] = pickle.load(f)
                else:
                    print("No number found in the string.")
        

        #################
        for fips, dictionary in data.items():
            # data is a dictionary where the key is the county code and the value is a another dictionary
    
            for key, value in dictionary.items():

                # if the value is not empty, append it to the list
                if len(value)> 0:

                    # get the coordinates into a numpy array
                    data_coordinates = np.array([np.array(coord) for coord in value['coords']])

                    # creating the persistence diagram from rips class
                    rips = Rips(maxdim=1, coeff=2)
                    dgms = rips.fit_transform(data_coordinates)

                    # seperate the diagrams H0 and H1
                    # diagrams_h0 = dgms[0]
                    diagrams_h1 = dgms[1]


                    if len(diagrams_h1) > 0:

                        pimgr_1 = PersistenceImager(pixel_size=0.001)
                        pimgr_1.fit(diagrams_h1)

                        pimgr_1.pixel_size = 0.001
                        pimgr_1.birth_range = (0.0, 0.31)
                        pimgr_1.pers_range = (0.0, 0.31)

                        pimgr_1.kernel_params = {'sigma': 0.00004}
                        image_h1 = pimgr_1.transform(diagrams_h1)

                    # saving the plot

                    if len(diagrams_h1) > 0:
                        # Rotate 90 degrees to the left(k=3), 90 degrees to the right(k=1), 180 degrees(k=2)
                        C_rotated = np.rot90(image_h1, k=1) 

                        # plt.figure(figsize=(2.4, 2.4))
                        # plt.imshow(C_rotated, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
                        # plt.axis('off')  # Turn off axis
                        # # tennessee/results/persistence images/percentiles/below 75/png
                        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to remove borders
                        
                        # plt.savefig(f'./results/persistence images/percentiles/below 90/h1/png/{key}/' + fips +'.png')
                        # plt.show()
                        # plt.close()
                        np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy 16 channels/{key}/' + fips, C_rotated)
                    else:
                        empty_image = np.zeros((310, 310))
                        np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy 16 channels/{key}/' + fips, empty_image)
                        # no_persitence_data_points.append(fips)
                else:
                    # no_data_points_to_compute_persistence_image.append(fips)
                    empty_image = np.zeros((310, 310))
                    np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy 16 channels/{key}/' + fips, empty_image)
        print('Done processing:', state)

    except Exception as e:
        print(f"Error processing {state}: {e}")
        continue  # Continue to the next iteration if an error occurs

print('All states processed.')