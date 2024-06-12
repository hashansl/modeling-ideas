# import libraries
import numpy as np
import re
import os
import glob
import pickle
import traceback
from tqdm import tqdm
from ripser import Rips
from persim import PersistenceImager


# remove warnings
import warnings
warnings.filterwarnings("ignore")

# Define constant variables
ROOT_DIR = '/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/'
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results/persistence images/below 90th percentile/h1/test')
VARIABLES = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']
PERSISTENCE_IMAGE_SHAPE = (310, 310)
PERSISTENCE_IMAGE_PARAMS = {
    'pixel_size': 0.001,
    'birth_range': (0.0, 0.31),
    'pers_range': (0.0, 0.31),
    'kernel_params': {'sigma': 0.00004}
}

# Function to get the list of folders in a specified location
def get_folders(location):
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

# Get the list of state folders
states = get_folders(DATA_DIR)

# Create a folder for each variable if it does not exist
for variable in VARIABLES:
    os.makedirs(os.path.join(RESULTS_DIR, variable), exist_ok=True)
print('Done creating folders for each variable')

# Loop through each state
for state in tqdm(states, desc="Processing states"):
    print('Processing:', state)

    try:
        # Load data from pickle files into a dictionary
        data = {}

        for file in glob.glob(os.path.join(DATA_DIR, state, '*.pkl')):
            with open(file, 'rb') as f:
                
                # Extract the last 20 characters of the file name
                extracted_words = file[-20:]

                # Search for numbers in the extracted string
                match = re.search(r'(\d+)', extracted_words)
                if match:
                    extracted_number = match.group(1)
                    # Load the pickle file data into the dictionary
                    data[extracted_number] = pickle.load(f)
                else:
                    print("No number found in the string.")

        # Process each county (FIPS) in the data
        for fips, dictionary in data.items():
            # Dictionary where the key is the county code (FIPS) and the value is another dictionary
    
            for key, value in dictionary.items():

                # If the value is not empty, process it
                if len(value) > 0:

                    # Convert coordinates to a numpy array
                    data_coordinates = np.array([np.array(coord) for coord in value['coords']])

                    # Create persistence diagram using Rips class
                    rips = Rips(maxdim=1, coeff=2)
                    dgms = rips.fit_transform(data_coordinates)

                    # Diagrams for H1
                    diagrams_h0 = dgms[0]
                    diagrams_h1 = dgms[1]

                    # If H1 diagram is not empty, process it
                    if len(diagrams_h1) > 0:
                        pimgr_1 = PersistenceImager(PERSISTENCE_IMAGE_PARAMS['pixel_size'])
                        pimgr_1.fit(diagrams_h1)

                        pimgr_1.pixel_size = PERSISTENCE_IMAGE_PARAMS['pixel_size']
                        pimgr_1.birth_range = PERSISTENCE_IMAGE_PARAMS['birth_range']
                        pimgr_1.pers_range = PERSISTENCE_IMAGE_PARAMS['pers_range']
                        pimgr_1.kernel_params = PERSISTENCE_IMAGE_PARAMS['kernel_params']

                        image_h1 = pimgr_1.transform(diagrams_h1)

                    # Save the persistence image
                    save_path = os.path.join(RESULTS_DIR, key, fips)

                    if len(diagrams_h1) > 0:
                        peristence_image = np.rot90(image_h1, k=1) 
                        np.save(save_path, peristence_image)
                    else:
                        peristence_image = np.zeros(PERSISTENCE_IMAGE_SHAPE)
                        np.save(save_path, peristence_image)
                else:
                    # If there is no data to compute persistence image, save an empty image
                    peristence_image = np.zeros(PERSISTENCE_IMAGE_SHAPE)
                    np.save(os.path.join(RESULTS_DIR, key, fips), peristence_image)
        print('Done processing:', state)

    except Exception as e:
        print(f"Error processing {state}: {e}")
        traceback.print_exc()
        continue  # Continue to the next iteration if an error occurs

print('All states processed.')