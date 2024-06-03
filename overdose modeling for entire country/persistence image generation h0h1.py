# import libraries
import numpy as np
import re
import os
import glob
import pickle
from tqdm import tqdm
from ripser import Rips
from persim import PersistenceImager
import traceback



# remove warnings
import warnings
warnings.filterwarnings("ignore")

# Function to get the list of folders in a specified location
def get_folders(location):
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

# Get the list of state folders
states = get_folders('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables')

# List of variables to be processed
variables = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']

# Create a folder for each variable if it does not exist
for variable in variables:
    os.makedirs(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1h0/npy 16 channels/{variable}", exist_ok=True)
print('Done creating folders for each variable')


# loop through each state
for state in tqdm(states, desc="Processing states"):
    print('Processing:', state)

    try:
        # Load data from pickle files into a dictionary
        data = {}

        for file in glob.glob(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables/{state}/*.pkl"):
            with open(file, 'rb') as f:
                
                # Extract the last 20 characters of the file name
                extracted_words = file[-20:]

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
                # Key is the variable name and value is a dataframe with the selected coordinates of the county

                # If the value is not empty, process it
                if len(value)> 0:

                    # Convert coordinates to a numpy array
                    data_coordinates = np.array([np.array(coord) for coord in value['coords']])

                    # Create persistence diagram using Rips class
                    rips = Rips(maxdim=1, coeff=2)
                    dgms = rips.fit_transform(data_coordinates)

                    # Seperate the diagrams H0 and H1
                    diagrams_h0 = dgms[0]
                    diagrams_h1 = dgms[1]


                    # If H0 diagram is not empty, process it
                    if len(diagrams_h0) > 1: 

                        # remove last data point in H0 diagram - it is infinity
                        diagrams_h0_without_inf = diagrams_h0[0:-1]

                        pimgr_0 = PersistenceImager(pixel_size=0.001)
                        pimgr_0.fit(diagrams_h0_without_inf)

                        pimgr_0.pixel_size = 0.001
                        pimgr_0.birth_range = (0.0, 0.31)
                        pimgr_0.pers_range = (0.0, 0.31)

                        pimgr_0.kernel_params = {'sigma': 0.00004}
                        image_h0 = pimgr_0.transform(diagrams_h0_without_inf)

                    # If H1 diagram is not empty, process it
                    if len(diagrams_h1) > 0:

                        pimgr_1 = PersistenceImager(pixel_size=0.001)
                        pimgr_1.fit(diagrams_h1)

                        pimgr_1.pixel_size = 0.001
                        pimgr_1.birth_range = (0.0, 0.31)
                        pimgr_1.pers_range = (0.0, 0.31)

                        pimgr_1.kernel_params = {'sigma': 0.00004}
                        image_h1 = pimgr_1.transform(diagrams_h1)

                    # Save the persistence image as a numpy file
                    if len(diagrams_h0) > 1 & len(diagrams_h1) > 0:
                        peristence_image = np.rot90(image_h0+image_h1, k=1) 
                        np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1h0/npy 16 channels/{key}/' + fips, peristence_image)
                    
                    elif len(diagrams_h0) > 1 & len(diagrams_h1) == 0:
                        # Rotate 90 degrees to the left(k=3), 90 degrees to the right(k=1), 180 degrees(k=2)
                        peristence_image = np.rot90(image_h0, k=1) 
                        np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1h0/npy 16 channels/{key}/' + fips, peristence_image)
                    
                    elif len(diagrams_h0) < 1 & len(diagrams_h1) > 0:
                        # Rotate 90 degrees to the left(k=3), 90 degrees to the right(k=1), 180 degrees(k=2)
                        peristence_image = np.rot90(image_h1, k=1) 
                        np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1h0/npy 16 channels/{key}/' + fips, peristence_image)
                    
                    else:
                        peristence_image = np.zeros((310, 310))
                        np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1h0/npy 16 channels/{key}/' + fips, peristence_image)
                else:
                    # If there is no data to compute persistence image, save an empty image
                    peristence_image = np.zeros((310, 310))
                    np.save(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1h0/npy 16 channels/{key}/' + fips, peristence_image)
        print('Done processing:', state)

    except Exception as e:
        print(f"Error processing {state}: {e}")
        traceback.print_exc()
        continue  # Continue to the next iteration if an error occurs

print('All states processed.')