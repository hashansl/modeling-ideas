import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle as pickle
from pylab import *
import os    
import numpy as np

import warnings

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import itertools
from itertools import combinations
from scipy import spatial
import pickle as pickle
import gudhi
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import io
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageChops, ImageFont
import shapely.geometry as geom
from shapely.ops import unary_union
import warnings

import invr

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# get the list of folders in a location
def get_folders(location):
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

states = get_folders('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/SVI2018 census tracts with death rate HepVu-5 classes')

print('Number of states:', len(states))
print(states)

for state in states:
    # create a folder for each state if it does not exist
    os.makedirs(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables/{state}", exist_ok=True)
print('Done creating folders for each state')


selected_variables = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ','NOD_Rate']
selected_variables_without_y = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']

# selected_variables_for_state_with_geo = ['FIPS','EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV','NOD_Rate','geometry']
# selected_variables_for_state = ['EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV']
# selected_variables_tn_with_od = ['EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV','NOD_Rate']

selected_variables_for_state_with_geo = ['FIPS','EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ','NOD_Rate','geometry']
selected_variables_for_state = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']
selected_variables_tn_with_od = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ','NOD_Rate']


def generate_adjacent_counties(dataframe,filtration_threshold,variable_name):

    
    filtered_df = dataframe[dataframe[variable_name] < filtration_threshold]

    # Perform a spatial join to find adjacent precincts
    adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')

    # Filter the results to include only the adjacent states
    adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')

    # Group the resulting dataframe by the original precinct Name and create a list of adjacent precinct Name
    adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()

    adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)

    adjacencies_list = adjacent_counties['adjacent'].tolist()
    county_list = adjacent_counties['county'].tolist()

    merged_df = pd.merge(adjacent_counties, dataframe, left_on='county',right_on='sortedID', how='left')
    merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')

    return adjacencies_list,merged_df,county_list


def form_simplicial_complex(adjacent_county_list,county_list):
    max_dimension = 3

    V = []
    V = invr.incremental_vr(V, adjacent_county_list, max_dimension,county_list)

    return V

def fig2img(fig):
     #convert matplot fig to image and return it

     buf = io.BytesIO()
     fig.savefig(buf)
     buf.seek(0)
     img = Image.open(buf)
     return img

def plot_simplicial_complex(dataframe,V):

    #city centroids
    city_coordinates = {city.sortedID: np.array((city.geometry.centroid.x, city.geometry.centroid.y)) for _, city in dataframe.iterrows()}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_axis_off() 

    # Plot the "wyoming_svi" DataFrame
    dataframe.plot(ax=ax, edgecolor='black', linewidth=0.3, color="white")

    # Plot the centroid of the large square with values
    # for i, row in dataframe.iterrows():
    #     centroid = row['geometry'].centroid
    #     # text_to_display = f"FIPS: {row['FIPS']}\nFilteration: {row['EP_SNGPNT']}"
    #     plt.text(centroid.x, centroid.y, str(row['FIPS']), fontsize=8, ha='center', color="black")
    #     # plt.text(centroid.x, centroid.y, text_to_display, fontsize=10, ha='center', color="black")

    for edge_or_traingle in V:

        
        if len(edge_or_traingle) == 2:
            # Plot an edge
            ax.plot(*zip(*[city_coordinates[vertex] for vertex in edge_or_traingle]), color='red', linewidth=1)
            # img = fig2img(fig)
            # list_gif.append(img)
        elif len(edge_or_traingle) == 3:
            # Plot a triangle
            ax.add_patch(plt.Polygon([city_coordinates[vertex] for vertex in edge_or_traingle], color='green', alpha=0.2))
            # img = fig2img(fig)
            # list_gif.append(img)
    # plt.show()
    plt.close()


for state in states:

    print('Processing:', state)

    # Initialize dictionaries to store the percentiles
    percentiles_50 = {}
    percentiles_75 = {}
    percentiles_90 = {}

    try:

        svi_od = gpd.read_file(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/SVI2018 census tracts with death rate HepVu-5 classes/{state}/{state}.shp')

        # Replace -999 with 0 for each variable in selected_variables_for_state
        for variable in selected_variables_for_state:
            svi_od[variable] = svi_od[variable].replace(-999, 0)


        # Calculate the percentiles for each variable
        for variable in selected_variables_for_state:
            percentiles_50[variable] = svi_od[variable].quantile(0.5)
            percentiles_75[variable] = svi_od[variable].quantile(0.75)
            percentiles_90[variable] = svi_od[variable].quantile(0.9)


        tn_filtered = svi_od[selected_variables_for_state_with_geo]

        #reset index
        tn_filtered = tn_filtered.reset_index(drop=True)

        # get the uniques fips codes
        fips = tn_filtered['FIPS'].unique()
        
        #multiple

        
        # # Create selected_variables_and_threshold using a loop
        # selected_variables_and_threshold_percentiles_50 = {}
        # for variable in selected_variables_for_state:
        #     selected_variables_and_threshold_percentiles_50[variable] = percentiles_50[variable]

        # # create a empty dictionary
        # edges_and_traingles_for_each_variable_below_50th_percentile = {}

        # for variable_name, threshold in selected_variables_and_threshold_percentiles_50.items():

        #     # Sorting based on the variable and selecting only the FIPS and the variable columns is important
        #     # Also we need to keep  the dataframe sorted based on the variable

        #     df_one_variable = tn_filtered[['FIPS',variable_name, 'geometry']]

        #     # # Sorting the DataFrame based on the 'rate' column
        #     df_one_variable = df_one_variable.sort_values(by=variable_name)
        #     df_one_variable['sortedID'] = range(len(df_one_variable))

        #     # Convert the DataFrame to a GeoDataFrame
        #     df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
        #     df_one_variable.crs = "EPSG:3395"  # This is a commonly used projected CRS


        #     # print(df_one_variable.head(100))

        #     adjacencies_list,adjacent_counties_df,county_list = generate_adjacent_counties(df_one_variable,threshold,variable_name)

        #     # create a dictionary adjacent_counties_df column county as key and column adjacent as value(to avoid NULL adjacencies error)
        #     adjacent_counties_dict = dict(zip(adjacent_counties_df['county'],adjacent_counties_df['adjacent']))

        #     # this take only counties that have adjacent counties
        #     county_list = adjacent_counties_df['county'].tolist()

        #     V = form_simplicial_complex(adjacent_counties_dict,county_list)

        #     # This is a new feature that I added to the code. It creates a new list replace the sorted ID with the FIPS on the V list
        #     # create a new list replace the sorted ID with the FIPS on the V list
        #     V_FIPS = [[df_one_variable.iloc[x]['FIPS'] for x in i] for i in V]
            
        #     #add V list to the edges_and_traingles_for_each_variable dictionary with the key as the variable name
        #     edges_and_traingles_for_each_variable_below_50th_percentile[variable_name] = V_FIPS

        #     # # # store the list of images for each variable
        #     # # list_img = []

        #     # # plot the simplicial complex
        #     print(f"Plotting simplicial complex for {variable_name} variable at threshold {threshold}")
        #     # plot_simplicial_complex(df_one_variable,V)

        
        
        # # Create selected_variables_and_threshold using a loop
        # selected_variables_and_threshold_percentiles_75 = {}
        # for variable in selected_variables_for_state:
        #     selected_variables_and_threshold_percentiles_75[variable] = percentiles_75[variable]
        
        # # create a empty dictionary
        # edges_and_traingles_for_each_variable_below_75th_percentile = {}

        # for variable_name, threshold in selected_variables_and_threshold_percentiles_75.items():

        #     # Sorting based on the variable and selecting only the FIPS and the variable columns is important
        #     # Also we need to keep  the dataframe sorted based on the variable

        #     df_one_variable = tn_filtered[['FIPS',variable_name, 'geometry']]

        #     # # Sorting the DataFrame based on the 'rate' column
        #     df_one_variable = df_one_variable.sort_values(by=variable_name)
        #     df_one_variable['sortedID'] = range(len(df_one_variable))

        #     # Convert the DataFrame to a GeoDataFrame
        #     df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
        #     df_one_variable.crs = "EPSG:3395"  # This is a commonly used projected CRS

        #     adjacencies_list,adjacent_counties_df,county_list = generate_adjacent_counties(df_one_variable,threshold,variable_name)

        #     # create a dictionary adjacent_counties_df column county as key and column adjacent as value(to avoid NULL adjacencies error)
        #     adjacent_counties_dict = dict(zip(adjacent_counties_df['county'],adjacent_counties_df['adjacent']))

        #     # this take only counties that have adjacent counties
        #     county_list = adjacent_counties_df['county'].tolist()

        #     V = form_simplicial_complex(adjacent_counties_dict,county_list)

        #     # This is a new feature that I added to the code. It creates a new list replace the sorted ID with the FIPS on the V list
        #     # create a new list replace the sorted ID with the FIPS on the V list
        #     V_FIPS = [[df_one_variable.iloc[x]['FIPS'] for x in i] for i in V]

        #     #add V list to the edges_and_traingles_for_each_variable dictionary with the key as the variable name
        #     edges_and_traingles_for_each_variable_below_75th_percentile[variable_name] = V_FIPS

        #     # # # store the list of images for each variable
        #     # # list_img = []

        #     # # plot the simplicial complex
        #     print(f"Plotting simplicial complex for {variable_name} variable at threshold {threshold}")
        #     # plot_simplicial_complex(df_one_variable,V)

        # 90 th percentile ---------------------------------------

        # Create selected_variables_and_threshold using a loop
        selected_variables_and_threshold_percentiles_90 = {}
        for variable in selected_variables_for_state:
            selected_variables_and_threshold_percentiles_90[variable] = percentiles_90[variable]
        
        
        # create a empty dictionary
        edges_and_traingles_for_each_variable_below_90th_percentile = {}

        for variable_name, threshold in selected_variables_and_threshold_percentiles_90.items():

            # Sorting based on the variable and selecting only the FIPS and the variable columns is important
            # Also we need to keep  the dataframe sorted based on the variable

            df_one_variable = tn_filtered[['FIPS',variable_name, 'geometry']]

            # # Sorting the DataFrame based on the 'rate' column
            df_one_variable = df_one_variable.sort_values(by=variable_name)
            df_one_variable['sortedID'] = range(len(df_one_variable))

            # Convert the DataFrame to a GeoDataFrame
            df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
            df_one_variable.crs = "EPSG:3395"  # This is a commonly used projected CRS

            adjacencies_list,adjacent_counties_df,county_list = generate_adjacent_counties(df_one_variable,threshold,variable_name)

            # create a dictionary adjacent_counties_df column county as key and column adjacent as value(to avoid NULL adjacencies error)
            adjacent_counties_dict = dict(zip(adjacent_counties_df['county'],adjacent_counties_df['adjacent']))

            # this take only counties that have adjacent counties
            county_list = adjacent_counties_df['county'].tolist()

            V = form_simplicial_complex(adjacent_counties_dict,county_list)

            # This is a new feature that I added to the code. It creates a new list replace the sorted ID with the FIPS on the V list
            # create a new list replace the sorted ID with the FIPS on the V list
            V_FIPS = [[df_one_variable.iloc[x]['FIPS'] for x in i] for i in V]

            #add V list to the edges_and_traingles_for_each_variable dictionary with the key as the variable name
            edges_and_traingles_for_each_variable_below_90th_percentile[variable_name] = V_FIPS

            # # # store the list of images for each variable
            # # list_img = []

            # # plot the simplicial complex
            print(f"Plotting simplicial complex for {variable_name} variable at threshold {threshold}")
            plot_simplicial_complex(df_one_variable,V)


        ###############################################################################################################################

        # Initialize lists to store selected regions for each variable
        selected_regions = {variable: [] for variable in selected_variables_for_state}

        # Loop through the dictionary and for each variable, create a list of edges
        for variable_name, V_FIPS in edges_and_traingles_for_each_variable_below_90th_percentile.items():
            # For each set of vertices in V_FIPS
            for vertex_set in V_FIPS:
                if len(vertex_set) in (2, 3):
                    # Add vertices to the appropriate list if not already present
                    for vertice in vertex_set:
                        if vertice not in selected_regions[variable_name]:
                            selected_regions[variable_name].append(vertice)
        
        
        # this set of variables needed - specially STCNTY to identify the county
        selected_variables_tn_with_censusinfo = ['FIPS','STCNTY','EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ','NOD_Rate','geometry']

        # get a filtered dataframe with the selected regions
        filtered_df_census = svi_od[selected_variables_tn_with_censusinfo]

        # get the unique counties
        unique_counties = svi_od['STCNTY'].unique()



        # Process each county
        for county in unique_counties:
            print(f"County: {county}")

            # Create a temp_df with the selected county
            temp_census = filtered_df_census[filtered_df_census['STCNTY'] == county]

            # Initialize a dictionary to store the data for each variable
            selected_coordinates_dic = {}

            # Loop through each selected variable and process the data
            for variable in selected_variables_for_state:
                # Filter the temp_census dataframe for the selected regions for the current variable
                selected_census = temp_census[temp_census['FIPS'].isin(selected_regions[variable])][['FIPS', variable, 'geometry']]
                
                # Create a new column with the coordinates of the centroid of each polygon
                selected_census['coords'] = selected_census['geometry'].apply(lambda x: x.representative_point().coords[:])
                selected_census['coords'] = [coords[0] for coords in selected_census['coords']]
                
                # Add the processed data to the dictionary
                selected_coordinates_dic[variable] = selected_census

            # Save the dictionary to a pickle file
            with open(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/selected coordinates for each state - percentiles(below 90th)- all variables/{state}/selected_coordinates_dic_{county}.pkl', 'wb') as f:
                pickle.dump(selected_coordinates_dic, f)
        
        print(f"Finished processing {state}")

    except Exception as e:
            print(f"Error processing {state}: {e}")
            continue  # Continue to the next iteration if an error occurs

print('All states processed.')
