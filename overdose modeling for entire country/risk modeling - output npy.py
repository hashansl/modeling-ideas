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




# selected_variables = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ','NOD_Rate']
# selected_variables_without_y = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']
# selected_variables_tn_with_geo = ['FIPS','EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV','NOD_Rate','geometry']
# selected_variables_tn = ['EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV']
# selected_variables_tn_with_od = ['EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV','NOD_Rate']



# def generate_adjacent_counties(dataframe,filtration_threshold,variable_name):

    
#     filtered_df = dataframe[dataframe[variable_name] < filtration_threshold]

#     # Perform a spatial join to find adjacent precincts
#     adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')

#     # Filter the results to include only the adjacent states
#     adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')

#     # Group the resulting dataframe by the original precinct Name and create a list of adjacent precinct Name
#     adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()

#     adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)

#     adjacencies_list = adjacent_counties['adjacent'].tolist()
#     county_list = adjacent_counties['county'].tolist()

#     merged_df = pd.merge(adjacent_counties, dataframe, left_on='county',right_on='sortedID', how='left')
#     merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')

#     return adjacencies_list,merged_df,county_list


# def form_simplicial_complex(adjacent_county_list,county_list):
#     max_dimension = 3

#     V = []
#     V = invr.incremental_vr(V, adjacent_county_list, max_dimension,county_list)

#     return V

# def fig2img(fig):
#      #convert matplot fig to image and return it

#      buf = io.BytesIO()
#      fig.savefig(buf)
#      buf.seek(0)
#      img = Image.open(buf)
#      return img

# def plot_simplicial_complex(dataframe,V):

#     #city centroids
#     city_coordinates = {city.sortedID: np.array((city.geometry.centroid.x, city.geometry.centroid.y)) for _, city in dataframe.iterrows()}

#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_axis_off() 

#     # Plot the "wyoming_svi" DataFrame
#     dataframe.plot(ax=ax, edgecolor='black', linewidth=0.3, color="white")

#     # Plot the centroid of the large square with values
#     # for i, row in dataframe.iterrows():
#     #     centroid = row['geometry'].centroid
#     #     # text_to_display = f"FIPS: {row['FIPS']}\nFilteration: {row['EP_SNGPNT']}"
#     #     plt.text(centroid.x, centroid.y, str(row['FIPS']), fontsize=8, ha='center', color="black")
#     #     # plt.text(centroid.x, centroid.y, text_to_display, fontsize=10, ha='center', color="black")

#     for edge_or_traingle in V:

        
#         if len(edge_or_traingle) == 2:
#             # Plot an edge
#             ax.plot(*zip(*[city_coordinates[vertex] for vertex in edge_or_traingle]), color='red', linewidth=1)
#             # img = fig2img(fig)
#             # list_gif.append(img)
#         elif len(edge_or_traingle) == 3:
#             # Plot a triangle
#             ax.add_patch(plt.Polygon([city_coordinates[vertex] for vertex in edge_or_traingle], color='green', alpha=0.2))
#             # img = fig2img(fig)
#             # list_gif.append(img)
#     # plt.show()
#     plt.close()


# for state in states:
#     svi_od = gpd.read_file(f'/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/SVI2018 census tracts with death rate HepVu-5 classes/{state}/{state}.shp')

#     # if value equals -999, replace with 0 in selected_variablesWy_
#     svi_od[selected_variables_tn[0]] = svi_od[selected_variables_tn[0]].replace(-999, 0)
#     svi_od[selected_variables_tn[1]] = svi_od[selected_variables_tn[1]].replace(-999, 0)
#     svi_od[selected_variables_tn[2]] = svi_od[selected_variables_tn[2]].replace(-999, 0)
#     svi_od[selected_variables_tn[3]] = svi_od[selected_variables_tn[3]].replace(-999, 0)
#     svi_od[selected_variables_tn[4]] = svi_od[selected_variables_tn[4]].replace(-999, 0)

#     # calculate the 50th and 75th and 90th percentile for each variable
#     variable_1_percentile_50 = svi_od[selected_variables_tn[0]].quantile(0.5)
#     variable_1_percentile_75 = svi_od[selected_variables_tn[0]].quantile(0.75)
#     variable_1_percentile_90 = svi_od[selected_variables_tn[0]].quantile(0.9)

#     variable_2_percentile_50 = svi_od[selected_variables_tn[1]].quantile(0.5)
#     variable_2_percentile_75 = svi_od[selected_variables_tn[1]].quantile(0.75)
#     variable_2_percentile_90 = svi_od[selected_variables_tn[1]].quantile(0.9)

#     variable_3_percentile_50 = svi_od[selected_variables_tn[2]].quantile(0.5)
#     variable_3_percentile_75 = svi_od[selected_variables_tn[2]].quantile(0.75)
#     variable_3_percentile_90 = svi_od[selected_variables_tn[2]].quantile(0.9)

#     variable_4_percentile_50 = svi_od[selected_variables_tn[3]].quantile(0.5)
#     variable_4_percentile_75 = svi_od[selected_variables_tn[3]].quantile(0.75)
#     variable_4_percentile_90 = svi_od[selected_variables_tn[3]].quantile(0.9)

#     variable_5_percentile_50 = svi_od[selected_variables_tn[4]].quantile(0.5)
#     variable_5_percentile_75 = svi_od[selected_variables_tn[4]].quantile(0.75)
#     variable_5_percentile_90 = svi_od[selected_variables_tn[4]].quantile(0.9)

#     tn_filtered = svi_od[selected_variables_tn_with_geo]

#     #reset index
#     tn_filtered = tn_filtered.reset_index(drop=True)

#     # get the uniques fips codes
#     fips = tn_filtered['FIPS'].unique()
    
#     #multiple

#     selected_variables_and_threshold = {selected_variables_tn[0]: variable_1_percentile_50, selected_variables_tn[1]: variable_2_percentile_50, selected_variables_tn[2]: variable_3_percentile_50, selected_variables_tn[3]: variable_4_percentile_50, selected_variables_tn[4]: variable_5_percentile_50}
    
#     # create a empty dictionary
#     edges_and_traingles_for_each_variable_below_50th_percentile = {}

#     for variable_name, threshold in selected_variables_and_threshold.items():

#         # Sorting based on the variable and selecting only the FIPS and the variable columns is important
#         # Also we need to keep  the dataframe sorted based on the variable

#         df_one_variable = tn_filtered[['FIPS',variable_name, 'geometry']]

#         # # Sorting the DataFrame based on the 'rate' column
#         df_one_variable = df_one_variable.sort_values(by=variable_name)
#         df_one_variable['sortedID'] = range(len(df_one_variable))

#         # Convert the DataFrame to a GeoDataFrame
#         df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
#         df_one_variable.crs = "EPSG:3395"  # This is a commonly used projected CRS


#         # print(df_one_variable.head(100))

#         adjacencies_list,adjacent_counties_df,county_list = generate_adjacent_counties(df_one_variable,threshold,variable_name)

#         # create a dictionary adjacent_counties_df column county as key and column adjacent as value(to avoid NULL adjacencies error)
#         adjacent_counties_dict = dict(zip(adjacent_counties_df['county'],adjacent_counties_df['adjacent']))

#         # this take only counties that have adjacent counties
#         county_list = adjacent_counties_df['county'].tolist()

#         V = form_simplicial_complex(adjacent_counties_dict,county_list)

#         # This is a new feature that I added to the code. It creates a new list replace the sorted ID with the FIPS on the V list
#         # create a new list replace the sorted ID with the FIPS on the V list
#         V_FIPS = [[df_one_variable.iloc[x]['FIPS'] for x in i] for i in V]
        
#         #add V list to the edges_and_traingles_for_each_variable dictionary with the key as the variable name
#         edges_and_traingles_for_each_variable_below_50th_percentile[variable_name] = V_FIPS

#         # # # store the list of images for each variable
#         # # list_img = []

#         # # plot the simplicial complex
#         print(f"Plotting simplicial complex for {variable_name} variable at threshold {threshold}")
#         # plot_simplicial_complex(df_one_variable,V)

    
#     selected_variables_and_threshold = {selected_variables_tn[0]: variable_1_percentile_75, selected_variables_tn[1]: variable_2_percentile_75, selected_variables_tn[2]: variable_3_percentile_75, selected_variables_tn[3]: variable_4_percentile_75, selected_variables_tn[4]: variable_5_percentile_75}
#     # create a empty dictionary
#     edges_and_traingles_for_each_variable_below_75th_percentile = {}

#     for variable_name, threshold in selected_variables_and_threshold.items():

#         # Sorting based on the variable and selecting only the FIPS and the variable columns is important
#         # Also we need to keep  the dataframe sorted based on the variable

#         df_one_variable = tn_filtered[['FIPS',variable_name, 'geometry']]

#         # # Sorting the DataFrame based on the 'rate' column
#         df_one_variable = df_one_variable.sort_values(by=variable_name)
#         df_one_variable['sortedID'] = range(len(df_one_variable))

#         # Convert the DataFrame to a GeoDataFrame
#         df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
#         df_one_variable.crs = "EPSG:3395"  # This is a commonly used projected CRS

#         adjacencies_list,adjacent_counties_df,county_list = generate_adjacent_counties(df_one_variable,threshold,variable_name)

#         # create a dictionary adjacent_counties_df column county as key and column adjacent as value(to avoid NULL adjacencies error)
#         adjacent_counties_dict = dict(zip(adjacent_counties_df['county'],adjacent_counties_df['adjacent']))

#         # this take only counties that have adjacent counties
#         county_list = adjacent_counties_df['county'].tolist()

#         V = form_simplicial_complex(adjacent_counties_dict,county_list)

#         # This is a new feature that I added to the code. It creates a new list replace the sorted ID with the FIPS on the V list
#         # create a new list replace the sorted ID with the FIPS on the V list
#         V_FIPS = [[df_one_variable.iloc[x]['FIPS'] for x in i] for i in V]

#         #add V list to the edges_and_traingles_for_each_variable dictionary with the key as the variable name
#         edges_and_traingles_for_each_variable_below_75th_percentile[variable_name] = V_FIPS

#         # # # store the list of images for each variable
#         # # list_img = []

#         # # plot the simplicial complex
#         print(f"Plotting simplicial complex for {variable_name} variable at threshold {threshold}")
#         # plot_simplicial_complex(df_one_variable,V)

#     selected_variables_and_threshold = {selected_variables_tn[0]: variable_1_percentile_90, selected_variables_tn[1]: variable_2_percentile_90, selected_variables_tn[2]: variable_3_percentile_90, selected_variables_tn[3]: variable_4_percentile_90, selected_variables_tn[4]: variable_5_percentile_90}
#     # create a empty dictionary
#     edges_and_traingles_for_each_variable_below_90th_percentile = {}

#     for variable_name, threshold in selected_variables_and_threshold.items():

#         # Sorting based on the variable and selecting only the FIPS and the variable columns is important
#         # Also we need to keep  the dataframe sorted based on the variable

#         df_one_variable = tn_filtered[['FIPS',variable_name, 'geometry']]

#         # # Sorting the DataFrame based on the 'rate' column
#         df_one_variable = df_one_variable.sort_values(by=variable_name)
#         df_one_variable['sortedID'] = range(len(df_one_variable))

#         # Convert the DataFrame to a GeoDataFrame
#         df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
#         df_one_variable.crs = "EPSG:3395"  # This is a commonly used projected CRS

#         adjacencies_list,adjacent_counties_df,county_list = generate_adjacent_counties(df_one_variable,threshold,variable_name)

#         # create a dictionary adjacent_counties_df column county as key and column adjacent as value(to avoid NULL adjacencies error)
#         adjacent_counties_dict = dict(zip(adjacent_counties_df['county'],adjacent_counties_df['adjacent']))

#         # this take only counties that have adjacent counties
#         county_list = adjacent_counties_df['county'].tolist()

#         V = form_simplicial_complex(adjacent_counties_dict,county_list)

#         # This is a new feature that I added to the code. It creates a new list replace the sorted ID with the FIPS on the V list
#         # create a new list replace the sorted ID with the FIPS on the V list
#         V_FIPS = [[df_one_variable.iloc[x]['FIPS'] for x in i] for i in V]

#         #add V list to the edges_and_traingles_for_each_variable dictionary with the key as the variable name
#         edges_and_traingles_for_each_variable_below_90th_percentile[variable_name] = V_FIPS

#         # # # store the list of images for each variable
#         # # list_img = []

#         # # plot the simplicial complex
#         print(f"Plotting simplicial complex for {variable_name} variable at threshold {threshold}")
#         plot_simplicial_complex(df_one_variable,V)


#     selected_regions_variable_1 = []
#     selected_regions_variable_2 = []
#     selected_regions_variable_3 = []
#     selected_regions_variable_4 = []
#     selected_regions_variable_5 = []

#     # loop through the dictionary and for each variable create a list of edges
#     for variable_name, V_FIPS in edges_and_traingles_for_each_variable_below_90th_percentile.items():
#     # for variable_name, V_FIPS in edges_and_traingles_for_each_variable_below_mean_plus_1sd.items():
#         for set in V_FIPS:
#             if len(set) == 2 or len(set) == 3:
#                 # if variable is EP_DISABL
#                 if variable_name == selected_variables_tn[0]:
#                     #check if the edge(both values) is not already in the list
#                     for vertice in set:
#                         if vertice not in selected_regions_variable_1:
#                             selected_regions_variable_1.append(vertice)
#                 elif variable_name == selected_variables_tn[1]:
#                     for vertice in set:
#                         if vertice not in selected_regions_variable_2:
#                             selected_regions_variable_2.append(vertice)
#                 elif variable_name == selected_variables_tn[2]:
#                     for vertice in set:
#                         if vertice not in selected_regions_variable_3:
#                             selected_regions_variable_3.append(vertice)
#                 elif variable_name == selected_variables_tn[3]:
#                     for vertice in set:
#                         if vertice not in selected_regions_variable_4:
#                             selected_regions_variable_4.append(vertice)
#                 elif variable_name == selected_variables_tn[4]:
#                     for vertice in set:
#                         if vertice not in selected_regions_variable_5:
#                             selected_regions_variable_5.append(vertice)
    
    
#     # this set of variables needed - specially STCNTY to identify the county
#     selected_variables_tn_with_censusinfo = ['FIPS','STCNTY','EP_DISABL', 'EP_NOHSDP', 'EP_PCI', 'EP_MOBILE', 'EP_POV','NOD_Rate','geometry']

#     # get a filtered dataframe with the selected regions
#     filtered_df_census = svi_od[selected_variables_tn_with_censusinfo]

#     # get the unique counties
#     unique_counties = svi_od['STCNTY'].unique()

#     # loop through the unique counties and get the number of rows for each county
#     for county in unique_counties:
#         print(f"County: {county}")

#         # create a temp_df with the selected county
#         temp_census = filtered_df_census[filtered_df_census['STCNTY'] == county]

#         # filter the filtered_df_census dataframe to only include the selected census for each selected variable
#         # FIPS includes the census info
#         variable_1_selected_census = temp_census[temp_census['FIPS'].isin(selected_regions_variable_1)][['FIPS',selected_variables_tn[0],'geometry']]
#         variable_2_selected_census = temp_census[temp_census['FIPS'].isin(selected_regions_variable_2)][['FIPS',selected_variables_tn[1],'geometry']]
#         variable_3_selected_census = temp_census[temp_census['FIPS'].isin(selected_regions_variable_3)][['FIPS',selected_variables_tn[2],'geometry']]
#         variable_4_selected_census = temp_census[temp_census['FIPS'].isin(selected_regions_variable_4)][['FIPS',selected_variables_tn[3],'geometry']]
#         variable_5_selected_census = temp_census[temp_census['FIPS'].isin(selected_regions_variable_5)][['FIPS',selected_variables_tn[4],'geometry']]

#         # create a new column in df that contains the x and y coordinates of the centroid of each polygon
#         variable_1_selected_census['coords'] = variable_1_selected_census['geometry'].apply(lambda x: x.representative_point().coords[:])
#         variable_1_selected_census['coords'] = [coords[0] for coords in variable_1_selected_census['coords']]

#         variable_2_selected_census['coords'] = variable_2_selected_census['geometry'].apply(lambda x: x.representative_point().coords[:])
#         variable_2_selected_census['coords'] = [coords[0] for coords in variable_2_selected_census['coords']]

#         variable_3_selected_census['coords'] = variable_3_selected_census['geometry'].apply(lambda x: x.representative_point().coords[:])
#         variable_3_selected_census['coords'] = [coords[0] for coords in variable_3_selected_census['coords']]

#         variable_4_selected_census['coords'] = variable_4_selected_census['geometry'].apply(lambda x: x.representative_point().coords[:])
#         variable_4_selected_census['coords'] = [coords[0] for coords in variable_4_selected_census['coords']]

#         variable_5_selected_census['coords'] = variable_5_selected_census['geometry'].apply(lambda x: x.representative_point().coords[:])
#         variable_5_selected_census['coords'] = [coords[0] for coords in variable_5_selected_census['coords']]


#         # create a dictionary with variable name as key and the data as value for all the selected regions
#         selected_coordinates_dic = {selected_variables_tn[0]: variable_1_selected_census, selected_variables_tn[1]: variable_2_selected_census, selected_variables_tn[2]: variable_3_selected_census, selected_variables_tn[3]: variable_4_selected_census, selected_variables_tn[4]: variable_5_selected_census}


#         # save the dictionary to a pickle file
#         with open(f'./results/selected coordinates for each county - percentiles(below 90th)/selected_coordinates_dic_{county}.pkl', 'wb') as f:
#             pickle.dump(selected_coordinates_dic, f)
            
