import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import pickle as pickle
from pylab import *
import warnings
import os

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# imoporting SVI data for the entire US(county level) 
us_svi = gpd.read_file('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/SVI 2018 with HepVu census tracts/SVI2018_US_census_with_opioid_indicators.shp')

# get unique State Abbreviations to a list
states = us_svi['ST_ABBR'].unique()

# print('Number of states:', len(states))

# importing the overdose with county information
overdose_df = pd.read_excel('/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/HepVu_County_Opioid_Indicators_05DEC22.xlsx')

# convert the GEO ID to string
overdose_df['GEO ID'] = overdose_df['GEO ID'].astype(str)

# GEOID column id if any GEOID is less than 6 characters insert 0 at the beginning and make it 6 characters
overdose_df['GEO ID'] = overdose_df['GEO ID'].apply(lambda x: x.zfill(5))


# # loop through each state
# for state in states:
#     # create a folder for each state if it does not exist
#     os.makedirs(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/SVI2018 census tracts with death rate HepVu-5 classes/{state}", exist_ok=True)

# loop through each state and filter the census tracts
for state in states:
    print('Processing:', state)
    try:
        # filter the census tracts for each state
        state_svi = us_svi[us_svi['ST_ABBR'] == state]

        # filter by State Abbreviation	
        state_overdose = overdose_df[overdose_df['State Abbreviation'] == state]

        state_overdose['Narcotic Overdose Mortality Rate 2018'] = state_overdose['Narcotic Overdose Mortality Rate 2018'].astype(float)

        state_overdose['percentile'] = pd.qcut(state_overdose['Narcotic Overdose Mortality Rate 2018'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=['0', '1', '2', '3', '4'])

        # reset the index
        state_svi.reset_index(drop=True, inplace=True)

        # merge the two dataframes
        merged_df = pd.merge(state_svi, state_overdose[['GEO ID','percentile']], left_on='STCNTY', right_on='GEO ID', how='left')

        # drop GEO ID column
        merged_df.drop(columns=['GEO ID'], inplace=True)

        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')

        gdf['percentile'] = gdf['percentile'].astype(str)

        # Save the GeoDataFrame to a Shapefile
        gdf.to_file(f"/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/SVI2018 census tracts with death rate HepVu-5 classes/{state}/{state}.shp", driver='ESRI Shapefile')
    except Exception as e:
        print(f"Error processing {state}: {e}")
        continue  # Continue to the next iteration if an error occurs

    





    
