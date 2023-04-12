#!/user/bin/env python3
# -*- coding: utf-8 -*-

##############
# Summary
##############

# Script Author: Tanya
# Contributors: Novin, Frank, Tanya

"""Handles the pre-processing tasks for the Hospital Mortalility
prediction project
"""


##############
# Imports
##############

import pandas as pd



##############
# Functions
##############


# Author: Novin
def drop_initial_features(df:pd.DataFrame):
    """
    Drop the initial features based on trend and information value

    Converts 'outcome' to an int
    
    Parameters:
        - df (pandas DataFrame):  Input features 
        
    Returns:
        - df_updated (pandas DataFrame): unnecessary features dropped
    """
    # define the features to drop 
    feature_drop_list = ['group','ID','diabetes','deficiencyanemias'
                         , 'depression', 'Hyperlipemia'
                         , 'Renal failure', 'COPD'] 
    df_updated = df.drop(feature_drop_list, axis=1)

    # Convert outcome into 1 and 0 values
    df_updated['outcome'] = df_updated['outcome'].apply(
            lambda x: 0 if x < 0.5 else 1)
    
    return df_updated



# Author: Novin
# The origianl data file contained missing information, which is 
# filled with either the mean or median value from the same outcome.
def missing_imputation(df:pd.DataFrame):
    """
    Fills missing values with either median or mean based on the 
    skewness of the columns.
    
    Parameters:
        - df (pandas DataFrame): initial features removed 
        
    Returns:
        - filled_df (pandas DataFrame): missing values in continous 
        columns are replaced with mean or median
    """
    # Calculate skewness of each column
    skewness = df.skew()

    # Create a copy of the input DataFrame to avoid modifying the 
    # original DataFrame
    filled_df = df.copy()
    output_col = 'outcome'
    # Group the data by the output column
    groups = filled_df.groupby(output_col, group_keys=False)
    # Fill missing values with mean/median based on the output column
    for col in df.columns:
        if col != output_col:
            # If the skewness is greater than 0.5, not a normal 
            # distribution therefore, replace with median 
            # else replace with mean
            if skewness[col] > 0.5:
                filled_df[col] = groups[col].apply(
                      lambda x: x.fillna(x.median()))
            else:
                filled_df[col] = groups[col].apply(
                      lambda x: x.fillna(x.mean()))

    return filled_df



# Authors: Tanya, Novin
# Function for code called from outside of this script. Will read 
# original .csv data downloaded, and carry out data cleansing.
def initial_data_cleansing():
    """ carries out data cleansing and 'naa' updates 

    Imports a .csv file into a pandas Dataframe and then applies 
    various data cleansing routines.
    
    Returns:
        df_updated (pandas Dataframe): with data cleansing applied
    """
    # Use the pandas read_csv method to read the file
    df = pd.read_csv('data01.csv')
    df_updated = drop_initial_features(df)
    df_updated = missing_imputation(df_updated)
    return df_updated

      



##############
# Main
##############

def main():
    """ function which controls code called """
    print("pre-processing...")
    df = initial_data_cleansing()
    print(df.head)
    
    



# fist line of code exuted in preprocess.py if run as script
if __name__ == '__main__':
		main()

