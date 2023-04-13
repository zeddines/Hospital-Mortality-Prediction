#!/user/bin/env python3
# -*- coding: utf-8 -*-

##############
# Summary
##############

# Script Author: Tanya
# Contributors: Frank, Novin, Tanya

"""
Creates new heuristically informed features to replace the less 
informative features.
"""


##############
# Imports
##############
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
import preprocessing as pp
from statsmodels.stats.outliers_influence import variance_inflation_factor



##############
# Functions
##############

# Author: Frank
# Dictionary of healthy ranges
def health_dictionary():
    """ Using medical heuristics, define feature healthy ranges
    
    Returns:
        - Good_range (dict): dictionary with structure
            [male/female][feature_name]: (min, max)
    """
    Good_range = {"M": dict(), "F": dict()}
    # Initialize the good range values for male patients
    Good_range['M']['age'] = (30, 35)
    Good_range['M']['BMI'] = (18.5, 24.9)
    Good_range['M']["Systolic blood pressure"] = (90.0, 120.0)
    Good_range['M']['Diastolic blood pressure'] = (60, 80)
    Good_range['M']["Respiratory rate"] = (12.0, 20.0)
    Good_range['M']["temperature"] = (36.5, 37.5)
    Good_range['M']["SP O2"] = (95.0, 100.0)
    Good_range['M']["Urine output"] = (500.0, 2000.0)
    Good_range['M']["Hematocrit"] = (38.8, 50)
    Good_range['M']["Heart rate"] = (60.0, 100.0)
    Good_range['M']["Diastolic blood pressure"] = (60.0, 80.0)
    Good_range['M']["RBC"] = (4.7, 6.1)
    Good_range['M']["MCH"] = (27.0, 33.0)
    Good_range['M']["MCHC"] = (33, 36.0)
    Good_range['M']["MCV"] = (80.0, 100.0)
    Good_range['M']["RDW"] = (11.5, 14.5)
    Good_range['M']["Leucocyte"] = (4, 11)
    Good_range['M']["Platelets"] = (150, 450)
    Good_range['M']["Neutrophils"] = (18.0, 75.0)
    Good_range['M']["Basophils"] = (0.0, 0.2)
    Good_range['M']["Lymphocytes"] = (10.0, 48.0)
    Good_range['M']["PT"] = (11.0, 13.5)
    Good_range['M']["INR"] = (0.8, 1.2)
    Good_range['M']["Creatine kinase"] = (39, 308)
    Good_range['M']["Creatinine"] = (0.74, 1.35)
    Good_range['M']["Urea nitrogen"] = (6.0, 20.0)
    Good_range['M']["Glucose"] = (70, 100)
    Good_range['M']["Blood potassium"] = (3.6, 5.2)
    Good_range['M']["Blood sodium"] = (135.0, 145.0)
    Good_range['M']["Blood calcium"] = (8.6, 10.3)
    Good_range['M']["Chloride"] = (96.0, 106.0)
    Good_range['M']["Anion gap"] = (3.0, 11.0)
    Good_range['M']["Magnesium ion"] = (1.7, 2.2)
    Good_range['M']["pH"] = (7.35, 7.45)
    Good_range['M']["Bicarbonate"] = (22.0, 29.0)
    Good_range['M']["Lactic acid"] = (0.45, 1.98)
    Good_range['M']["PCO2"] = (35.0, 45.0)
    Good_range['M']["EF"] = (50, 70)
    # Initialize the good range values for female patients
    Good_range['F']['BMI'] = (18.5, 24.9)
    Good_range['F']['age'] = (23, 30)
    Good_range['F']["Systolic blood pressure"] = (90.0, 120.0)
    Good_range['F']['Diastolic blood pressure'] = (60, 80)
    Good_range['F']["Respiratory rate"] = (12.0, 20.0)
    Good_range['F']["temperature"] = (36.5, 37.5)
    Good_range['F']["SP O2"] = (95.0, 100.0)
    Good_range['F']["Urine output"] = (500.0, 2000.0)
    Good_range['F']["Hematocrit"] = (34.9, 44.5)
    Good_range['F']["Heart rate"] = (60.0, 100.0)
    Good_range['F']["Diastolic blood pressure"] = (60.0, 80.0)
    Good_range['F']["RBC"] = (4.2, 5.4)
    Good_range['F']["MCH"] = (27.0, 33.0)
    Good_range['F']["MCHC"] = (33, 36.0)
    Good_range['F']["MCV"] = (80.0, 100.0)
    Good_range['F']["RDW"] = (11.5, 14.5)
    Good_range['F']["Leucocyte"] = (4, 11)
    Good_range['F']["Platelets"] = (150, 450)
    Good_range['F']["Neutrophils"] = (18.0, 75.0)
    Good_range['F']["Basophils"] = (0, 0.2)
    Good_range['F']["Lymphocytes"] = (10.0, 48.0)
    Good_range['F']["PT"] = (11.0, 13.5)
    Good_range['F']["INR"] = (0.8, 1.2)
    Good_range['F']["Creatine kinase"] = (26, 192)
    Good_range['F']["Creatinine"] = (0.59, 1.04)
    Good_range['F']["Urea nitrogen"] = (6.0, 20.0)
    Good_range['F']["Glucose"] = (70, 100)
    Good_range['F']["Blood potassium"] = (3.6, 5.2)
    Good_range['F']["Blood sodium"] = (135.0, 145.0)
    Good_range['F']["Blood calcium"] = (8.6, 10.3)
    Good_range['F']["Chloride"] = (96.0, 106.0)
    Good_range['F']["Anion gap"] = (3.0, 11.0)
    Good_range['F']["Magnesium ion"] = (1.7, 2.2)
    Good_range['F']["pH"] = (7.35, 7.45)
    Good_range['F']["Bicarbonate"] = (22.0, 29.0)
    Good_range['F']["Lactic acid"] = (0.45, 1.98)
    Good_range['F']["PCO2"] = (35.0, 45.0)
    Good_range['F']["EF"] = (50, 70)
    # return the newly created dictionary
    return Good_range



# Author: Frank
# Define a function to transform values based on given parameters
def more_extrem(x, lower,  upper, upper_index, lower_index):
    """
    calculates a more extreme value if outside healthy range

    Parameters:
        - x(float): current value
        - lower(float): heuristic lower range of feature
        - upper(float): heiristic upper range of feature
        - upper_index(float): extreme upper value multiplier
        - lower_index(float): extreme lower value multiplier

    Returns:
        - (float) more exteme value
    """
    average = (lower + upper) / 2
    if x >= lower and upper >= x:
        if x < average:
            return average + (average - x) 
        return x
    elif x > upper:
        return upper + upper_index * (x - upper)
    else:
        return (lower - x) * lower_index + upper



# Author: Frank
# Define a function to count the number of features outside 
# the good range
def num_outside_range(df, patient_data, gender, Good_range):
    """ counts number of features outside healthy range

    Parameters:
        - df (pandas Dataframe): Features and target
        - patient_data (vector): Feature instance row
        - gender (str): M/F
        - Good_range (dict): dictionary of features 
          high low health range
    
    Returns (int): count of features outside healthy range
    """
    num_outside = 0
    for i in range(patient_data.shape[0]):
        feature = df.columns[i]
        if feature == 'gendera' or feature == 'age':
            continue
        val = patient_data[i]
        if gender == 'M':
            lower, upper = Good_range['M'][feature]
        else:
            lower, upper = Good_range['F'][feature]
        if val < lower or val > upper:
            num_outside += 1
    return num_outside



# Author: Frank
# Define a function to process the age column based on gender
def age_processing(df, upper_index, lower_index, Good_range):
    """ Augment df age column w.r.t gender

    Parameters:
        - df (pandas Dataframe): Features and target
        - upper_index (float): Feature upper index
        - lower_index (float): Feature lower index
        - Good_range (dict): dictionary of features 
          high low health range
    """
    for index, row in df.iterrows():
        gender = row['gendera']
        if gender == 1:
            # if gender is M
            df.at[index, 'age'] = more_extrem(
                row['age']
                , Good_range['M']['age'][0]
                , Good_range['M']['age'][1]
                , upper_index, lower_index)
        else:
            # if gender is F
            df.at[index, 'age'] = more_extrem(
                row['age']
                , Good_range['F']['age'][0]
                , Good_range['F']['age'][1]
                , upper_index, lower_index)



# Author: Frank
# Define a function to generate a list of health indicators
def count_unhealthy_indicator(df, Good_range):
    """ Returns a list (for each feature) of how many out of healthy
    range feture values
    
    Parameters:
        - df (Dataframe): original dataFrame
        - Good_range (dict): dictionary of healthy ranges"""
    howmany = []
    for i in range(len(df)):
        row = df.iloc[i]
        gender = 'M' if row['gendera'] == 1 else 'F'
        num_outside_range = 0
        for col in df.columns:
            if col in Good_range[gender]:
                lower, upper = Good_range[gender][col]
                if row[col] < lower or row[col] > upper:
                    num_outside_range += 1
        howmany.append(num_outside_range)
    return howmany



# Author: Frank
# Define a function to check if a column has binary values
def is_binary(column):
    """returns true if column values are only 0 or 1."""
    return set(column.unique()).issubset({0, 1})



# Author: Frank
# Define a function to generate a correlation list
def correlatioin_list_generate(df, Good_range, y):
    """ calulates 'corr' and 'r2' for feature correlation dictionary

    Parameters:
        - df (Dataframe): original dataFrame
        - Good_range (dict): dictionary of healthy ranges
        - y (pandas Series): target, column 'outcome'
    
    Returns:
        - correlation (dict)
    """
    for i in range(len(df)):
        row = df.iloc[i]
        gender = 'M' if row['gendera'] == 1 else 'F'
        num_outside_range = 0
        for col in df.columns:
            if col in Good_range[gender]:
                df.at[i, col] = more_extrem(df.at[i, col]
                                        , Good_range[gender][col][0]
                                        , Good_range[gender][col][1]
                                        , 1.9, 14.9)
    correlation = {col: {'corr': None, 'r2': None} for col in df.columns}
    y = y
    for col in df.columns:
        X_col = df[[col]]
        # print(X_col)
        model = LinearRegression()
        model.fit(X_col, y)
        corr = np.corrcoef(X_col.T, y)[0, 1]
        r2 = model.score(X_col, y)
        correlation[col]['corr'] = corr
        correlation[col]['r2'] = r2
    return correlation



# Author: Frank
# Create a new field 'z-score sum' to indicate patient overall health
def populate_info_dict(df, Good_range):
    """ Calculates the mean and std. deviation for each feature

    Then using feature mean, std.Dev and current value, 
    calculates z-score sum.
    
    Parameters:
        - df (Dataframe): original dataFrame
        - Good_range (dict): dictionary of healthy ranges
    """
    information = {col: {'corr': None, 'r2': None} for col in df.columns}
    for column_name, column_data in df.iteritems():
        if column_name in Good_range['M'].keys():
            information[column_name]["mean"] = \
                                (Good_range['M'][column_name][0] 
                                + Good_range["M"][column_name][1] 
                                + Good_range['F'][column_name][0] 
                                + Good_range["F"][column_name][1]) / 4
            information[column_name]["SD"] = df[column_name].std()

    df_z = df.copy()
    for i in range(len(df_z)):
        for col in df_z.columns:
            if col in Good_range['M'].keys():
                df_z.at[i, col] = (df_z.at[i, col] - \
                                   information[col]["mean"]) \
                                    / information[col]["SD"]
    
    df_z_weighted_by_correlation = df_z.copy()
    df_z['z-score sum'] = df_z.sum(axis=1)
    return df_z



# Author: Novin
# Adding new features based on heuristics and sold features, then 
# dropping the old features
def binned_feature(df_updated):
    """
    Binning certain features to improve the feature quality
    
    Parameters:
        - df_updated (pandas Dataframe): updated feature set
        
    Returns:
        - df_updated (pandas Dataframe): Created binned versions of 
        the features below and remove the original feature
    """
    # Redpiratory rate is in a healthy range between 12-18. 
    # The data shows that the normal death rates for 12-22
    # Created a binary feature where 12-22 = 0 and everywhere else 
    # (too high or too low) its 1.
    df_updated['Respiratory rate binned'] = \
      df_updated['Respiratory rate'].apply(
        lambda x: 0 if 12 <= x <= 22 else 1)
    df_updated = df_updated.drop(['Respiratory rate'], axis=1)

    # Healthy temperature is in the range of 36 - 37.7
    # A binned feature for temp
    df_updated['temperature binned'] = df_updated['temperature']. \
      apply(lambda temp: 0 if 36 <= temp <= 37.7 else 1)
    df_updated = df_updated.drop(['temperature'], axis=1)

    # Healthy SP O2 range is 95% - 100% 
    # the data shows low death rates for 94 - 100% 
    # There for this range is used for the binned feature
    df_updated['SP O2 binned'] = df_updated['SP O2'].apply(
        lambda sp: 0 if 94 <= sp <= 100 else 1)
    df_updated = df_updated.drop(['SP O2'], axis=1)

    df_updated['hematocrit binned'] = df_updated.apply(
        lambda x: 0 if (25 <= x['hematocrit'] <= 40 
                        and x['gendera'] == 2) \
                    or (26 <= x['hematocrit'] <= 43 
                        and x['gendera'] == 1) else 1, axis=1)
    df_updated = df_updated.drop(['hematocrit'], axis=1)

    # The lowestest death rates are between 3 to 5
    # Binned feature in that range
    df_updated['RBC_updated'] = df_updated['RBC'].apply(
        lambda sp: 0 if 3 <= sp <= 5 else 1)
    df_updated = df_updated.drop(['RBC'], axis=1)

    # Healthy MCH range is 26 - 33
    # The lowest death rates are in this range 
    df_updated['MCH_updated'] = df_updated['MCH'].apply(
        lambda mch: 0 if 26 <= mch <= 33 else 1)
    df_updated = df_updated.drop(['MCH'], axis=1)

    # Healthy MCHC range is above 31
    # binned feature in that range
    df_updated['MCHC_updated'] = df_updated['MCHC'].apply(
        lambda mchc: 0 if 31 <= mchc else 1)
    df_updated = df_updated.drop(['MCHC'], axis=1)

    # Healthy  RDW is between 12 - 15 
    # binned feature in that range
    df_updated['RDW_updated'] = df_updated['RDW'].apply(
        lambda rdw: 0 if 12 <= rdw <= 15 else 1)
    df_updated = df_updated.drop(['RDW'], axis=1)

    # Healthy Lymphocyte is between 22-40
    # binned feature in that range
    df_updated['Lymphocyte_updated'] = df_updated['Lymphocyte']. \
      apply(lambda lym: 0 if 22 <= lym <= 40 else 1)
    df_updated = df_updated.drop(['Lymphocyte'], axis=1)

    # Healthy blood sodium is between 135-143
    # binned feature in that range
    df_updated['Blood_sodium_updated'] = \
      df_updated['Blood sodium'].apply(
        lambda bs: 0 if 135 <= bs <= 143 else 1)
    df_updated = df_updated.drop(['Blood sodium'], axis=1)

    # Healthy blood calcium is between 8.6 to 10.3
    # binned feature in that range
    df_updated['Blood_calcium_updated'] = \
      df_updated['Blood calcium'].apply(
        lambda bc: 0 if 8.6 <= bc <= 10.3 else 1)
    df_updated = df_updated.drop(['Blood calcium'], axis=1)

    # Healthy chloride is between 96 to 106
    # binned feature in that range
    df_updated['Chloride_updated'] = df_updated['Chloride'].apply(
        lambda chlo: 0 if 96 <= chlo <= 106 else 1)
    df_updated = df_updated.drop(['Chloride'], axis=1)

    # Healthy Magnesium ion is between 1.7 to 2.2
    # binned feature in that range
    df_updated['Magnesium_ion_updated'] = \
      df_updated['Magnesium ion'].apply(
        lambda mag_ion: 0 if 1.7 <= mag_ion <= 2.2 else 1)
    df_updated = df_updated.drop(['Magnesium ion'], axis=1)

    return df_updated



# Author: Novin
def normalize_continuous_columns(df):
    """
    Normalize continuous columns in dataset using Min-Max scaling.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized continuous columns.
    """
    # Find continuous columns with more than two unique values
    continuous_cols = []
    for col in df.columns:
        if df[col].nunique() > 2:
            continuous_cols.append(col)

    # Normalize continuous columns using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    for col in continuous_cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df


# Author: Novin
def smote_sampling(X, y, ratio=1.0, random_state=None):
    """
    Uses sampling to balance the classes and creates synthetic data 
    using SMOTE.

    Parameters:
        X: All the finalised features 
        y: Outcome column data
        ratio: sampling ration

    Returns:
        sampled_df: Returns a sampled dataset for training
    """
    # Instantiate the RandomOverSampler
    ros = SMOTE(sampling_strategy=ratio, random_state=random_state)

    # Perform oversampling
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Combine the oversampled feature and target data into a 
    # DataFrame
    sampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    sampled_df['outcome'] = y_resampled

    return sampled_df

# Author: Bertram
# Get Vif for all features in X
def get_vif(X):
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(X.values,i)\
                 for i in range(0, X.shape[1])]
    vif['Feature'] = X.columns
    return vif

# Author: Bertram
# Iterative method to eliminate feature that have the highest VIF
# Ends when all VIF are below VIF_thresshold
def get_vif_drop(X, cat_features = [], VIF_threshold = 10):
    """
    Parameters:
        X  (pandas Dataframe): updated feature set
        cat_features (list): list of categorical feature
                             excluded from VIF filtering
        VIF_threshold: VIF threshold, the method ends
                       when all numerical feature have
                       VIF values less than this threshold
    Returns:
        drop_feature_vif (list): list of feature that should
                                   be dropped according to VIF
                                   filtering
    """
    drop_feature_vif = []
    X_test_updated_cont = X.drop(cat_features, axis = 1)
    while (True):
        vif = get_vif(X_test_updated_cont)
        mv = max(vif['VIF'])
        if (mv < VIF_threshold):
            break
        feature_tbr = vif.loc[vif['VIF'] == mv, "Feature"]
        feature_tbr = feature_tbr.values[0]
        drop_feature_vif.append(feature_tbr)
        l = []
        for i in X_test_updated_cont.columns.values:
            if (i == feature_tbr):
                l.append(False)
            else:
                l.append(True)
        X_test_updated_cont = X_test_updated_cont.loc[:, l]
    return drop_feature_vif



# Author: Tanya
# Given a dataframe with pre-processing complete, engineer features 
# to include heuristic knowledge
def model_ready_datasets(df_pp:pd.DataFrame):
    """ Applies feature engineering to add heuristics to features

    Parameters:
        df_pp (pd.DataFrame): DataFrame which has had pre-processing
        carried out.
    
    Returns:
        pandas Dataframe with newly added engineered features
        X_train (pandas Dataframe): training features
        X_test (pandas Dataframe): features reserved for testing
        y_train (pandas Dataframe): target labels for training
        y_test (pandas Dataframe): true target for testing metrics
    """
    # create the dictionary of healthy ranges
    Good_range = health_dictionary()
    # get the pre-processed data file
    df = pp.initial_data_cleansing()
    # Process the age column
    age_processing(df, 4.1, 7, Good_range)
    # Add two new fields 'howmany' and 'z-score sum'
    df['howmany'] = count_unhealthy_indicator(df, Good_range)
    df_z = populate_info_dict(df, Good_range)
    df['z-score sum'] = df_z.sum(axis=1)
    df = binned_feature(df)
    df = normalize_continuous_columns(df)
    # Sperate the features and the outcome/target
    y = df[['outcome']]
    cols_to_drop = ['outcome','gendera', 'NT-proBNP', 'INR'
                    ,'Chloride_updated', 'Diastolic blood pressure'
                    ,'hematocrit binned','SP O2 binned'
                    ,'Magnesium_ion_updated','atrialfibrillation']
    X = df.drop(cols_to_drop, axis=1)
    # Drop features based on VIF  
    VIF_threshold = 7
    feature_tbd = get_vif_drop(X, [ "hypertensive", "CHD with no MI",\
                                   'Respiratory rate binned',\
                                    'temperature binned', 'RBC_updated',\
                                    'MCH_updated',\
                                    'MCHC_updated','RDW_updated',\
                                    'Lymphocyte_updated',\
                                    'Blood_sodium_updated',\
                                    'Blood_calcium_updated'], VIF_threshold)
    # Additional feautures dropped, based on VIF_threshold = 7
    # 'Neutrophils', 'PH', 'Anion gap', 'Bicarbonate', 'MCV', 
    # 'howmany', 'Blood potassium', 'age', 'heart rate', 'Systolic blood pressure']
    X = X.drop(feature_tbd, axis = 1)
    # Split data into development (90%) and test (10%) sets
    X_dev, X_test, y_dev, y_test = \
      train_test_split(X, y, test_size=0.1, random_state=42)
    # Only sample the training dataset
    df = smote_sampling(X_dev, y_dev, ratio=0.18, random_state=18)
    # return training and testing data sets
    y_train = df['outcome']
    X_train = df.drop(['outcome'], axis=1)
    return X_train, X_test, y_train, y_test
    


##############
# Main
##############

# Author: Tanya
def main():
    """ function which controls code called """
    pp_df = pp.initial_data_cleansing()
    X_train, X_test, y_train, y_test = model_ready_datasets(pp_df)
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_test.shape: ', y_test.shape)



# fist line of code executed in preprocessing.py if run as script
if __name__ == '__main__':
		main()

