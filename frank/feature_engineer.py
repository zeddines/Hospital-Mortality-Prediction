import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


df = pd.read_csv('data01.csv')

not_serious_list = ["COPD", "Renal failure", "Hyperlipemia", "depression", "deficiencyanemias", "diabetes", "CHD with no MI", "atrialfibrillation", "hypertensive"]

Good_range = {"M": dict(), "F": dict()}
information = dict()

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

#count howmany unserious_diseases the patient has
def unserious_count(df, not_serious_list):
    unserious = []
    for i in range(len(df)):
        row = df.iloc[i]
        counter = 0
        for col in df.columns:
            if col in not_serious_list:
                counter = counter + df.iloc[i][col]
        unserious.append(counter)
    return unserious

# Define a function to transform values based on given parameters
def more_extrem(x, lower,  upper, upper_index, lower_index):
    average = (lower + upper) / 2
    if x >= lower and upper >= x:
        if x < average:
            return average + (average - x) 
        return x
    elif x > upper:
        return upper + upper_index * (x - upper)
    else:
        return (lower - x) * lower_index + upper
    
# Define a function to count the number of features outside the good range
def num_outside_range(patient_data, gender, Good_range):
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

# Define a function to process the age column based on gender
def age_processing(df, upper_index, lower_index):
    for index, row in df.iterrows():
        gender = row['gendera']
        if gender == 1:
            # if gender is M
            df.at[index, 'age'] = more_extrem(row['age'], Good_range['M']['age'][0], Good_range['M']['age'][1], upper_index, lower_index)
        else:
            # if gender is F
            df.at[index, 'age'] = more_extrem(row['age'], Good_range['F']['age'][0], Good_range['F']['age'][1], upper_index, lower_index)

# Define a function to generate a list of health indicators
def count_unhealthy_indicator(df, Good_range):
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

# Define a function to check if a column has binary values
def is_binary(column):
    return set(column.unique()).issubset({0, 1})

# Define a function to generate a correlation list
def correlatioin_list_generate(df):
    for i in range(len(df)):
        row = df.iloc[i]
        gender = 'M' if row['gendera'] == 1 else 'F'
        num_outside_range = 0
        for col in df.columns:
            if col in Good_range[gender]:
                df.at[i, col] = more_extrem(df.at[i, col], Good_range[gender][col][0], Good_range[gender][col][1], 1.9, 14.9)

    correlation = {col: {'corr': None, 'r2': None} for col in df.columns}
    y = Y
    for col in df.columns:
        X_col = df[[col]]
        print(X_col)
        model = LinearRegression()
        model.fit(X_col, y)
        corr = np.corrcoef(X_col.T, y)[0, 1]
        r2 = model.score(X_col, y)
        correlation[col]['corr'] = corr
        correlation[col]['r2'] = r2
    return correlation

# Replace missing values (NA) with the mean of their respective columns
df.replace('NA', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Extract the target variable and drop unnecessary columns
Y = df.iloc[:, 2].to_numpy()
print(Y)
df = df.drop(df.columns[[0,1,2]], axis = 1)

# Process the age column
age_processing(df, 4.1, 7)

# Generate the correlation list and information dict
correlation = correlatioin_list_generate(df)
how_many = count_unhealthy_indicator(df, Good_range)
unserious_count = unserious_count(df, not_serious_list)
information = {col: {'corr': None, 'r2': None} for col in df.columns}


# Iterate through DataFrame columns and update information dictionary
for column_name, column_data in df.iteritems():
    if column_name in Good_range['M'].keys():
        information[column_name]["mean"] = (Good_range['M'][column_name][0] + Good_range["M"][column_name][1] + Good_range['F'][column_name][0] + Good_range["F"][column_name][1]) / 4
        information[column_name]["SD"] = df[column_name].std()

df_z = df.copy()
for i in range(len(df_z)):
    for col in df_z.columns:
        if col in Good_range['M'].keys():
            df_z.at[i, col] = (df_z.at[i, col] - information[col]["mean"])/information[col]["SD"]

df_z_weighted_by_correlation = df_z.copy()

df_z['z-score sum'] = df_z.sum(axis=1)

for col in df_z_weighted_by_correlation.columns:
    df_z_weighted_by_correlation[col] = df_z_weighted_by_correlation[col] * correlation[col]['corr'] * correlation[col]['r2']

df_z_weighted_by_correlation['z-score adjusted sum'] = df_z_weighted_by_correlation.sum(axis=1)

for i in range(len(df)):
    row = df.iloc[i]
    gender = 'M' if row['gendera'] == 1 else 'F'
    num_outside_range = 0
    for col in df.columns:
        if col in Good_range[gender]:
            df.at[i, col] = more_extrem(df.at[i, col], Good_range[gender][col][0], Good_range[gender][col][1], 1.9, 14.9)
    

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
df['howmany'] = how_many
df['unserious_counter'] = unserious_count
df['z-score sum'] = df_z.sum(axis=1)
df['z-score adjusted sum'] = df_z_weighted_by_correlation.sum(axis=1)
df.to_csv("haha.csv")
