import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D


# Load data into a DataFrame
df = pd.read_csv('data01.csv')

Good_range = {"M": dict(), "F": dict()}

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

def age_processing(df, upper_index, lower_index):
    for index, row in df.iterrows():
        gender = row['gendera']
        if gender == 1:
            # if gender is M
            df.at[index, 'age'] = more_extrem(row['age'], Good_range['M']['age'][0], Good_range['M']['age'][1], upper_index, lower_index)
        else:
            # if gender is F
            df.at[index, 'age'] = more_extrem(row['age'], Good_range['F']['age'][0], Good_range['F']['age'][1], upper_index, lower_index)


df.replace('NA', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)
age_processing(df, 411, 7)

for i in range(len(df)):
    row = df.iloc[i]
    gender = 'M' if row['gendera'] == 1 else 'F'
    num_outside_range = 0
    for col in df.columns:
        if col in Good_range[gender]:
            df.at[i, col] = more_extrem(df.at[i, col], Good_range[gender][col][0], Good_range[gender][col][1], 1.9, 14.9)

# Separate X and y variables
y = df['outcome']  # select the last column

df = df.drop(df.columns[[0,1,2]], axis = 1)

X = df.iloc[:, :-1] # select all columns except the last one

# Calculate the correlation and r-square between each X column and y
corr_values = []
r2_values = []
for col in X.columns:
    X_col = X[[col]]
    model = LinearRegression()
    model.fit(X_col, y)
    corr = np.corrcoef(X_col.T, y)[0, 1]
    r2 = model.score(X_col, y)
    corr_values.append(corr)
    r2_values.append(r2)



# Plot the correlation and r-square values
fig, ax = plt.subplots()
bars = ax.bar(X.columns, corr_values, label='Correlation')
ax.bar(X.columns, r2_values, label='R-square', alpha=0.5)
ax.set_xlabel('X columns', fontsize=12)
ax.set_ylabel('Correlation/R-square')
ax.set_title('Correlation and R-square values for X columns')
ax.legend()
ax.set_xticklabels(X.columns, fontsize=8, rotation=45, ha='right')  # adjust font size of x-axis labels



# Display the correlation values on top of the bars
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{corr_values[i]:.2f}", ha='center', va='bottom', fontsize=8)

plt.show()
fig.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')


# Calculate the correlation and r-square between each X column and y
corr_values = []
for col in X.columns:
    X_col = X[[col]]
    corr = np.corrcoef(X_col.T, y)[0, 1]
    corr_values.append(corr)

# Create a table of X column names and their corresponding correlation values
corr_table = pd.DataFrame({'X Column': X.columns, 'Correlation': corr_values})

# Plot the correlation table
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
ax.table(cellText=corr_table.values, colLabels=corr_table.columns, loc='center')
plt.show()
fig.savefig('correlation_table.png', dpi=300, bbox_inches='tight')
