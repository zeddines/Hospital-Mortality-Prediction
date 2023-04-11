import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

df_origin = pd.read_csv('data01.csv')
df = df_origin.copy()

Good_range = {"M": dict(), "F": dict()}
Good_range['M']['age'] = (30, 35)


Good_range['F']['BMI'] = (18.5, 24.9)
Good_range['F']['age'] = (23, 30)


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



df = pd.read_csv('data01.csv')

df.replace('NA', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# define linear function
def linear_func(x, a, b):
    return a*x + b

# define exponential function
def exponential_func(x, a, b):
    return a*np.exp(b*x)

# define quadratic function
def quadratic_func(x, a, b, c):
    return a*x**2 + b*x + c

# load data
df = pd.read_csv('data01.csv')

# drop missing values
df.dropna(inplace=True)


age_processing(df, 1, 1)

# separate data into x and y
x_data = df['age'].values
y_data = df['outcome'].values

# fit linear model
popt, pcov = curve_fit(linear_func, x_data, y_data)
y_pred_linear = linear_func(x_data, *popt)
r2_linear = r2_score(y_data, y_pred_linear)

# fit exponential model
popt, pcov = curve_fit(exponential_func, x_data, y_data)
y_pred_exp = exponential_func(x_data, *popt)
r2_exp = r2_score(y_data, y_pred_exp)

# fit quadratic model
popt, pcov = curve_fit(quadratic_func, x_data, y_data)
y_pred_quad = quadratic_func(x_data, *popt)
r2_quad = r2_score(y_data, y_pred_quad)

# print R-squared values
print('Linear R-squared:', r2_linear)
print('Exponential R-squared:', r2_exp)
print('Quadratic R-squared:', r2_quad)


max = 0
best_x = 0
best_y = 0

for x in range (41,42):
    for y in range(0,400):
        df1 = df.copy()
        age_processing(df1, x/10, y/10)
        # separate data into x and y
        x_data = df1['age'].values
        y_data = df1['outcome'].values
        # Calculate correlation coefficient
        correlation_coef = np.corrcoef(x_data, y_data)[0, 1]

        # Calculate R-squared
        model = np.polyfit(x_data, y_data, 1)
        residuals = np.polyval(model, x_data) - y_data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if max < r_squared * correlation_coef:
            max = r_squared * correlation_coef
            best_x = x
            best_y = y
            print(f"-------{best_x}-------{best_y}----------{max}--------{correlation_coef}---------{r_squared}")

print(best_x)
print(best_y)
        

