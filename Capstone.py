import pandas as pd
import numpy as np


df = pd.read_csv(r'D:\VS Code Programs\NumPy Library\Projects\employee_data.csv')
# print(df.head())


# Firstly always check missing values in data-set
# print("Missing values in each column:")   
# print(df.isnull().sum())

# Replace 'inf' and '-inf' with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# Replace nan values from Salary column
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# Replace nan values from PerformanceScore
df['PerformanceScore'] = df['PerformanceScore'].fillna(df['PerformanceScore'].median())
df.fillna(df.mean(numeric_only=True), inplace=True)

# print(df.isna().sum())

# Remove duplicates 
df.drop_duplicates(inplace=True)

# Remove negative values
# np.where(condition, value_if_true, value_if_false):
df['Salary'] = np.where(df['Salary'] < 0, df['Salary'].mean(), df['Salary'])
df['ExperienceYears'] = np.where(df['ExperienceYears'] < 0, df['ExperienceYears'].mean(), df['ExperienceYears'])
# df['Salary'] = df['Salary'].fillna(df['Salary'].mean())


# applying standard deviation for outliers
salary_mean = df['Salary'].mean()
salary_std = df['Salary'].std()
lower_bound = salary_mean - (3 * salary_std)
upper_bound = salary_mean + (3 * salary_std)

df = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]



# Save cleaned data
print("Cleaned Employee Data safely!")
df.to_csv(r'D:\VS Code Programs\NumPy Library\Projects\Cleaned_Employee_data.csv', index=False)


       
