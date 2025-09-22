# Task 1: Data Cleaning & Preprocessing
# Dataset: Titanic (https://www.kaggle.com/datasets/yasserh/titanic-dataset)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Import dataset
df = pd.read_csv("train.csv")   
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# 2. Handle missing values
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# 3. Convert categorical variables into numerical
# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 4. Normalize / Standardize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 5. Detect and remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Example: Remove outliers in 'Fare'
print("\nShape before removing outliers (Fare):", df.shape)
df = remove_outliers(df, 'Fare')
print("Shape after removing outliers (Fare):", df.shape)

# Visualization (optional)
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare (after removing outliers)")
plt.show()

# 6. Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as 'titanic_cleaned.csv'")
