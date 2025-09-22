Task 1 - Data Cleaning & Preprocessing (AI & ML Internship)
📌 Objective

The goal of this task is to clean and preprocess raw data to make it suitable for machine learning models.
We are using the Titanic Dataset from Kaggle.

📂 Dataset

Titanic Dataset (Kaggle)

File used: train.csv

🛠️ Tools & Libraries

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

🔎 Steps Performed

Import Dataset

Loaded train.csv using Pandas.

Checked dataset shape, null values, and data types.

Handle Missing Values

Filled missing Age with median.

Filled missing Embarked with mode.

Dropped Cabin column (too many missing values).

Convert Categorical to Numerical

Applied One-Hot Encoding for categorical columns (Sex, Embarked).

Feature Scaling

Standardized numerical features (Age, Fare) using StandardScaler.

Outlier Detection & Removal

Used Boxplots and IQR method to detect/remove outliers in Fare.

Save Cleaned Dataset

Exported final dataset as titanic_cleaned.csv.

📊 Visualizations

Boxplots were used to detect outliers in numerical features (Fare).

📁 Files in Repository

task1_preprocessing.py → Python code for data cleaning & preprocessing.

train.csv → Original dataset (from Kaggle).

titanic_cleaned.csv → Cleaned and preprocessed dataset.

README.md → Documentation of the task.

📝 Interview Questions to Prepare

What are the different types of missing data (MCAR, MAR, MNAR)?

How do you handle categorical variables?

Difference between Normalization vs Standardization.

Methods to detect outliers (IQR, Z-score, boxplots).

Why is preprocessing important in ML?

One-hot encoding vs Label encoding.

How to handle data imbalance?

Can preprocessing affect model accuracy?
