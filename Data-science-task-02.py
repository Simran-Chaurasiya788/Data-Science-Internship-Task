import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# Display first few rows
print(df.head())

# Basic info
print(df.info())

# Summary statistics
print(df.describe())

# Missing values
print(df.isnull().sum())

# Countplot - Survived
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()