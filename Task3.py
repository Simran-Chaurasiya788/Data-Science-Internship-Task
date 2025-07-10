# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load data
df = pd.read_csv("bank.csv", sep=';')
print(df.head())

# Step 3: Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Split features and target
X = df_encoded.drop('y_yes', axis=1)  # 'y' is target
y = df_encoded['y_yes']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 7: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))