import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'Kerala.csv' is in the same directory as this script
csv_file_path = 'Kerala.csv'
data = pd.read_csv(csv_file_path)

# Print column names
print("Columns in the DataFrame:")
print(data.columns.tolist())

# Check if 'ANNUAL RAINFALL' column exists
if 'ANNUAL RAINFALL' in data.columns:
    rainfall_column = 'ANNUAL RAINFALL'
elif 'ANNUALRAINFALL' in data.columns:
    rainfall_column = 'ANNUALRAINFALL'
else:
    raise ValueError("Could not find 'ANNUAL RAINFALL' or 'ANNUALRAINFALL' column in the CSV file.")

# Reshape X to be a 2D array
X = data[rainfall_column].values.reshape(-1, 1)
y = data['FLOODS'].replace({'YES': 1, 'NO': 0}).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\\nLogistic Regression Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("\\nClassification Report:")
print(report)


