import pandas as pd
import numpy as np
import math 
# Load the dataset
file_name = "flood.csv"
data = pd.read_csv(file_name)
data.insert(0, 'Column_of_Ones', 1)  # Add a column of ones for the intercept term

# Separate the features (X) and the target (y)
X = data.drop(columns=["FloodProbability"]).values
y = data["FloodProbability"].values

# Data Scaling (Standardization: mean=0, std=1) using NumPy
mean = np.mean(X[:, 1:21], axis=0)  # Compute mean for features (exclude Column_of_Ones)
std = np.std(X[:, 1:21], axis=0)    # Compute std for features (exclude Column_of_Ones)
X[:, 1:21] = (X[:, 1:21] - mean) / std  # Scale features (Column_of_Ones remains unchanged)

# Initialize parameters
theta = np.ones(X.shape[1])  # Initialize theta
alpha = 0.01  # Learning rate
m = X.shape[0]  # Number of training examples

# Gradient Descent with MAE
J = float('inf')
iteration = 0

while J > 1e-3:
    predictions = np.dot(X, theta)
    errors = predictions - y
    J = (1 / m) * np.sum(errors ** 2)  # Mean Absolute Error
    # Update theta
    theta -= alpha * (1/m) * np.dot(X.T, np.sign(errors))  # Use the sign of errors for update
    iteration += 1
    print(f"Iteration {iteration}: Cost = {round(J, 10)}")