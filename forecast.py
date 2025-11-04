import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the California housing dataset (mirrors OC market trends)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Select key features for prediction
X = df[["MedInc", "HouseAge", "AveRooms"]]
y = df["MedHouseVal"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate performance
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction (adjust features as needed)
example_features = [[9.0, 20, 6]]  # MedInc, HouseAge, AveRooms
print("Predicted Median Price (100k's):", model.predict(example_features))
