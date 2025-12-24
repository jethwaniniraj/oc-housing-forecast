import pandas as pd
# Import the new model!
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Select Features (All of them)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model (The new model is here!)
# n_estimators=100 means it builds 100 individual decision trees to make the final prediction.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("--- Random Forest Evaluation ---")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 7. Visualize
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted (Random Forest Regressor)")
plt.show()
