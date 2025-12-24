# 3:46 PM
import pandas as pd
# IMPORT THE NEW MODEL!
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- FIX (ADDED AFTER 7. VISUALIZE) FOR MACOS PLOTTING STARTS HERE ---
import tkinter as tk
plt.switch_backend('TkAgg') 
# --- FIX ENFS HERE ---

# ... REST OF THE CODE ...
# 1. Load Data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Selecting Features (All of them)
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

# --- NEW CODE BLOCK FOR FEATURE IMPORTANCE STARTS HERE ---
# 1. Get the list of feature importances from the trained model
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)

# 2. Print the top 5 most important features
print("\n--- Feature Importances (Model's Insights) ---")
print(feature_importances.sort_values(ascending=False).head(5))

# 3. Plot the top 5 importances for visualization
feature_importances.sort_values(ascending=False).head(5).plot(kind='barh')
plt.title("Top 5 Feature Importances")
plt.xlabel("Importance Score")
# plt.show() 
# --- NEW CODE BLOCK ENDS HERE ---
# --- NEW CODE BLOCK FOR INTERACTIVE PREDICTION STARTS HERE ---

print("\n--- Interactive Scenario Analysis ---")

# THE COLUMN MODEL IS CRITICAL FOR THE MODEL!
# Order: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
print("Enter 8 values separated by commas in this order:")
print("MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude")

try:
    # GET ALL INPUTS AS A SINGLE STRING
    user_input_str = input("Enter 8 values: ")
    # Convert the comma-separated string into a list of floats
    # Note: The model expects the data as a 2D array: [[v1, v2, ...]]
    user_values = [[float(x.strip()) for x in user_input_str.split(',')]]
    
    # CHECK IF I HAVE EXACTLY 8 VALUES
    if len(user_values[0]) == 8:
        # Predict the price for the user's input
        new_pred = model.predict(user_values)
        print(f"\n✅ Predicted Median House Price: ${new_pred[0]*100000:,.2f}")
    else:
        print("Error: You must enter exactly 8 comma-separated values.")

except ValueError:
    print("Error: Please ensure all inputs are valid numbers.")

# --- NEW CODE BLOCK ENDS HERE ---
# 7. Visualize
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted (Random Forest Regressor)")
# plt.show()
