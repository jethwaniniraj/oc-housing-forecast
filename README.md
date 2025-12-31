# 10:50 am Dec 31
import pandas as pd
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

# ----------------------------------------------------------------------------------
#  <<<<<<<<<< PASTE THE NEW, UPDATED BLOCK HERE >>>>>>>>>>
# ----------------------------------------------------------------------------------

# --- NEW CODE BLOCK FOR INTERACTIVE PREDICTION STARTS HERE ---

print("\n--- Interactive Scenario Analysis (Orange County ZIP-Aware) ---")

# --- ZIP CODE UTILITY FUNCTION ---
# This function maps Orange County ZIP codes to approximate Latitude and Longitude.
# The original dataset does not have ZIP codes, so this is a workaround using coordinates.
def get_zip_coords(zip_code):
    """Returns approximate (Latitude, Longitude) for a given Orange County ZIP."""
    # Source: Approximate coordinates from public domain geographical data
    zip_coords = {
        '92618': (33.66, -117.73),  # Irvine (Example)
        '92602': (33.70, -117.82),  # Irvine (Another Example)
        '92660': (33.62, -117.93),  # Newport Beach (Example)
        '92672': (33.43, -117.58),  # San Clemente (Example)
        # Add more ZIP codes and their coordinates here as needed!
    }
    return zip_coords.get(zip_code)

# THE COLUMN MODEL IS CRITICAL FOR THE MODEL!
# Order: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
print("Enter the characteristics for the property, or start with a supported ZIP code.")
print("Supported ZIPs for quick-fill: 92618 (Irvine), 92602 (Irvine), 92660 (Newport), 92672 (San Clemente)")
print("-" * 50)
print("1. Enter an Orange County ZIP Code (e.g., 92618) OR hit ENTER to input all 8 values manually.")
user_zip = input("Enter ZIP Code: ").strip()

latitude = None
longitude = None

if user_zip.isdigit() and len(user_zip) == 5:
    coords = get_zip_coords(user_zip)
    if coords:
        latitude, longitude = coords
        print(f"✅ Found coordinates for ZIP {user_zip}: Lat={latitude}, Lon={longitude}")
    else:
        print(f"⚠️ ZIP Code {user_zip} not found in the list. Proceeding to manual entry.")
        
# --- MANUAL INPUT SECTION ---
print("\n2. Enter the remaining 6 (or all 8) values separated by commas in this order:")
print("MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup")
if latitude is None:
    # If no ZIP was entered/found, we need all 8, so we ask for the remaining two as well.
    print("... and don't forget Latitude, Longitude")
else:
    print(f"**Lat and Lon will be automatically set to: {latitude}, {longitude}**")

try:
    # GET ALL INPUTS AS A SINGLE STRING
    user_input_str = input("Enter values: ")
    # Convert the comma-separated string into a list of floats
    input_values = [float(x.strip()) for x in user_input_str.split(',')]
    
    # Check if the user entered 6 values (for ZIP mode) or 8 values (for manual mode)
    if latitude is not None and len(input_values) == 6:
        # ZIP Mode: Append the coordinates we found
        input_values.extend([latitude, longitude])
        user_values = [input_values]
    elif latitude is None and len(input_values) == 8:
        # Manual Mode: Use the 8 values provided
        user_values = [input_values]
    else:
        # Invalid input count
        expected_count = 6 if latitude is not None else 8
        print(f"Error: You must enter exactly {expected_count} comma-separated values for your chosen mode.")
        user_values = None

    if user_values is not None:
        # Predict the price for the user's input
        new_pred = model.predict(user_values)
        print(f"\n✅ Predicted Median House Price: ${new_pred[0]*100000:,.2f}")

except ValueError:
    print("Error: Please ensure all inputs are valid numbers and you use commas to separate them.")

# --- NEW CODE BLOCK ENDS HERE ---

# ----------------------------------------------------------------------------------
#  <<<<<<<<<< END OF REPLACEMENT >>>>>>>>>>
# ----------------------------------------------------------------------------------

# 7. Visualize
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted (Random Forest Regressor)")
# plt.show()
