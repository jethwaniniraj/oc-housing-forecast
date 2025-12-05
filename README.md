# oc-housing-forecast
```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df[["MedInc", "HouseAge", "AveRooms"]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction – change values as needed for scenario analysis
print("Predicted Median Price (100k's):", model.predict([[9.0, 20, 6]]))

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Plots a red "perfect fit" line
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Housing Prices (in $100k)")
plt.show()
