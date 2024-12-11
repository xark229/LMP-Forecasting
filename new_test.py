import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data from data.csv
data_path = "bus_data.csv"  # Replace with the correct path if needed
df = pd.read_csv(data_path)

# Features and target variable
X = df[["Voltage_Mag(pu)","Voltage_Ang(deg)","P(MW)","Q(MVAr)"]]
y = df["Lamda"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print("\n================ Model Evaluation ================")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("=================================================\n")

# Plot actual vs predicted values for the test set
plt.figure(figsize=(14, 8))

plt.plot(y_test.values, label="Actual Test Set", color="blue", linewidth=2, marker='o')
plt.plot(y_pred, label="Predicted Test Set", color="orange", linewidth=2, marker='x')

# Add titles and labels
plt.title("Actual vs Predicted on Test Set", fontsize=18, fontweight="bold", color="navy")
plt.xlabel("Data Points (Index in Test Set)", fontsize=14)
plt.ylabel("Lambda_P", fontsize=14)

# Add legend and grid
plt.legend(fontsize=12)
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

# Annotate performance metrics
plt.text(
    0.05 * len(y_test), max(y_test) * 0.9, f"MSE: {mse:.2f}\nR²: {r2:.2f}",
    fontsize=12, color="black", bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5")
)

plt.tight_layout()
plt.show()