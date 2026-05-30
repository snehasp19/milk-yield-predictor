import pandas as pd
import joblib

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("milk_yield.csv")

# Features and Target
X = df[["Parity", "Days_in_Milk", "BCS", "Feed_Intake"]]
y = df["Milk_Yield"]

# 10-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_rmse = float("inf")

print("\nModel Performance:\n")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )

    rmse = -scores.mean()

    print(f"{name}: RMSE = {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

# Train best model on full dataset
best_model.fit(X, y)

# Save model
joblib.dump(best_model, "models/milk_model.pkl")

print("\nBest Model Saved Successfully!")