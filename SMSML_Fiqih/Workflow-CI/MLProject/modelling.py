import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    explained_variance_score
)

# 1. Load data
df = pd.read_csv("SMSML_Fiqih/Workflow-CI/MLProject/nasa_preprocessing/clean/train_FD001_clean.csv")

# 2. Feature engineering
max_cycle_per_unit = df.groupby("unit")["time_in_cycles"].transform("max")
df["RUL"] = max_cycle_per_unit - df["time_in_cycles"]

# 3. Feature selection
X = df.drop(columns=["unit", "time_in_cycles", "RUL"])
y = df["RUL"]

print("Feature columns used in training:")
print(X.columns.tolist())

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Manual MLflow Logging
with mlflow.start_run():

    # 6. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    exp_var = explained_variance_score(y_test, y_pred)

    # 8. Log metrics manually
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("max_error", max_err)
    mlflow.log_metric("explained_variance", exp_var)

    # 9. Log model as artifact
    mlflow.sklearn.log_model(model, "model")

    # 10. Optional: save local copy
    joblib.dump(model, "model.joblib")

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")