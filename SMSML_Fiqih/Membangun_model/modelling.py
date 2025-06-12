import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("SMSML_Fiqih/Membangun_model/nasa_preprocessing\clean/train_FD001_clean.csv")

max_cycle_per_unit = df.groupby("unit")["time_in_cycles"].transform("max")
df["RUL"] = max_cycle_per_unit - df["time_in_cycles"]

# 2. Feature selection
X = df.drop(columns=["unit", "time_in_cycles", "RUL"])
y = df["RUL"]

print("Feature columns used in training:")
print(X.columns.tolist())

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Autolog with MLflow
mlflow.sklearn.autolog()

with mlflow.start_run():
    # 5. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

#joblib.dump(model, "SMSML_Fiqih/Membangun_model/model.joblib")