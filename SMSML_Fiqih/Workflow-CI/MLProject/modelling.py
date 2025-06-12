import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    explained_variance_score
)

# 1. Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--l1_ratio", type=float, default=0.01)
args = parser.parse_args()

# 2. Load data
df = pd.read_csv("SMSML_Fiqih/Workflow-CI/MLProject/nasa_preprocessing/clean/train_FD001_clean.csv")

# 3. Feature engineering
max_cycle_per_unit = df.groupby("unit")["time_in_cycles"].transform("max")
df["RUL"] = max_cycle_per_unit - df["time_in_cycles"]

# 4. Feature selection
X = df.drop(columns=["unit", "time_in_cycles", "RUL"])
y = df["RUL"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.columns.tolist())

# 6. Start MLflow run
with mlflow.start_run():
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("l1_ratio", args.l1_ratio)

    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.log_metric("max_error", max_error(y_test, y_pred))
    mlflow.log_metric("explained_variance", explained_variance_score(y_test, y_pred))

    # Simpan model ke MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Simpan model lokal (opsional)
    joblib.dump(model, "model.joblib")