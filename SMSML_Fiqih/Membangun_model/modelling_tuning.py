import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("SMSML_Fiqih/Membangun_model/nasa_preprocessing/clean/train_FD001_clean.csv")

# Siapkan fitur dan target
X = df.drop(['RUL', 'unit'], axis=1)
y = df['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi experiment MLflow
mlflow.set_experiment("Tuning_Model_Fiqih")

# Daftar hyperparameter untuk tuning
n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, 15]

# Loop hyperparameter tuning
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run():
            # Inisialisasi model
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Prediksi
            y_pred = model.predict(X_test)

            # Hitung metrik
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Logging manual
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2", r2)

            # Simpan model
            mlflow.sklearn.log_model(model, "model")

            print(f"Model dengan n_estimators={n_estimators}, max_depth={max_depth} => MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
