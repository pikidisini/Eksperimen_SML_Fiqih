# modelling.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_preprocessed_data(path='nasa_preprocessing/train_FD001.txt'):
    return pd.read_csv(path)

def train_model():
    df = load_preprocessed_data()
    
    X = df.drop("RUL", axis=1)
    y = df["RUL"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "trained_model.pkl")
        print("Model trained and saved with RMSE:", rmse)

if __name__ == "__main__":
    train_model()
