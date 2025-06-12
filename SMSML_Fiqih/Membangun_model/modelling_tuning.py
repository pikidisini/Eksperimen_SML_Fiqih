import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from nasa_preprocessing.automate_Fiqih import preprocess_fd001

def run_hyperparameter_tuning(data_path):
    # Load data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = preprocess_fd001(df)

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5]
    }

    # Initialize model and GridSearchCV
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)

    # Fit model
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Logging
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")

        print(f"Best RMSE: {rmse}")
        print("Best Parameters:", best_params)

if __name__ == "__main__":
    run_hyperparameter_tuning("data/train_FD001.txt")
