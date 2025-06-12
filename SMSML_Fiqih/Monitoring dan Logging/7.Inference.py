import mlflow.pyfunc
import pandas as pd
import sys
import os

# Fungsi memuat model dari MLflow
def load_model_from_mlflow(model_name="best_random_forest_model", model_stage="None", model_uri=None):
    if model_uri is None:
        model_uri = f"runs:/{get_latest_run_id()}/{model_name}"
    print(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Fungsi bantu untuk ambil ID run terakhir
def get_latest_run_id():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["attributes.start_time DESC"],
                              max_results=1)
    return runs[0].info.run_id

# Fungsi untuk prediksi
def run_inference(model, input_data_path):
    df = pd.read_csv(input_data_path)
    
    # Asumsikan input sudah siap dipakai, jika tidak, panggil fungsi preprocessing dulu
    if "unit_number" in df.columns or "cycle" in df.columns:
        from nasa_preprocessing.automate_Fiqih import preprocess
