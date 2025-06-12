from prometheus_client import start_http_server, Counter, Summary
import time
from Inference import load_model_from_mlflow, run_inference
import pandas as pd

# METRICS
INFERENCE_REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
INFERENCE_DURATION = Summary('inference_duration_seconds', 'Time spent processing inference')
PREDICTION_VALUE = Summary('prediction_value', 'Summary of predicted values')

# LOAD MODEL ONCE
model = load_model_from_mlflow()

@INFERENCE_DURATION.time()
def infer(input_path):
    INFERENCE_REQUEST_COUNT.inc()
    
    df = pd.read_csv(input_path)
    preds = run_inference(model, input_path)
    
    for pred in preds:
        PREDICTION_VALUE.observe(pred)
    
    return preds

def main():
    start_http_server(8000)  # Prometheus scrape endpoint
    print("Prometheus metrics server started on port 8000")

    # Simulasi setiap 10 detik
    while True:
        try:
            preds = infer("data/test_input.csv")
            print(f"Inference done, predicted values: {preds[:5]}")
        except Exception as e:
            print("Error during inference:", e)
        time.sleep(10)

if __name__ == "__main__":
    main()
