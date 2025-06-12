import pandas as pd
import joblib

# Load model
model_path = "SMSML_Fiqih/Membangun_model/model.joblib"
model = joblib.load(model_path)

# Contoh data baru (bisa kamu sesuaikan)
sample_input = pd.DataFrame([{
    'op_setting_1': 0.5,
    'op_setting_2': 0.2,
    'op_setting_3': 0.0,
    'sensor_2': 641.82,
    'sensor_3': 1587.85,
    'sensor_4': 1400.6,
    'sensor_5': 14.62,
    'sensor_6': 21.61,
    'sensor_7': 554.37,
    'sensor_8': 2388.02,
    'sensor_9': 9046.19,
    'sensor_11': 47.47,
    'sensor_12': 522.28,
    'sensor_13': 2388.07,
    'sensor_14': 8138.82,
    'sensor_15': 8.42,
    'sensor_16': 0.03,
    'sensor_17': 392,
    'sensor_20': 39.06,
    'sensor_21': 23.0
}])

# Predict
prediction = model.predict(sample_input)

print("Prediksi RUL:", round(prediction[0], 2))
