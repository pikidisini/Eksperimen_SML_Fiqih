import requests
import json

url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

data = {
    "dataframe_split": {
        "columns": [
            "op_setting_1", "op_setting_2", "op_setting_3",
            "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
            "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17",
            "sensor_20", "sensor_21"
        ],
        "data": [[
            0.0, 0.0, 100.0, 518.67, 641.82, 1587.73, 1400.6,
            14.62, 21.61, 554.37, 2388.06, 9030.71, 2.64,
            1.33, 8.42, 0.03, 392.0, 2388.0, 100.0, 39.0
        ]]
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print("Prediction:", response.json())