import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_raw_data(file_path):
    """
    Load data mentah dari NASA Turbofan FD001
    """
    column_names = ['unit', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = column_names
    return df

def add_rul(df):
    """
    Tambahkan kolom Remaining Useful Life (RUL)
    """
    rul = df.groupby('unit')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit', 'max_cycle']
    df = df.merge(rul, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def drop_flat_sensors(df):
    """
    Hapus sensor yang memiliki nilai konstan (std == 0)
    """
    sensor_cols = [col for col in df.columns if 'sensor_' in col]
    flat_cols = [col for col in sensor_cols if df[col].std() == 0]
    df.drop(columns=flat_cols, inplace=True)
    return df, flat_cols

def normalize_features(df):
    """
    Normalisasi fitur numerik (selain unit, time, dan RUL)
    """
    feature_cols = [col for col in df.columns if col not in ['unit', 'time_in_cycles', 'RUL']]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def main():
    print("⏳ Memuat dan memproses data FD001...")

    # Sesuaikan path data mentah dan path simpan
    raw_path = "SMSML_Fiqih/Membangun_model/nasa_preprocessing/raw/train_FD001.txt"
    save_path = "SMSML_Fiqih/Membangun_model/nasa_preprocessing/clean/train_FD001_clean.csv"
    
    # 1. Load
    df = load_raw_data(raw_path)
    
    # 2. Add RUL
    df = add_rul(df)

    # 3. Drop sensor yang tidak berguna
    df, dropped = drop_flat_sensors(df)
    print(f"🗑️  Sensor yang dibuang: {dropped}")

    # 4. Normalize
    df = normalize_features(df)

    # 5. Simpan hasil
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Data berhasil disimpan ke: {save_path}")

if __name__ == "__main__":
    main()
