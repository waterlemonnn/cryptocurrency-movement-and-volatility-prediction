import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from utils import process_data 

def load_kaggle_data(filepath):
    df = pd.read_csv(filepath)
    
    # Standardize Column Names
    rename_map = {
        'Open time': 'Date', 'Open': 'Open', 'High': 'High', 
        'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
    }
    
    cols = df.columns
    final_rename = {}
    for k, v in rename_map.items():
        if k in cols: final_rename[k] = v
        
    df.rename(columns=final_rename, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    final_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]
    
    return df

def create_sequences(data_scaled, target_dir, target_vol, seq_len):
    X, y_dir, y_vol = [], [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i]) 
        y_dir.append(target_dir[i]) 
        y_vol.append(target_vol[i]) 
    return np.array(X), np.array(y_dir), np.array(y_vol)

def get_processed_data(raw_filepath, processed_filepath, seq_len):
    # Load Raw Data
    df = load_kaggle_data(raw_filepath)
    
    # Feature Engineering
    df_processed = process_data(df)
    
    # Save intermediate processed data
    base_dir = os.getcwd()
    full_proc_path = os.path.join(base_dir, processed_filepath)
    os.makedirs(os.path.dirname(full_proc_path), exist_ok=True)
    df_processed.to_csv(full_proc_path)
    
    # Split Data (Chronological Split 80/10/10)
    n = len(df_processed)
    train_df = df_processed[0 : int(n*0.8)]
    val_df = df_processed[int(n*0.8) : int(n*0.9)]
    test_df = df_processed[int(n*0.9) :]
    
    # 4. Scaling
    feature_cols = [c for c in df_processed.columns if 'Target' not in c and 'Return' not in c]
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    
    train_scaled = scaler.transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # 5. Create Sequences
    X_train, y_train_dir, y_train_vol = create_sequences(train_scaled, train_df['Target_Dir'].values, train_df['Target_Vol'].values, seq_len)
    X_val, y_val_dir, y_val_vol = create_sequences(val_scaled, val_df['Target_Dir'].values, val_df['Target_Vol'].values, seq_len)
    X_test, y_test_dir, y_test_vol = create_sequences(test_scaled, test_df['Target_Dir'].values, test_df['Target_Vol'].values, seq_len)
    
    return (X_train, y_train_dir, y_train_vol), (X_val, y_val_dir, y_val_vol), (X_test, y_test_dir, y_test_vol), scaler