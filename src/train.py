import json
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_loader import get_processed_data
from model import build_attention_model
from utils import plot_training_history

# Setup base paths
BASE_DIR = os.getcwd()
config_path = os.path.join(BASE_DIR, 'config', 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)

# Create output directories
model_dir = os.path.join(BASE_DIR, 'outputs', 'models')
log_dir = os.path.join(BASE_DIR, 'outputs', 'logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

model_save_path = os.path.join(BASE_DIR, config['paths']['model_save'])
log_save_path = os.path.join(BASE_DIR, config['paths']['log_save'])
scaler_save_path = os.path.join(model_dir, 'scaler_data.pkl')

# Load and prepare data
print("Loading data...")
(X_train, y_dir_train, y_vol_train), \
(X_val, y_dir_val, y_vol_val), \
(X_test, y_test_dir, y_test_vol), scaler = get_processed_data(
    config['data']['filename'], 
    config['data']['processed_filename'],
    config['data']['seq_len']
)

# Save scaler for consistency
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to {scaler_save_path}")

# Build model architecture
print("Building model...")
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_attention_model(input_shape, lr=config['training']['learning_rate'])
model.summary()

# Configure training
print("Starting training...")
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# Setup callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=config['training']['patience'], restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)

# Train model
history = model.fit(
    X_train, [y_dir_train, y_vol_train],
    validation_data=(X_val, [y_dir_val, y_vol_val]),
    epochs=config['training']['epochs'],
    batch_size=config['training']['batch_size'],
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Save logs and plots
print("Saving artifacts...")
pd.DataFrame(history.history).to_csv(log_save_path)
plot_training_history(history)