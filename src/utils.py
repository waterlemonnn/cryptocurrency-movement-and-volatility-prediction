import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def process_data(df):
    data = df.copy()
    
    # Calculate log returns and log volume
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Log_Vol'] = np.log(data['Volume'] / data['Volume'].shift(1).replace(0, 1))
    
    # Calculate candle shadows
    data['Upper_Shadow'] = (data['High'] - data[['Close', 'Open']].max(axis=1)) / data['Close']
    data['Lower_Shadow'] = (data[['Close', 'Open']].min(axis=1) - data['Low']) / data['Close']
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate normalized MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Norm'] = (ema12 - ema26) / data['Close']
    
    # Create target variables for next hour
    data['Return_Future'] = data['Close'].shift(-1) / data['Close'] - 1
    data['Target_Dir'] = (data['Return_Future'] > 0).astype(int)
    data['Target_Vol'] = data['Return_Future'].abs()
    
    # Clean NaN and infinite values
    data.dropna(inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Filter columns to keep specific features
    cols_to_keep = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Log_Ret', 'Log_Vol', 'Upper_Shadow', 'Lower_Shadow', 
        'RSI', 'MACD_Norm', 
        'Target_Dir', 'Target_Vol'
    ]
    
    # Select only existing columns
    final_cols = [c for c in cols_to_keep if c in data.columns]
    
    return data[final_cols]

def plot_training_history(history, save_dir='outputs/plots'):
    
    # Create output directory
    base_dir = os.getcwd()
    full_save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(full_save_dir, exist_ok=True)
        
    # Initialize subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot direction accuracy
    acc_key = 'out_dir_accuracy' if 'out_dir_accuracy' in history.history else 'out_dir_acc'
    val_acc_key = 'val_' + acc_key
    
    if acc_key in history.history:
        axes[0].plot(history.history[acc_key], label='Train Acc', color='#00E676')
        axes[0].plot(history.history[val_acc_key], label='Val Acc', color='#FF1744', linestyle='--')
        axes[0].set_title('1. Direction Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot volatility error
    mae_key = 'out_vol_mae'
    val_mae_key = 'val_out_vol_mae'
    
    if mae_key in history.history:
        axes[1].plot(history.history[mae_key], label='Train MAE', color='cyan')
        axes[1].plot(history.history[val_mae_key], label='Val MAE', color='orange', linestyle='--')
        axes[1].set_title('2. Volatility Error (MAE)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot total loss
    axes[2].plot(history.history['loss'], label='Train Total', color='blue')
    axes[2].plot(history.history['val_loss'], label='Val Total', color='gray', linestyle='--')
    axes[2].set_title('3. Total Loss System')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot to file
    filename = 'training_metrics_final.png'
    save_path = os.path.join(full_save_dir, filename)
    
    try:
        plt.savefig(save_path)
        print(f"Training plots saved at: {save_path}")
    except Exception as e:
        print(f"Failed to save plots: {e}")