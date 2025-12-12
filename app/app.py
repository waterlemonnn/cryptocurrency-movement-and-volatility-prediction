import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Multiply, Permute, Reshape, Dense
import os
import json
import joblib 
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

# Config & Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.json')

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

DATA_FILENAME = os.path.join(BASE_DIR, config['data']['processed_filename'])
MODEL_FILENAME = os.path.join(BASE_DIR, config['paths']['model_save'])
LOG_FILENAME = os.path.join(BASE_DIR, config['paths']['log_save'])
SCALER_FILENAME = os.path.join(BASE_DIR, 'outputs', 'models', 'scaler_data.pkl')
SEQ_LEN = config['data']['seq_len']

st.set_page_config(page_title="BTC Predictor", layout="wide")
st.title("Bitcoin AI Predictor (Model: Bi-LSTM + Attention)")

# Custom Layer
def attention_layer(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Reshape((inputs.shape[2], time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    return Multiply()([inputs, a_probs])

# Helpers
def create_sequences(data_scaled, seq_len):
    X = []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i])
    return np.array(X)

# Loaders
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_FILENAME): return None
    return load_model(MODEL_FILENAME, custom_objects={'attention_layer': attention_layer})

@st.cache_data
def get_data():
    if not os.path.exists(DATA_FILENAME): return None, None
    
    # Load processed data
    df = pd.read_csv(DATA_FILENAME)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Features (Must match training)
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Log_Ret', 'Log_Vol', 'Upper_Shadow', 'Lower_Shadow', 
        'RSI', 'MACD_Norm'
    ]
    
    # Load Saved Scaler (Critical for Consistency)
    if not os.path.exists(SCALER_FILENAME):
        st.error("⚠️ Scaler file missing. Please run train.py first.")
        return None, None
        
    scaler = joblib.load(SCALER_FILENAME)
    
    # Transform using saved scaler
    data_scaled = scaler.transform(df[feature_cols])
    
    # Limit for performance
    LIMIT_DATA = 20000 
    if len(data_scaled) > LIMIT_DATA:
        data_scaled = data_scaled[-LIMIT_DATA:]
        df_view = df.iloc[-LIMIT_DATA:].copy()
    else:
        df_view = df.copy()

    X = create_sequences(data_scaled, SEQ_LEN)
    df_view = df_view.iloc[SEQ_LEN:].reset_index(drop=True)
    
    return df_view, X

# Main Execution
model = load_ai_model()
if model is None: st.error(f"Model not found at: {MODEL_FILENAME}"); st.stop()

with st.spinner('Loading data...'):
    df, X_input = get_data()

if df is None: st.stop()

# Run Prediction
if 'predictions' not in st.session_state:
    with st.spinner('Running AI prediction...'):
        preds = model.predict(X_input, verbose=0)
        st.session_state['predictions'] = preds[0].flatten()
        st.session_state['vol_preds'] = preds[1].flatten()

# Assign Predictions
df['prob_naik'] = st.session_state['predictions']
df['vol_pred'] = st.session_state['vol_preds']

# Standard Threshold (0.5)
df['pred_class'] = (df['prob_naik'] > 0.5).astype(int)
df['label'] = df['pred_class'].map({1: 'UP', 0: 'DOWN'})
df['confidence'] = np.where(df['prob_naik'] > 0.5, df['prob_naik'], 1 - df['prob_naik'])

# Metrics Check
if 'Target_Dir' in df.columns:
    df['is_correct'] = df['pred_class'] == df['Target_Dir']
else:
    df['Actual_Dir'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['is_correct'] = df['pred_class'] == df['Actual_Dir']

# Dashboard Layout
c1, c2 = st.columns([1, 4])
with c1:
    st.subheader("📅 Select Date")
    date_sel = st.date_input("Date", value=df['Date'].max())
    
    st.markdown("---")
    st.info(
        """
        **Note:**
        
        This result is not a one-shot prediction for 24 hours.
        
        The model performs **24 individual predictions** (step-by-step). 
        Each hour, the model looks at historical data to predict the next hour's price direction.
        """
    )

start = pd.to_datetime(date_sel)
day_df = df[(df['Date'] >= start) & (df['Date'] <= start + pd.Timedelta(hours=23, minutes=59))]

with c2:
    if not day_df.empty:
        acc = day_df['is_correct'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daily Accuracy", f"{acc:.0%}")
        m2.metric("Win", f"{day_df['is_correct'].sum()}")
        m3.metric("Loss", f"{len(day_df)-day_df['is_correct'].sum()}")
        m4.metric("Avg Conf", f"{day_df['confidence'].mean():.1%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=day_df['Date'], y=day_df['Close'], 
            name='Price', line=dict(color='#2E86C1', width=2),
            hovertemplate='<b>Price:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        wins = day_df[day_df['is_correct']]
        losses = day_df[~day_df['is_correct']]
        
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins['Date'], y=wins['Close'], mode='markers', name='WIN',
                marker=dict(symbol='triangle-up', color='#00E676', size=14, line=dict(width=1, color='black')),
                customdata=np.stack((wins['confidence']*100, wins['label']), axis=-1),
                hovertemplate="<b>WIN</b><br>Price: $%{y:,.2f}<br>Pred: %{customdata[1]}<br>Conf: %{customdata[0]:.2f}%<extra></extra>"
            ))
            
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses['Date'], y=losses['Close'], mode='markers', name='LOSS',
                marker=dict(symbol='triangle-down', color='#FF1744', size=14, line=dict(width=1, color='black')),
                customdata=np.stack((losses['confidence']*100, losses['label']), axis=-1),
                hovertemplate="<b>LOSS</b><br>Price: $%{y:,.2f}<br>Pred: %{customdata[1]}<br>Conf: %{customdata[0]:.2f}%<extra></extra>"
            ))
        
        fig.update_layout(height=500, template="plotly_dark", title=f"Analysis: {date_sel}", hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Detailed Data"):
            st.dataframe(day_df[['Date', 'Close', 'label', 'confidence', 'is_correct']].style.format({'confidence': '{:.2%}', 'Close': '${:.2f}'}))
    else:
        st.warning("No data for selected date.")

# 7. Tabs
st.markdown("---")
st.header("🔍 Model & Data Diagnostics")
tab1, tab2, tab3 = st.tabs(["Processed Data", "Training History", "Evaluation Metrics"])

with tab1:
    st.subheader("1. Data Sample")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("2. Feature Analysis")
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("**RSI (Relative Strength Index)**")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='#E040FB', width=1)))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        fig_rsi.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_f2:
        st.markdown("**MACD Normalized**")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Norm'], fill='tozeroy', line=dict(color='#00E676', width=1)))
        fig_macd.add_hline(y=0, line_color="white", line_width=0.5)
        fig_macd.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
        st.plotly_chart(fig_macd, use_container_width=True)

    st.markdown("**Logarithmic Return (Volatility)**")
    fig_log = go.Figure()
    fig_log.add_trace(go.Scatter(x=df['Date'], y=df['Log_Ret'], mode='lines', line=dict(color='#FFFF00', width=0.5)))
    fig_log.update_layout(height=300, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_log, use_container_width=True)

with tab2:
    st.header("📈 Training Performance")
    if os.path.exists(LOG_FILENAME):
        history_df = pd.read_csv(LOG_FILENAME)
        
        # Accuracy (Full Width)
        st.subheader("1. Direction Accuracy")
        fig_acc = go.Figure()
        acc_col = 'out_dir_accuracy' if 'out_dir_accuracy' in history_df.columns else 'out_dir_acc'
        val_acc_col = 'val_' + acc_col
        
        if acc_col in history_df.columns:
            fig_acc.add_trace(go.Scatter(y=history_df[acc_col], name='Train Acc', line=dict(color='#00E676')))
            fig_acc.add_trace(go.Scatter(y=history_df[val_acc_col], name='Val Acc', line=dict(color='#FF1744', dash='dot')))
            fig_acc.update_layout(height=400, template="plotly_dark", xaxis_title="Epoch", yaxis_title="Accuracy")
            st.plotly_chart(fig_acc, use_container_width=True)

        # Volatility & Loss (Split Columns)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2. Volatility Error (MAE)")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(y=history_df['out_vol_mae'], name='Train MAE', line=dict(color='cyan')))
            fig_vol.add_trace(go.Scatter(y=history_df['val_out_vol_mae'], name='Val MAE', line=dict(color='orange', dash='dot')))
            fig_vol.update_layout(height=350, template="plotly_dark", xaxis_title="Epoch", yaxis_title="MAE")
            st.plotly_chart(fig_vol, use_container_width=True)

        with c2:
            st.subheader("3. Total Loss")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history_df['loss'], name='Train Loss', line=dict(color='white')))
            fig_loss.add_trace(go.Scatter(y=history_df['val_loss'], name='Val Loss', line=dict(color='gray', dash='dot')))
            fig_loss.update_layout(height=350, template="plotly_dark", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.error("Log file not found.")

with tab3:
    st.header("📝 Evaluation Report")
    
    if 'predictions' in st.session_state:
        # Prepare evaluation data
        eval_df = df.copy()
        eval_df['Actual_Vol'] = (eval_df['Close'].shift(-4) / eval_df['Close'] - 1).abs()
        target_col = 'Target_Dir' if 'Target_Dir' in eval_df.columns else 'Actual_Dir'
        eval_df = eval_df.dropna(subset=[target_col, 'Actual_Vol'])
        
        limit_idx = len(eval_df)
        X_eval = X_input[:limit_idx]
        
        # Run prediction
        preds_eval = model.predict(X_eval, verbose=0)
        y_pred_probs = preds_eval[0].flatten() 
        y_pred_vol = preds_eval[1].flatten() 
        
        y_true = eval_df[target_col].values
        y_true_vol = eval_df['Actual_Vol'].values
        y_pred_class = (y_pred_probs > 0.5).astype(int)

        # Classification (Split 1:2)
        st.subheader("A. Direction Prediction")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            cm = confusion_matrix(y_true, y_pred_class)
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred DOWN', 'Pred UP'], y=['True DOWN', 'True UP'], colorscale='Blues', text=cm, texttemplate="%{text}", showscale=False))
            fig_cm.update_layout(title="Confusion Matrix", width=300, height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c2:
            report = classification_report(y_true, y_pred_class, output_dict=True)
            st.markdown("**Detailed Metrics:**")
            st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Greens', subset=['f1-score', 'precision', 'recall']).format("{:.2%}"))

        # Regression
        st.markdown("---")
        st.subheader("B. Volatility Prediction")
        
        mae = mean_absolute_error(y_true_vol, y_pred_vol)
        mse = mean_squared_error(y_true_vol, y_pred_vol)
        rmse = np.sqrt(mse)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("MAE", f"{mae:.5f}")
        k2.metric("RMSE", f"{rmse:.5f}")
        k3.metric("MSE", f"{mse:.6f}")
        
        st.markdown("**Actual vs Predicted Volatility**")
        
        # Downsample for scatter
        if len(y_true_vol) > 500:
            idx = np.random.choice(len(y_true_vol), 500, replace=False)
            p_true, p_pred = y_true_vol[idx], y_pred_vol[idx]
        else:
            p_true, p_pred = y_true_vol, y_pred_vol
            
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=p_true, y=p_pred, mode='markers', marker=dict(color='orange', size=5, opacity=0.6), name='Data'))
        max_val = max(p_true.max(), p_pred.max())
        fig_reg.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='white', dash='dash'), name='Ideal'))
        fig_reg.update_layout(xaxis_title="Actual Volatility", yaxis_title="Predicted", template="plotly_dark", height=400)
        st.plotly_chart(fig_reg, use_container_width=True)
        
    else:
        st.warning("Prediction data not yet available in session.")