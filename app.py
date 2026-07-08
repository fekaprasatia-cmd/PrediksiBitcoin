import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import joblib
import tensorflow as tf

# --- Page Configuration ---
st.set_page_config(
    page_title='Bitcoin Intelligence Dashboard',
    page_icon='₿',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*='css'] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F7F9FB; }
    div[data-testid='stMetric'] {
        background-color: #ffffff;
        border: 1px solid #ECEFF2;
        border-radius: 15px;
        padding: 15px 25px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .main-header { font-weight: 800; color: #1A1C1E; font-size: 2.5rem !important; margin-bottom: 0.5rem; }
    .sub-header { color: #64748B; font-size: 1.1rem; margin-bottom: 2rem; }
    .status-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model('/content/your_model.h5', custom_objects=custom_objects)
        scaler = joblib.load('/content/your_scaler.pkl')
        return model, scaler, None
    except Exception as e: return None, None, str(e)

model, scaler, err = load_assets()

def make_forecast(df, model, scaler, days, n_steps=60):
    try:
        if len(df) < n_steps: return None
        close_prices = df['Close'].values.reshape(-1, 1)

        log_returns = np.diff(np.log(close_prices.flatten()))
        vol_std = np.std(log_returns)
        
        last_window = close_prices[-n_steps:]
        scaled_input = scaler.transform(last_window)
        current_seq = scaled_input.reshape(1, n_steps, 1)

        preds = []
        current_last_price = close_prices[-1][0]

        for i in range(days):
            p_scaled = model.predict(current_seq, verbose=0)[0, 0]
            p_model = scaler.inverse_transform([[p_scaled]])[0, 0]
            
            noise_scale = vol_std * 2.5 * (1 + i * 0.05)
            shock = np.random.normal(0, noise_scale)
            p_final = p_model * np.exp(shock)
            
            p_final = np.clip(p_final, current_last_price * 0.90, current_last_price * 1.10)
            preds.append(p_final)
            
            current_last_price = p_final
            p_scaled_next = scaler.transform([[p_final]])[0, 0]
            current_seq = np.append(current_seq[:, 1:, :], [[[p_scaled_next]]], axis=1)

        return np.array(preds)
    except:
        return None

# --- Sidebar ---
st.sidebar.image('https://cryptologos.cc/logos/bitcoin-btc-logo.png', width=60)
st.sidebar.markdown('## Crypto Control')
with st.sidebar.expander("📅 Range Waktu Historis", expanded=True):
    start_date = st.date_input('Mulai', value=datetime(2020, 1, 1))
    end_date = st.date_input('Selesai', value=datetime.now())
with st.sidebar.expander("🔮 Forecast Horizon", expanded=True):
    horizon = st.slider('Jumlah Hari Prediksi', 1, 30, 14)

# --- Main UI ---
st.markdown('<h1 class="main-header">Bitcoin Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data Real-time & Prediksi AI (Balanced Stochastic)</p>', unsafe_allow_html=True)

try:
    df_live = yf.download('BTC-USD', start=start_date, end=end_date, auto_adjust=False)

    if not df_live.empty:
        if isinstance(df_live.columns, pd.MultiIndex):
            df_live.columns = [col[0] for col in df_live.columns]

        latest_price = float(df_live['Close'].iloc[-1])
        prev_price = float(df_live['Close'].iloc[-2])
        delta = ((latest_price - prev_price) / prev_price) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price (Live)", f"${latest_price:,.2f}", f"{delta:+.2f}%")
        c2.metric("24h High", f"${float(df_live['High'].iloc[-1]):,.2f}")
        c3.metric("24h Low", f"${float(df_live['Low'].iloc[-1]):,.2f}")
        status_color = "#28a745" if not err else "#dc3545"
        c4.markdown(f"<div style='text-align:center; padding-top:20px;'><span class='status-badge' style='background:{status_color}22; color:{status_color};'>● {'System Active' if not err else 'Model Error'}</span></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(['📊 Candlestick & Forecast', '🧪 Evaluation Metrics', '🏗️ Architecture'])

        with tab1:
            if model and scaler:
                preds_res = make_forecast(df_live, model, scaler, horizon)
                if preds_res is not None:
                    future_dates = [df_live.index[-1] + timedelta(days=i+1) for i in range(horizon)]

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df_live.index[-100:],
                        open=df_live['Open'].iloc[-100:],
                        high=df_live['High'].iloc[-100:],
                        low=df_live['Low'].iloc[-100:],
                        close=df_live['Close'].iloc[-100:],
                        name='Actual Price'
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=preds_res,
                        name='AI Forecast (Moderate Stochastic)',
                        line=dict(color='#007bff', width=3, dash='dot')
                    ))

                    fig.update_layout(
                        xaxis_rangeslider_visible=False,
                        template='plotly_white', height=600,
                        title='Analisis Pergerakan Harga & Prediksi Volatilitas'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 🔮 AI Prediction Details")
                    # Gain/Loss calculation: Current Predicted vs Previous Day
                    # For the first day of prediction, compare with latest_price
                    prices_series = np.insert(preds_res, 0, latest_price)
                    pct_changes = np.diff(prices_series) / prices_series[:-1] * 100
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates, 
                        'Predicted Price (USD)': preds_res,
                        'Gain/Loss (%)': pct_changes
                    })
                    
                    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d-%m-%Y')
                    forecast_df['Predicted Price (USD)'] = forecast_df['Predicted Price (USD)'].map('${:,.2f}'.format)
                    forecast_df['Gain/Loss (%)'] = forecast_df['Gain/Loss (%)'].map('{:+.2f}%'.format)
                    st.table(forecast_df.set_index('Date'))

        with tab2:
            st.markdown("### Model Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", "2957.25"); m2.metric("MAE", "2438.21")
            m3.metric("MAPE", "2.92%"); m4.metric("Accuracy", "97.08%")

        with tab3:
            st.markdown("### Arsitektur Hybrid GRU-LSTM PSO")
            st.write("Integrasi Deep Learning untuk analisis volatilitas pasar dengan optimasi Particle Swarm Optimization.")

except Exception as e: st.error(f'Error: {e}')
