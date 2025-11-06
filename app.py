

# # ==============================================
# # 💰 Interactive LSTM Crypto Price Prediction Dashboard (Multi-Day Forecast)
# # ==============================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("💰 Interactive LSTM Crypto Price Prediction Dashboard")
st.markdown("Select a cryptocurrency to view its historical trend and multi-day predicted prices.")

# --------------------------
# Paths
# --------------------------
MODEL_DIR = "save_models/"
DATA_FILE = "data/top5_crypto_data.csv"  # default dataset

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("Settings")

# Upload new CSV
uploaded_file = st.sidebar.file_uploader("Upload your dataset CSV (optional)", type=["csv"])
# if uploaded_file:
#     df_all = pd.read_csv(uploaded_file)
# else:
if os.path.exists(DATA_FILE):
    df_all = pd.read_csv(DATA_FILE)
else:
    st.error("No dataset found. Please upload a CSV file!")
    st.stop()

# Available models (remove duplicates)
available_models = list({
    f.replace("lstm_", "").replace(".h5", "").replace(".keras","")
    for f in os.listdir(MODEL_DIR)
    if f.endswith(".h5") or f.endswith(".keras")
})
available_models.sort()
coin_symbol = st.sidebar.selectbox("Select Cryptocurrency", available_models)

# Number of days to display (historical)
last_n_days = st.sidebar.slider("Last N Days", min_value=30, max_value=180, value=60, step=10)

# Next N days to forecast
next_n_days = st.sidebar.slider("Next N Days Forecast", min_value=1, max_value=5, value=1, step=1)

# Chart type
chart_type = st.sidebar.radio("Chart Type", ["Line Chart", "Candlestick Chart"])

# --------------------------
# Load model and scaler
# --------------------------
model_path_h5 = os.path.join(MODEL_DIR, f"lstm_{coin_symbol}.h5")
model_path_keras = os.path.join(MODEL_DIR, f"lstm_{coin_symbol}.keras")
scaler_path = os.path.join(MODEL_DIR, f"scaler_{coin_symbol}.save")

if not os.path.exists(scaler_path):
    st.error(f"Scaler for {coin_symbol} not found!")
    st.stop()
scaler = joblib.load(scaler_path)

if os.path.exists(model_path_h5):
    model = load_model(model_path_h5, compile=False)
elif os.path.exists(model_path_keras):
    model = load_model(model_path_keras, compile=False)
else:
    st.error(f"Model for {coin_symbol} not found!")
    st.stop()

# --------------------------
# Filter coin data
# --------------------------
df = df_all[df_all["symbol"]==coin_symbol].copy()
if df.empty:
    st.error(f"No data for {coin_symbol}")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# Limit last N days
df_display = df.tail(last_n_days)

# --------------------------
# Prepare features
# --------------------------
features = ['open','high','low','close','adjclose','volume']
look_back = 60

if len(df) < look_back:
    st.error("Not enough data for prediction (needs >=60 rows)")
    st.stop()

scaled_data = scaler.transform(df[features].values)
X_input = scaled_data[-look_back:]
X_seq = np.expand_dims(X_input, axis=0)  # shape: (1, 60, 6)

# --------------------------
# Multi-day prediction
# --------------------------
pred_prices = []
pred_dates = []

last_date = df["timestamp"].iloc[-1]

for day in range(next_n_days):
    pred_scaled = model.predict(X_seq)
    dummy = np.zeros((1, scaled_data.shape[1]))
    dummy[0,4] = pred_scaled
    pred_price_day = scaler.inverse_transform(dummy)[0,4]
    pred_prices.append(pred_price_day)
    pred_dates.append(last_date + pd.Timedelta(days=day+1))
    
    # Update X_seq for next prediction
    new_seq = X_seq[0,1:,:].copy()  # drop first row
    last_row = X_seq[0,-1,:].copy()
    last_row[4] = pred_scaled  # update adjclose
    new_seq = np.vstack([new_seq, last_row])
    X_seq = np.expand_dims(new_seq, axis=0)

# --------------------------
# Display Metrics
# --------------------------
latest_close = df["adjclose"].iloc[-1]
next_day_price = pred_prices[0]
change = ((next_day_price - latest_close)/latest_close)*100

col1, col2, col3 = st.columns(3)
col1.metric("📅 Latest Close Price", f"${latest_close:.4f}")
col2.metric("🔮 Predicted Next Close", f"${next_day_price:.4f}")
col3.metric("Expected Change (%)", f"{change:.2f}%", delta_color="inverse" if change<0 else "normal")

# --------------------------
# Moving Averages
# --------------------------
df_display["MA7"] = df_display["adjclose"].rolling(7).mean()
df_display["MA30"] = df_display["adjclose"].rolling(30).mean()

# --------------------------
# Plotly Chart with Multi-Day Forecast
# --------------------------
if chart_type == "Line Chart":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_display["timestamp"], y=df_display["adjclose"],
                             mode="lines+markers", name="Actual Adj Close"))
    fig.add_trace(go.Scatter(x=df_display["timestamp"], y=df_display["MA7"], mode="lines", name="MA7"))
    fig.add_trace(go.Scatter(x=df_display["timestamp"], y=df_display["MA30"], mode="lines", name="MA30"))
    # Multi-day forecast
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=pred_prices,
        mode="lines+markers+text",
        name="Forecast",
        marker=dict(color="red", size=10),
        line=dict(dash="dash"),
        text=[f"${p:.4f}" for p in pred_prices],
        textposition="top center"
    ))
    fig.update_layout(title=f"{coin_symbol} Price Trend (Last {last_n_days} Days + Next {next_n_days} Days)",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template="plotly_dark")
else:
    fig = go.Figure(data=[go.Candlestick(
        x=df_display["timestamp"],
        open=df_display["open"],
        high=df_display["high"],
        low=df_display["low"],
        close=df_display["adjclose"],
        name="Candlestick"
    )])
    fig.add_trace(go.Scatter(x=df_display["timestamp"], y=df_display["MA7"], mode="lines", name="MA7"))
    fig.add_trace(go.Scatter(x=df_display["timestamp"], y=df_display["MA30"], mode="lines", name="MA30"))
    # Multi-day forecast
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=pred_prices,
        mode="lines+markers+text",
        name="Forecast",
        marker=dict(color="red", size=10),
        line=dict(dash="dash"),
        text=[f"${p:.4f}" for p in pred_prices],
        textposition="top center"
    ))
    fig.update_layout(title=f"{coin_symbol} Candlestick Trend (Last {last_n_days} Days + Next {next_n_days} Days)",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Download Multi-Day Prediction CSV
# --------------------------
pred_df = pd.DataFrame({
    "timestamp": pred_dates,
    "coin": [coin_symbol]*len(pred_dates),
    "predicted_adjclose": pred_prices,
    "latest_adjclose": [latest_close]*len(pred_dates),
    "expected_change_percent": [((p-latest_close)/latest_close)*100 for p in pred_prices]
})
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast CSV", data=csv, file_name=f"{coin_symbol}_forecast_{next_n_days}_days.csv", mime="text/csv")

st.markdown("---")
st.caption("📘 Predictions based on last 60 days of data using trained LSTM models. Multi-day forecast enabled.")





