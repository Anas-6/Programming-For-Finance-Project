import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set Zombie Theme Styling
def apply_zombie_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');
            body {
                background-color: #0b0c10;
                color: #c5c6c7;
            }
            h1, h2, h3, h4 {
                font-family: 'Creepster', cursive;
                color: #66fcf1;
            }
            .stButton>button {
                background-color: #1f2833;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        st.image("assets/gifs/zombie_header.gif", use_container_width=True)
    except Exception:
        st.warning("⚠️ Header GIF not found.")

# 🔁 Main App Function
def zombie_app():
    apply_zombie_theme()
    st.title("💀 Zombie Theme: Stock Price Prediction")
    st.markdown("Enter a stock ticker or upload your dataset to begin.")

    data_source = st.radio("Choose data source:", ("Yahoo Finance", "Upload CSV"))
    df = None

    if data_source == "Yahoo Finance":
        import yfinance as yf
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
        if st.button("Fetch Data"):
            df = yf.download(ticker, period="1y")
            if not df.empty:
                st.success(f"Fetched {len(df)} rows for {ticker}")
            else:
                st.error("Failed to fetch data. Please check the ticker symbol.")
    else:
        uploaded_file = st.file_uploader("Upload your Kragle CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")

    if df is not None and not df.empty:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)

        # Fix: Closing Price Trend Graph
        st.subheader("📈 Closing Price Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='lime')))
        fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Closing Price")
        st.plotly_chart(fig)

        # Feature Engineering
        df = df.copy()
        df['Date_ordinal'] = df.index.map(pd.Timestamp.toordinal)

        X = df[['Date_ordinal']]
        y = df['Close']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📉 Model Performance")
        try:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**R² Score**: {r2:.4f}")
            st.write(f"**RMSE**: {rmse:.4f}")
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")

        # Prediction Plot
        st.subheader("🧟 Predicted vs Actual Closing Prices")
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Actual', color='orange')
        ax.plot(y_pred, label='Predicted', color='lime')
        ax.set_title("Prediction vs Actual")
        ax.legend()
        st.pyplot(fig2)

        st.markdown("---")
        try:
            st.image("assets/gifs/zombie_line.gif", use_container_width=True)
        except Exception:
            st.warning("⚠️ Footer GIF not found.")

    else:
        st.warning("Please upload a dataset or fetch a valid ticker first.")
