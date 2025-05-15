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

    st.image("assets/gifs/zombie_header.gif", use_column_width=True)


# Main Zombie Theme Logic
def run_zombie_theme():
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
            st.success(f"Fetched {len(df)} rows for {ticker}")
    else:
        uploaded_file = st.file_uploader("Upload your Kragle CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")

    if df is not None and not df.empty:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Basic EDA
        st.subheader("📈 Closing Price Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig)

        # Feature Engineering
        df = df.copy()
        df['Date'] = df.index
        df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

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
        st.write(f"**R² Score**: {r2_score(y_test, y_pred):.4f}")
        st.write(f"**RMSE**: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

        # Prediction Plot
        st.subheader("🧟 Predicted vs Actual Closing Prices")
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        st.pyplot(plt)

        st.markdown("---")
        st.image("assets/gifs/zombie_line.gif", use_column_width=True)

    else:
        st.warning("Please upload a dataset or fetch a valid ticker first.")
