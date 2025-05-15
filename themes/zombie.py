import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# Apply eerie zombie theme
def apply_zombie_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Butcherman&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Nosifer&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Macondo+Swash+Caps&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Metal+Mania&display=swap');

            html, body, [class*="css"] {
                background-color: #0b0c10;
                color: #c5c6c7;
                font-family: 'Butcherman', cursive;
            }

            h1, h2, h3, h4 {
                font-family: 'Butcherman', cursive;
                color: #66fcf1;
            }

            .stButton>button {
                background-color: #1f2833;
                color: white;
                font-family: 'Metal Mania', cursive;
            }

            .stRadio > div {
                font-family: 'Macondo Swash Caps', cursive;
            }

            .stMarkdown {
                font-family: 'Nosifer', cursive;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        st.image("assets/gifs/zombie_header.gif", use_container_width=True)
    except:
        st.warning("⚠️ Zombie header GIF not found.")


# 🧟‍♂️ Zombie Logic
def zombie_app():
    apply_zombie_theme()
    st.title("💀 Zombie Theme: Stock Price Prediction")

    st.markdown("Enter a stock ticker or upload your dataset to begin.")

    data_source = st.radio("Choose data source:", ("Yahoo Finance", "Upload CSV"))

    df = None
    if data_source == "Yahoo Finance":
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
        if st.button("Fetch Data"):
            df = yf.download(ticker, period="1y")
            if not df.empty:
                st.success(f"Fetched {len(df)} rows for {ticker}")
                df.reset_index(inplace=True)
    else:
        uploaded_file = st.file_uploader("Upload your Kragle CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")

    if df is not None and not df.empty:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Date processing
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df['Date']) if 'Date' in df else pd.to_datetime(df['index'], errors='coerce')
        df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

        # Default Feature Options
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Date_ordinal' not in numeric_cols:
            numeric_cols.append('Date_ordinal')
        target = st.selectbox("🎯 Select Target Variable", options=numeric_cols, index=numeric_cols.index("Close") if "Close" in numeric_cols else 0)
        default_features = [col for col in numeric_cols if col != target]

        features = st.multiselect("🧠 Select Feature Variables for Prediction", options=default_features, default=default_features[:2])

        if features:
            X = df[features]
            y = df[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Training
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Performance
            st.subheader("📉 Model Performance")
            try:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                rmse = mean_squared_error(y_test, y_pred) ** 0.5
            r2 = r2_score(y_test, y_pred)

            st.write(f"**R² Score**: {r2:.4f}")
            st.write(f"**RMSE**: {rmse:.4f}")

            # Prediction Plot
            st.subheader("🧟 Predicted vs Actual Closing Prices")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test.values, label='Actual', color='red')
            ax.plot(y_pred, label='Predicted', color='lime')
            ax.legend()
            st.pyplot(fig)

            # Closing Price Trend with Colors
            st.subheader("📈 Closing Price Trend")
            fig2 = go.Figure()
            palette = ['#66fcf1', '#ff4c4c', '#c5c6c7', '#45a29e', '#ffe100', '#8b0000']
            for i, feature in enumerate(features):
                fig2.add_trace(go.Scatter(x=df['Date'], y=df[feature], mode='lines', name=feature, line=dict(color=palette[i % len(palette)])))
            st.plotly_chart(fig2)

            try:
                st.image("assets/gifs/zombie_line.gif", use_container_width=True)
            except:
                st.warning("⚠️ Zombie footer GIF not found.")
        else:
            st.warning("☠️ Please select at least one feature variable.")
    else:
        st.info("☠️ Please fetch a ticker or upload a valid CSV to begin.")
