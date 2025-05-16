import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --- Apply Zombie Theme ---
def apply_zombie_theme():
    # Inject Google Fonts via HTML
    components.html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');

        html, body, [class*="css"].creepster_regular  {
            font-family: "Creepster", system-ui;
            background-color: #0d0d0d;
            color: #39ff14;
        }

        .stButton>button {
            background-color: #1c1c1c;
            color: #39ff14;
            border: 1px solid #39ff14;
        }

        .stButton>button:hover {
            background-color: #39ff14;
            color: black;
        }

        .stRadio>div {
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 8px;
            color: #ffffff;
        }

        h1, h2, h3, h4 {
            font-family: 'Creepster', cursive !important;
            color: #39ff14;
            text-shadow: 0 0 10px #39ff14;
        }
        </style>
    """, height=0)

    try:
        st.image("assets/gifs/zombie_header.gif", use_container_width=True)
    except:
        st.warning("Zombie GIF not found at assets/gifs/zombie_header.gif")


# Color palette for line chart
plot_colors = ['#39ff14', '#ff073a', '#8a2be2', '#00ffff', '#ff8c00']

# --- Main Zombie Theme App ---
def zombie_app():
    apply_zombie_theme()
    st.title("🧟‍♂️ Zombie Mode: Financial Forecast with Linear Regression")

    # Data source selection
    st.subheader("☠️ Choose Your Data Source")
    data_source = st.radio("Select:", ["Yahoo Finance", "Upload Kragle CSV"])

    df = None
    if data_source == "Yahoo Finance":
        ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):", value="AAPL")
        if st.button("Fetch Yahoo Data"):
            df = yf.download(ticker, period="1y")
            if not df.empty:
                st.success(f"Fetched {len(df)} rows for {ticker}")
            else:
                st.error("Failed to fetch data. Try another ticker.")
    else:
        uploaded_file = st.file_uploader("Upload Kragle Financial CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Kragle dataset uploaded!")

    if df is not None and not df.empty:
        # Ensure 'Date' is datetime index if available
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        st.subheader("🧾 Data Preview")
        st.dataframe(df.head())

        # Line chart of closing price
        if 'Close' in df.columns:
            st.subheader("📉 Closing Price Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines',
                                     name='Close Price',
                                     line=dict(color=np.random.choice(plot_colors))))
            st.plotly_chart(fig)
        else:
            st.warning("'Close' column not found in dataset.")

        # Feature selection
        st.subheader("🧠 Select Features for Linear Regression")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Close' in numeric_cols:
            numeric_cols.remove('Close')

        selected_features = st.multiselect("Select independent variables (X):", numeric_cols)

        if selected_features:
            if 'Close' not in df.columns:
                st.error("Target column 'Close' not found.")
                return

            df = df.dropna(subset=selected_features + ['Close'])

            X = df[selected_features]
            y = df['Close']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model evaluation
            st.subheader("📊 Model Evaluation Metrics")
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.success(f"R² Score: {r2:.4f}")
            st.success(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

            # Predicted vs Actual chart
            st.subheader("🧟 Predicted vs Actual 'Close' Prices")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual', line=dict(color=plot_colors[0])))
            fig2.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted', line=dict(color=plot_colors[1])))
            st.plotly_chart(fig2)

            try:
                st.image("assets/gifs/zombie_line.gif", use_container_width=True)
            except:
                st.warning("Zombie footer GIF not found.")
        else:
            st.warning("Please select at least one feature for Linear Regression.")
    else:
        st.info("Upload a dataset or fetch a ticker to get started!")
