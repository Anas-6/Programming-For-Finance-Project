import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
                border: 1px solid #45a29e;
                transition: 0.3s;
            }

            .stButton>button:hover {
                background-color: #66fcf1;
                color: black;
            }

            .reportview-container {
                background: #0b0c10;
            }

        </style>
    """, unsafe_allow_html=True)

    try:
        st.image("assets/gifs/zombie_header.gif", use_container_width=True)
    except:
        st.warning("⚠️ Header GIF not found.")


# Main app logic
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
                st.success(f"✅ Fetched {len(df)} rows for {ticker}")
            else:
                st.error("❌ Failed to fetch data. Please check the ticker symbol.")
    else:
        uploaded_file = st.file_uploader("Upload your Kragle CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset uploaded successfully!")

    if df is not None and not df.empty:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Fix index for Plotly
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

        # Closing Price Plot
        st.subheader("📈 Closing Price Trend")
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='lime')))
            fig.update_layout(title='Closing Price Over Time', plot_bgcolor='black', paper_bgcolor='black', font_color='white')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"❌ Error plotting data: {e}")

        # Feature Engineering
        df = df.copy()
        df['Date'] = df.index
        df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

        X = df[['Date_ordinal']]
        y = df['Close']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📉 Model Performance")
        try:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.write(f"**R² Score**: {r2:.4f}")
            st.write(f"**RMSE**: {rmse:.4f}")
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")

        # 🎨 Random line color for prediction
        colors = ['deepskyblue', 'lime', 'magenta', 'yellow', 'red', 'orange', 'springgreen']
        line_color = random.choice(colors)

        # Prediction Plot
        st.subheader("🧟 Predicted vs Actual Closing Prices")
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Actual', color='orange')
        ax.plot(y_pred, label='Predicted', color=line_color)
        ax.set_title("Prediction vs Actual", fontsize=14, color='white')
        ax.legend()
        ax.set_facecolor('black')
        fig2.patch.set_facecolor('black')
        st.pyplot(fig2)

        st.markdown("---")
        try:
            st.image("assets/gifs/zombie_line.gif", use_container_width=True)
        except:
            st.warning("⚠️ Footer zombie GIF not found.")
    else:
        st.warning("⚠️ Please upload a dataset or fetch a valid ticker first.")
