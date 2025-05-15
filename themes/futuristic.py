import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

# Futuristic Theme CSS
def apply_futuristic_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');

            body {
                background-color: #0f0f0f;
                color: #33ffcc;
            }

            h1, h2, h3 {
                font-family: 'Orbitron', sans-serif;
                color: #00ffff;
                text-shadow: 0 0 10px #00ffff;
            }

            .stButton>button {
                background-color: #1a1a1a;
                border: 1px solid #33ffcc;
                color: white;
                transition: all 0.3s ease-in-out;
            }

            .stButton>button:hover {
                background-color: #00ffff;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)
    st.image("assets/gifs/futuristic_header.gif", use_column_width=True)


# Main Futuristic Theme Logic
def run_futuristic_theme():
    apply_futuristic_theme()
    st.title("🚀 Futuristic Theme: Stock Movement Prediction (Up/Down)")

    st.markdown("Predict whether a stock will go **up or down** tomorrow using Logistic Regression.")

    ticker = st.text_input("Enter a stock ticker (e.g., TSLA):", value="TSLA")
    if st.button("Fetch Data"):
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            st.error("No data found. Try a different ticker.")
            return
        st.success(f"Data fetched successfully for {ticker}")
        st.dataframe(df.tail())

        # Feature Engineering
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df = df.dropna()

        features = ['Return', 'MA5', 'MA10']
        X = df[features]
        y = df['Target']

        # Model Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader(f"⚡ Accuracy: {acc:.4f}")
        st.plotly_chart(
            px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Viridis')
        )

        st.subheader("📈 Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_[0]
        })
        fig = px.bar(coef_df, x='Feature', y='Coefficient', color='Coefficient', title="Logistic Coefficients")
        st.plotly_chart(fig)

        st.image("assets/gifs/futuristic_footer.gif", use_column_width=True)
