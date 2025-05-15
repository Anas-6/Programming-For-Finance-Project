import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import yfinance as yf

# 🌀 GTA VI Theme Styling
def apply_gta_vi_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Major+Mono+Display&display=swap');

            html, body {
                background-color: #0d0d0d;
                color: #BC13FE;
            }

            h1, h2, h3 {
                font-family: 'Major Mono Display', monospace;
                color: #00ffff;
                text-shadow: 2px 2px #ff66c4;
            }

            .stButton > button {
                background-color: #cc66ff;
                color: black;
                border-radius: 5px;
                padding: 0.5em 1em;
                font-weight: bold;
            }

            .stButton > button:hover {
                background-color: #00ffff;
                color: black;
            }

            .st-bx, .css-1dp5vir {
                background-color: #1a1a1a !important;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        st.image("assets/gifs/gta_header.gif", use_container_width=True)
    except Exception:
        st.warning("⚠️ Header GIF not found.")


# 🚓 GTA VI Logistic Regression Heist
def gaming_app():
    apply_gta_vi_theme()
    st.title("💸 GTA VI Theme: Logistic Regression Heist")
    st.markdown("**Vice City meets Finance** — Predict your financial empire's future moves.")

    data_source = st.radio("💾 Choose your data source", ("Upload CSV (Kragle)", "Yahoo Finance"))

    df = None

    if data_source == "Upload CSV (Kragle)":
        uploaded_file = st.file_uploader("📂 Upload your Kragle CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV uploaded successfully.")
    else:
        ticker = st.text_input("📈 Enter a stock ticker (e.g., TSLA)", value="TSLA")
        if st.button("🚦 Fetch Stock Data"):
            try:
                stock_data = yf.download(ticker.strip(), period="6mo", interval="1d")
                if 'Close' in stock_data.columns:
                    df = stock_data.copy()
                    st.success(f"✅ Data fetched for {ticker.upper()}")
                else:
                    st.error("❌ 'Close' price data not found.")
            except Exception as e:
                st.error(f"❌ Failed to fetch stock data: {e}")

    if df is not None and not df.empty:
        st.subheader("🔎 Vice City Data Preview")
        st.dataframe(df.tail())

        # 🎯 Feature Engineering
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df.dropna(inplace=True)

        # 📊 Model Training
        features = ['Return', 'MA5', 'MA10']
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 📈 Results
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader(f"⚡ Accuracy: {acc:.4f}")
        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Plasma')
        st.plotly_chart(fig_cm)

        st.subheader("🎮 Feature Impact")
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
        fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', color='Coefficient', template="plotly_dark")
        st.plotly_chart(fig_coef)

        try:
            st.image("assets/gifs/gta_footer.gif", use_container_width=True)
        except Exception:
            st.warning("⚠️ Footer GIF not found.")
    else:
        st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")
