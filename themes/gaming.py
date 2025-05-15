import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import yfinance as yf

# GTA VI Gaming Theme Styling (Miami Neon + Pixel Art vibes)
def apply_gaming_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

            html, body, [class*="css"] {
                background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                color: #f8f8f2;
                font-family: 'Press Start 2P', cursive;
            }
            h1, h2, h3 {
                color: #ff2a68;
                text-shadow:
                    0 0 5px #ff2a68,
                    0 0 10px #ff2a68,
                    0 0 20px #ff2a68,
                    0 0 40px #ff2a68;
            }
            .stButton > button {
                background-color: #ff2a68;
                color: #1a1a1a;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 20px;
                transition: 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #ff4d8a;
                color: white;
            }
            .stTextInput>div>input {
                background-color: #1a1a1a;
                color: #f8f8f2;
                border: 1px solid #ff2a68;
                border-radius: 8px;
                padding: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Try to load a GTA VI themed pixel art GIF (place your gif in assets/gifs)
    try:
        st.image("assets/gifs/gta_banner.gif", use_container_width=True)
    except Exception:
        st.warning("⚠️ GTA VI themed GIF not found. Please add one to assets/gifs/gta_vi_pixel_art.gif")

# Main Gaming App with Logistic Regression
def gaming_app():
    apply_gaming_theme()

    st.title("💸 GTA VI Theme: Logistic Regression Heist")
    st.markdown("Vice City meets Finance — Predict your financial empire's future moves.")

    # Data source choice
    data_source = st.radio("💾 Choose your data source", ("Upload CSV (Kragle)", "Yahoo Finance"))

    df = None
    if data_source == "Upload CSV (Kragle)":
        uploaded_file = st.file_uploader("Upload your financial dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Kragle dataset loaded successfully!")
    else:
        tickers = st.text_input("📈 Enter stock tickers (comma separated)", value="AAPL,MSFT,TSLA")
        if st.button("🚦 Fetch Stock Data"):
            try:
                prices = {}
                for ticker in tickers.split(','):
                    ticker = ticker.strip().upper()
                    stock_data = yf.download(ticker, period="6mo", interval="1d")['Close']
                    prices[ticker] = stock_data
                df = pd.DataFrame(prices).dropna()
                st.success("✅ Yahoo Finance data fetched!")
            except Exception as e:
                st.error(f"❌ Failed to fetch stock data: {e}")

    if df is not None:
        st.subheader("🔎 Dataset Preview")
        st.dataframe(df.head())

        # Prepare data for Logistic Regression
        st.markdown("### Prepare Data for Logistic Regression")
        df = df.sort_index()  # sort by date ascending
        returns = df.pct_change().dropna()

        # Select ticker to model
        ticker_list = df.columns.tolist()
        selected_ticker = st.selectbox("Select stock ticker to build logistic regression model", ticker_list)

        if selected_ticker:
            data = pd.DataFrame()
            data['Return'] = returns[selected_ticker]
            data['Target'] = (data['Return'].shift(-1) > 0).astype(int)  # predict if next day return > 0
            data = data.dropna()

            st.write("Sample of prepared data:")
            st.dataframe(data.head())

            # Features and target
            X = data[['Return']]
            y = data['Target']

            # Train/test split slider
            test_size = st.slider("Select test data size (%)", 10, 50, 30)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, shuffle=False)

            if st.button("🔥 Run Logistic Regression Heist"):
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.success(f"Model trained! Accuracy on test data: {acc:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='reds')
                st.plotly_chart(fig)

                st.subheader("📊 Classification Report")
                st.text(classification_report(y_test, y_pred))

                coef_df = pd.DataFrame({'Feature': ['Return'], 'Coefficient': model.coef_[0]})
                fig2 = px.bar(coef_df, x='Feature', y='Coefficient', color='Coefficient', color_continuous_scale='reds')
                st.subheader("📈 Feature Importance")
                st.plotly_chart(fig2)

    else:
        st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")

