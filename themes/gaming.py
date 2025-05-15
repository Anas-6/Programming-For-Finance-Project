import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Gaming Theme Visual Styling (bright colors + pixel art vibes) ---
def apply_gaming_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
            
            html, body, [class*="css"]  {
                background-color: #f0f8ff;
                color: #2c3e50;
                font-family: 'Press Start 2P', cursive;
            }

            h1, h2, h3 {
                color: #e67e22;
                text-shadow: 2px 2px #d35400;
            }

            .stButton > button {
                background-color: #e67e22;
                color: white;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 20px;
            }

            .stButton > button:hover {
                background-color: #d35400;
                color: white;
            }

            .stFileUpload>div>div {
                background-color: #dff9fb;
                border-radius: 10px;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    st.image("assets/gifs/pixel_art_gaming.gif", use_container_width=True)

# --- Main Gaming Theme Application ---
def gaming_app():
    apply_gaming_theme()

    st.title("🎮 Gaming Theme: Logistic Regression Heist")
    st.markdown("**Vice City meets Finance — Predict your financial empire's future moves.**")

    # Data Source Choice
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
                    stock_data = yf.download(ticker, period="6mo")['Close']
                    prices[ticker] = stock_data
                df = pd.DataFrame(prices).dropna()
                st.success("✅ Yahoo Finance data fetched!")
            except Exception as e:
                st.error(f"❌ Failed to fetch stock data: {e}")

    if df is not None:
        st.subheader("🔎 Dataset Preview")
        st.dataframe(df.head())

        # Prepare Data for Logistic Regression
        st.markdown("### Prepare Data for Logistic Regression")

        # We'll create a binary target to predict if the price will go UP (1) or DOWN (0) next day
        df = df.sort_index()  # Ensure data is sorted by date ascending
        df_returns = df.pct_change().dropna()

        st.markdown("**Note:** We create a target variable to predict if stock price will go UP the next day (1) or not (0).")

        # Let user select one ticker to model
        ticker_list = df.columns.tolist()
        selected_ticker = st.selectbox("Select stock ticker to build logistic regression model", ticker_list)

        if selected_ticker:
            data = pd.DataFrame()
            data['Return'] = df_returns[selected_ticker]
            data['Target'] = (data['Return'].shift(-1) > 0).astype(int)  # Predict if next day return > 0
            
            data = data.dropna()

            st.write("Sample of prepared data:")
            st.dataframe(data.head())

            # Feature and Target Split
            X = data[['Return']].values
            y = data['Target'].values

            # Train-test split
            test_size = st.slider("Select test data size (%)", 10, 50, 30)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, shuffle=False)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if st.button("🔥 Train Logistic Regression Model"):
                model = LogisticRegression()
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Model trained! Accuracy on test data: {acc:.2f}")

                # Confusion matrix plot
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

                # Classification report
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                # Show coefficients
                coef = model.coef_[0][0]
                st.write(f"Logistic Regression coefficient for Return feature: {coef:.4f}")

    else:
        st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")

