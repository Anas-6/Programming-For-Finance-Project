import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# GTA VI Neon Theme Styling
def apply_gta_vi_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Major+Mono+Display&display=swap');

            html, body {
                background-color: #0d0d0d;
                color: #ff66c4;
            }

            h1, h2, h3 {
                font-family: 'Major Mono Display', monospace;
                color: #00ffff;
                text-shadow: 2px 2px #ff66c4;
            }

            .stButton > button {
                background-color: #ff66c4;
                color: black;
                border-radius: 5px;
                padding: 0.5em 1em;
                font-weight: bold;
            }

            .stButton > button:hover {
                background-color: #00ffff;
                color: black;
            }

            .css-1v0mbdj, .st-bx, .css-1dp5vir {
                background-color: #1a1a1a !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.image("assets/gifs/gta_banner.gif", use_container_width=True)


def run_gta_vi_theme():
    apply_gta_vi_theme()
    st.title("💸 GTA VI Theme: Logistic Regression Heist")

    st.markdown("**Vice City meets Finance** — Predict your financial empire's future moves.")

    data_source = st.radio("💾 Choose your data source", ("Upload CSV (Kragle)", "Yahoo Finance"))

    df = None
    target_column = None

    if data_source == "Upload CSV (Kragle)":
        uploaded = st.file_uploader("Upload financial CSV file", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("💾 File uploaded successfully.")
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")
    else:
        tickers = st.text_input("📈 Enter stock tickers (comma separated)", value="AAPL,MSFT,TSLA")
        if st.button("🚦 Download Stock Data"):
            try:
                prices = {}
                for symbol in tickers.split(','):
                    sym = symbol.strip().upper()
                    data = yf.download(sym, period="6mo")
                    if data.empty or 'Close' not in data:
                        st.warning(f"⚠️ No data found for {sym}")
                    else:
                        prices[sym] = data['Close']
                if prices:
                    df = pd.DataFrame(prices).dropna()
                    st.success("✅ Stock prices loaded.")
                else:
                    st.error("❌ No valid data fetched for given tickers.")
            except Exception as e:
                st.error(f"❌ Failed to fetch stock data: {e}")

    if df is not None and not df.empty:
        st.subheader("🔎 Preview of Vice City Data")
        st.dataframe(df.head())

        st.subheader("🔥 Heatmap of Your Empire")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="magma", ax=ax)
        st.pyplot(fig)

        # --- Logistic Regression Section ---

        st.subheader("🚓 Logistic Regression Heist")

        # Select target column for classification
        target_column = st.selectbox("🎯 Select the target column (binary classification required)", options=df.columns)

        # Check binary target
        if target_column:
            unique_vals = df[target_column].dropna().unique()
            if len(unique_vals) != 2:
                st.warning(f"⚠️ Target column '{target_column}' is not binary. Please choose or prepare a binary target column.")
                # Optionally create binary target
                if st.checkbox("Create binary target from numeric column (e.g., above/below median)"):
                    median_val = df[target_column].median()
                    df[target_column + "_binary"] = (df[target_column] > median_val).astype(int)
                    target_column = target_column + "_binary"
                    st.success(f"✅ Binary target '{target_column}' created using median threshold.")
                else:
                    st.stop()

            # Feature selection
            features = st.multiselect("🔍 Select features for prediction", [col for col in df.columns if col != target_column], default=[col for col in df.columns if col != target_column][:3])

            if len(features) < 1:
                st.warning("⛔ Select at least one feature.")
                st.stop()

            # Prepare data
            X = df[features]
            y = df[target_column]

            # Encode target if needed
            if y.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Handle missing values
            if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                st.info("ℹ️ Dropping rows with missing values...")
                df_clean = pd.concat([X, pd.Series(y, name=target_column)], axis=1).dropna()
                X = df_clean[features]
                y = df_clean[target_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train/test split
            test_size = st.slider("Test set size (%)", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42)

            st.markdown(f"Training logistic regression with {len(X_train)} samples and testing on {len(X_test)} samples...")

            if st.button("🚓 Run Logistic Regression Heist"):
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.success(f"🏆 Accuracy: {acc:.4f}")

                st.markdown("### 📋 Classification Report")
                st.text(classification_report(y_test, y_pred))

                # Confusion matrix plot
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_title('Confusion Matrix')
                st.pyplot(fig_cm)

                st.image("assets/gifs/gta_footer.gif", use_container_width=True)
        else:
            st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")
    else:
        st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")
