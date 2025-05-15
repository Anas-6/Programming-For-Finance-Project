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

# Main Logic for GTA VI Logistic Regression
def gaming_app():
    apply_gta_vi_theme()
    st.title("💸 GTA VI Theme: Logistic Regression Heist")
    st.markdown("**Vice City meets Finance** — Predict your financial empire's future moves.")

    data_source = st.radio("💾 Choose your data source", ("Upload CSV (Kragle)", "Yahoo Finance"))

    df = None
    target_column = None

    if data_source == "Upload CSV (Kragle)":
        uploaded = st.file_uploader("Upload financial CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success("💾 File uploaded successfully.")
            target_column = st.selectbox("Select the target column (binary)", options=df.columns)
    else:
        tickers = st.text_input("📈 Enter stock tickers (comma separated)", value="AAPL,MSFT,TSLA")
        if st.button("🚦 Download Stock Data"):
            try:
                prices = {}
                for symbol in tickers.split(','):
                    data = yf.download(symbol.strip(), period="6mo")['Close']
                    if data.empty:
                        st.warning(f"No data fetched for {symbol.strip()}")
                    else:
                        prices[symbol.strip()] = data
                if prices:
                    df = pd.DataFrame(prices).dropna()
                    st.success("✅ Stock prices loaded.")
                    # For logistic regression we need a binary target, create one:
                    # Let's create a 'Target' column that predicts if price will go up the next day (1) or not (0)
                    df['Target'] = (df.shift(-1) > df).astype(int).iloc[:,0]  # use first ticker for target
                    target_column = 'Target'
                else:
                    st.error("❌ Failed to fetch any stock data.")
            except Exception as e:
                st.error(f"❌ Failed to fetch stock data: {e}")

    if df is not None and not df.empty:
        st.subheader("🔎 Preview of Vice City Data")
        st.dataframe(df.head())

        st.subheader("🔥 Heatmap of Your Empire")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="magma", ax=ax)
        st.pyplot(fig)

        if target_column:
            st.subheader("🚓 Logistic Regression Heist")

            # Select features (excluding target)
            features = [col for col in df.columns if col != target_column]
            selected_features = st.multiselect("🎯 Select features for prediction", features, default=features[:3])

            if len(selected_features) >= 1:
                data = df[selected_features]
                target = df[target_column]

                # Clean data - drop NA rows
                data = data.dropna()
                target = target.loc[data.index]

                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.3, random_state=42)

                model = LogisticRegression(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.markdown(f"**Accuracy:** {acc:.2%}")

                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="magma", ax=ax)
                st.pyplot(fig)

                st.image("assets/gifs/gta_footer.gif", use_container_width=True)
            else:
                st.warning("⛔ Select at least 1 feature for logistic regression.")
        else:
            st.warning("⛔ Please select the target column (binary) to proceed.")
    else:
        st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")
