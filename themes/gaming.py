import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px
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

# 🚓 GTA VI Theme Main Logic with Logistic Regression
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
            # Expect user to provide a target column in uploaded file
            possible_targets = df.columns.tolist()
            target_column = st.selectbox("Select target column (binary classification)", possible_targets)
    else:
        tickers = st.text_input("📈 Enter stock tickers (comma separated)", value="AAPL,MSFT,TSLA")
        if st.button("🚦 Download Stock Data"):
            try:
                prices_list = []
                for symbol in tickers.split(','):
                    symbol = symbol.strip()
                    data = yf.download(symbol, period="6mo")['Close']
                    if data.empty:
                        st.warning(f"No data fetched for {symbol}")
                    else:
                        prices_list.append(data.rename(symbol))
                if prices_list:
                    df = pd.concat(prices_list, axis=1).dropna()
                    st.success("✅ Stock prices loaded.")

                    # Create target column based on first ticker price movement (up=1, down=0)
                    first_ticker = tickers.split(',')[0].strip()
                    df['Target'] = (df[first_ticker].shift(-1) > df[first_ticker]).astype(int)
                    target_column = 'Target'
                else:
                    st.error("❌ Failed to fetch any stock data.")
            except Exception as e:
                st.error(f"❌ Failed to fetch stock data: {e}")

    if df is not None and not df.empty and target_column:
        st.subheader("🔎 Preview of Vice City Data")
        st.dataframe(df.head())

        st.subheader("🔥 Heatmap of Your Empire")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="magma", ax=ax)
        st.pyplot(fig)

        st.subheader("🚓 Logistic Regression Heist")

        features = df.columns.drop(target_column).tolist()
        selected_features = st.multiselect("🎯 Select features for logistic regression", features, default=features[:2])

        if len(selected_features) >= 1:
            X = df[selected_features]
            y = df[target_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            test_size = st.slider("Select test set size (percentage)", 10, 50, 30) / 100.0

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, shuffle=False)

            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.markdown(f"### 🎯 Model Accuracy: {accuracy*100:.2f}%")

            st.subheader("📊 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)

            # Plot prediction probabilities if 2 features selected
            if len(selected_features) == 2:
                probas = model.predict_proba(X_test)
                prob_df = pd.DataFrame({
                    selected_features[0]: X_test[:, 0],
                    selected_features[1]: X_test[:, 1],
                    'Probability of Class 1': probas[:, 1],
                    'Predicted Class': y_pred
                })

                fig_scatter = px.scatter(prob_df, x=selected_features[0], y=selected_features[1],
                                         color='Predicted Class', size='Probability of Class 1',
                                         color_continuous_scale=px.colors.sequential.Plasma,
                                         title="🌴 GTA VI Logistic Regression Prediction Map",
                                         template="plotly_dark")
                st.plotly_chart(fig_scatter)

            st.image("assets/gifs/gta_footer.gif", use_container_width=True)

        else:
            st.warning("⛔ Select at least 1 feature for logistic regression.")
    else:
        st.warning("🧾 Upload data or fetch stock prices to start your logistic regression heist.")

if __name__ == "__main__":
    gaming_app()
