import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

#  GTA VI Neon Theme Styling
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

    st.image("assets/gifs/gta_banner.gif", use_column_width=True)


# 🚓 GTA VI Theme Main Logic
def run_gta_vi_theme():
    apply_gta_vi_theme()
    st.title("💸 GTA VI Theme: Cluster Heist (K-Means)")

    st.markdown("**Vice City meets Finance** — Find hidden clusters in your financial empire.")

    data_source = st.radio("💾 Choose your data source", ("Upload CSV (Kragle)", "Yahoo Finance"))

    df = None
    if data_source == "Upload CSV (Kragle)":
        uploaded = st.file_uploader("Upload financial CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success("💾 File uploaded successfully.")
    else:
        tickers = st.text_input("📈 Enter stock tickers (comma separated)", value="AAPL,MSFT,TSLA")
        if st.button("🚦 Download Stock Data"):
            try:
                prices = {}
                for symbol in tickers.split(','):
                    prices[symbol.strip()] = yf.download(symbol.strip(), period="6mo")['Close']
                df = pd.DataFrame(prices).dropna()
                st.success("✅ Stock prices loaded.")
            except:
                st.error("❌ Failed to fetch stock data.")

    if df is not None and not df.empty:
        st.subheader("🔎 Preview of Vice City Data")
        st.dataframe(df.head())

        st.subheader("🔥 Heatmap of Your Empire")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="magma", ax=ax)
        st.pyplot(fig)

        st.subheader("🚓 K-Means Cluster Job")

        num_clusters = st.slider("Number of clusters", 2, 6, 3)
        selected_features = st.multiselect("🎯 Select features for clustering", df.columns.tolist(), default=df.columns[:2].tolist())

        if len(selected_features) >= 2:
            data = df[selected_features]
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            model = KMeans(n_clusters=num_clusters, random_state=42)
            labels = model.fit_predict(data_scaled)
            df["Cluster"] = labels

            st.success("💰 Clustering complete. You've unlocked new territories.")

            fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color="Cluster",
                             title="🌴 GTA VI Clusters: Vice City Turf Map",
                             template="plotly_dark", color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig)

            st.image("assets/gifs/gta_footer.gif", use_column_width=True)
        else:
            st.warning("⛔ Select at least 2 features to cluster.")
    else:
        st.warning("🧾 Upload data or fetch stock prices to start your cluster job.")
