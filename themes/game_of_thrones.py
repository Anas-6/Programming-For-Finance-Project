import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Game of Thrones Theme CSS
def apply_got_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');

            h1, h2, h3 {
                font-family: 'MedievalSharp', cursive;
                color: #e63946;
            }

            .stButton>button {
                background-color: #343a40;
                border: 1px solid #e63946;
                color: white;
                font-weight: bold;
            }

            .stButton>button:hover {
                background-color: #e63946;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        st.image("assets/gifs/got_banner.gif", use_container_width=True)
    except Exception:
        st.warning("⚠️ GOT banner GIF not found.")

# Main App Logic
def got_app():
    apply_got_theme()
    st.title("🐉 Game of Thrones Theme: Cluster the Kingdoms (K-Means)")

    st.markdown("Group stocks or financial features into clusters using K-Means Clustering.")

    data_source = st.radio("Choose Data Source:", ("Yahoo Finance", "Upload CSV"))
    df = None

    if data_source == "Yahoo Finance":
        tickers = st.text_input("Enter multiple tickers separated by commas (e.g., AAPL,GOOG,MSFT):")
        if st.button("Fetch Data"):
            with st.spinner("📡 Fetching stock data..."):
                try:
                    data = {}
                    for ticker in tickers.split(','):
                        ticker = ticker.strip().upper()
                        stock_data = yf.download(ticker, period="6mo")
                        if not stock_data.empty and 'Close' in stock_data.columns:
                            data[ticker] = stock_data['Close']
                    if data:
                        df = pd.DataFrame(data).dropna()
                        st.success(f"✅ Fetched {df.shape[0]} rows of data for {len(data)} tickers.")
                    else:
                        st.error("❌ No valid data fetched. Check tickers or try again later.")
                except Exception as e:
                    st.error(f"❌ Failed to fetch data: {e}")
    else:
        uploaded = st.file_uploader("Upload Kragle Financial CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ File uploaded.")
            except Exception as e:
                st.error(f"❌ Failed to read uploaded file: {e}")

    if df is not None and not df.empty:
        st.subheader("📊 Preview of Data")
        st.dataframe(df.head())

        st.subheader("🔥 Feature Correlation")
        try:
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, ax=ax, cmap="coolwarm")
            st.pyplot(fig)
        except:
            st.warning("Unable to generate correlation heatmap.")

        st.subheader("⚔️ K-Means Clustering")
        try:
            num_clusters = st.slider("Select number of clusters", 2, 6, 3)
            features = st.multiselect("Select features to use", df.columns.tolist(), default=df.columns[:2].tolist())
            if len(features) >= 2:
                X = df[features]
                X_scaled = StandardScaler().fit_transform(X)

                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(X_scaled)
                df['Cluster'] = kmeans.labels_

                st.success("🏰 Clustering complete!")

                fig = px.scatter(df, x=features[0], y=features[1], color='Cluster',
                                 title="🏰 Clusters of the Realm", template="plotly_dark")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ Select at least 2 features for clustering.")
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")

        try:
            st.image("assets/gifs/got_header.gif", use_container_width=True)
        except Exception:
            st.warning("⚠️ GOT footer GIF not found.")
    else:
        st.info("ℹ️ Please load data to begin.")
