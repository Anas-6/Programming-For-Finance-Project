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
def apply_game_of_thrones_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');

            body {
                background-color: #1a1a1a;
                color: #f8f1e5;
            }

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

    st.image("assets/gifs/got_header.gif", use_column_width=True)


# ✅ Main function to run GOT theme logic (renamed as requested)
def got_app():
    apply_game_of_thrones_theme()
    st.title("🐉 Game of Thrones Theme: Cluster the Kingdoms (K-Means)")

    st.markdown("Group stocks or financial features into clusters using K-Means Clustering.")

    data_source = st.radio("Choose Data Source:", ("Yahoo Finance", "Upload CSV"))

    df = None
    if data_source == "Yahoo Finance":
        tickers_input = st.text_input("Enter multiple tickers separated by commas (e.g., AAPL,GOOG,MSFT):")
        if st.button("Fetch Data"):
            if not tickers_input:
                st.warning("Please enter at least one ticker.")
            else:
                tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
                data = {}

                for ticker in tickers:
                    try:
                        stock_data = yf.download(ticker, period="6mo")['Close']
                        if isinstance(stock_data, pd.Series) and not stock_data.empty:
                            data[ticker] = stock_data
                        else:
                            st.warning(f"No valid data for ticker: {ticker}")
                    except Exception as e:
                        st.warning(f"Error fetching {ticker}: {e}")

                if data:
                    df = pd.DataFrame(data).dropna()
                    st.success(f"Fetched {df.shape[0]} rows of data for {len(data)} tickers.")
                else:
                    st.error("Failed to fetch data for any valid ticker.")
    else:
        uploaded = st.file_uploader("Upload Kragle Financial CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("File uploaded successfully.")
            except:
                st.error("Error reading the uploaded CSV file.")

    if df is not None and not df.empty:
        st.subheader("📊 Preview of Data")
        st.dataframe(df.head())

        st.subheader("🔥 Feature Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax, cmap="coolwarm")
        st.pyplot(fig)

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

                st.success("Clustering complete!")

                fig = px.scatter(df, x=features[0], y=features[1], color='Cluster',
                                 title="🏰 Clusters of the Realm", template="plotly_dark")
                st.plotly_chart(fig)
            else:
                st.warning("Select at least 2 features for clustering.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        st.image("assets/gifs/got_footer.gif", use_column_width=True)
    else:
        st.warning("Please load data to continue.")
