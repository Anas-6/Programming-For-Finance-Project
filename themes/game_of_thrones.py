import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Apply Game of Thrones Theme
def apply_got_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');
            html, body, [class*="css"]  {
                font-family: 'MedievalSharp', cursive;
                background-image: url("https://i.imgur.com/Bazj9K3.jpg");
                background-size: cover;
                background-attachment: fixed;
                color: #f8f1e5;
            }
            h1, h2, h3 {
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
            .reportview-container .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    st.image("assets/gifs/got_header.gif", use_container_width=True)

# Main App Function
def got_app():
    apply_got_theme()
    st.title("🐉 Game of Thrones Theme: Cluster the Kingdoms (K-Means)")
    st.markdown("Group stocks or financial features into clusters using K-Means Clustering.")

    data_source = st.radio("Choose Data Source:", ["Yahoo Finance", "Upload CSV (Kaggle)"])

    df = None

    # Yahoo Finance Option
    if data_source == "Yahoo Finance":
        tickers = st.text_input("Enter tickers (comma-separated, e.g., AAPL, MSFT, TSLA):")
        if st.button("Fetch Data"):
            tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            data_dict = {}

            for ticker in tickers_list:
                try:
                    data = yf.Ticker(ticker).history(period="6mo")
                    if not data.empty:
                        data_dict[ticker] = data["Close"]
                    else:
                        st.warning(f"No data for {ticker}")
                except Exception as e:
                    st.error(f"Error with {ticker}: {e}")

            if data_dict:
                df = pd.DataFrame(data_dict).dropna()
                st.success("Fetched data successfully.")
            else:
                st.error("No valid data fetched.")

    # Kaggle/CSV Option
    elif data_source == "Upload CSV (Kaggle)":
        uploaded = st.file_uploader("Upload Kragle Financial CSV File", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("CSV data loaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if df is not None and not df.empty:
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("🔥 Feature Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("⚔️ K-Means Clustering")

        try:
            num_clusters = st.slider("Select number of clusters", 2, 6, 3)
            features = st.multiselect("Select features for clustering", df.columns.tolist(), default=df.columns[:2].tolist())

            if len(features) >= 2:
                X = df[features]
                X_scaled = StandardScaler().fit_transform(X)

                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(X_scaled)
                df['Cluster'] = kmeans.labels_

                st.success("Clustering complete!")

                fig = px.scatter(df, x=features[0], y=features[1], color='Cluster',
                                 title="🏰 Realm Clusters", template="plotly_dark")
                st.plotly_chart(fig)
            else:
                st.warning("Please select at least 2 features.")
        except Exception as e:
            st.error(f"Error during clustering: {e}")

        st.image("assets/gifs/got_footer.gif", use_container_width=True)
    else:
        st.info("Please load data to proceed.")
