import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run(data):
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        .gaming-header {
            color: #00ffea;
            font-family: 'Press Start 2P', cursive;
            font-size: 32px;
            text-shadow: 2px 2px 8px #ff00ff;
        }
        .gaming-subheader {
            color: #ff6f61;
            font-family: 'Press Start 2P', cursive;
            font-size: 18px;
        }
        </style>
        <h1 class="gaming-header">🎮 Gaming Theme - K-Means Clustering 🎮</h1>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <p class="gaming-subheader">
        Experience clustering with pixel art vibes! Use K-Means to find patterns in your financial data.
        </p>
        """, unsafe_allow_html=True)

    # Pixel art gaming GIF - vibrant, high quality
    st.image("https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif", width=220)

    if data is None or data.empty:
        st.warning("🚨 Please upload a Kragle dataset or fetch Yahoo Finance data to start!")
        return

    st.write("### Dataset Snapshot")
    st.dataframe(data.head(10))

    # Filter numeric columns for clustering
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset requires at least 2 numeric columns for K-Means clustering.")
        return

    # Select features for clustering
    selected_features = st.multiselect(
        "Select 2 or more numeric features for clustering:",
        numeric_cols,
        default=numeric_cols[:2]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        return

    # Select number of clusters (K)
    n_clusters = st.slider("Choose number of clusters (K):", 2, 10, 3)

    # Prepare data for clustering
    df_cluster = data[selected_features].dropna()

    # Show info about dropped rows due to NaNs
    dropped_rows = data.shape[0] - df_cluster.shape[0]
    if dropped_rows > 0:
        st.info(f"Note: {dropped_rows} rows dropped due to missing values in selected features.")

    # Perform K-Means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df_cluster)
        df_cluster['Cluster'] = clusters
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return

    st.write(f"### Clustering Results (K = {n_clusters})")
    st.write(df_cluster.head())

    # Plot clustering results
    plt.figure(figsize=(9, 7))
    sns.set_style("darkgrid")

    # Scatter plot for first two selected features
    sns.scatterplot(
        x=df_cluster[selected_features[0]],
        y=df_cluster[selected_features[1]],
        hue=df_cluster['Cluster'],
        palette='bright',
        s=100,
        alpha=0.85,
        edgecolor='black'
    )

    plt.title("K-Means Clustering Visualization", fontsize=18, color='#00ffea', fontweight='bold')
    plt.xlabel(selected_features[0], fontsize=14)
    plt.ylabel(selected_features[1], fontsize=14)
    plt.legend(title='Cluster')
    plt.tight_layout()

    st.pyplot(plt.gcf())

    # Optional: Show cluster centers
    if st.checkbox("Show cluster centers"):
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
        st.write("Cluster Centers:")
        st.dataframe(centers.style.set_precision(3))

    # Extra gaming-style notification
    st.balloons()
    st.success("Cluster analysis complete! Ready for your next gaming move? 🎮")

