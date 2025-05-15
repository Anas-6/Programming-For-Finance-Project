import streamlit as st
from PIL import Image

def show_welcome():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>🌐 Welcome to FinVerse</h1>
            <h3>Explore the Financial Realms of Machine Learning</h3>
        </div>
        """, unsafe_allow_html=True
    )

    st.image("assets/images/finverse_banner.png", use_container_width=True)

    st.markdown("---")

    st.markdown("""
    **FinVerse** is an interactive financial machine learning application built by a team of students from FAST NUCES.  
    This app demonstrates how financial data can be explored and modeled using:
    
    - 📈 Real-time stock data from **Yahoo Finance**
    - 📊 Machine learning models: Linear Regression, Logistic Regression, and K-Means Clustering
    - 🎨 Custom visual themes: Zombie, Futuristic, Game of Thrones, and GTA VI

    <br>

    **📌 How to use this app:**
    - Use the sidebar to navigate to each themed section
    - Upload your dataset or choose a stock ticker
    - Run ML models and visualize predictions

    """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("👉 Choose a theme from the sidebar to begin!")
