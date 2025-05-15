import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# 🎮 GTA VI Neon Theme Styling
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

    # Banner image (check that the path is correct)
    try:
        st.image("assets/gifs/gta_banner.gif", use_container_width=True)
    except Exception:
        st.warning("⚠️ Banner image not found. Please make sure 'assets/gifs/gta_banner.gif' exists.")


# 🎮 GTA VI Theme Main Function
def gaming_app():
    import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# 🕶️ GTA VI Neon Theme Styling
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

    try:
        st.image("assets/gifs/gta_banner.gif", use_column_width=True)
    except:
        st.warning("⚠️ Could not load banner image. Check if 'gta_banner.gif' exists in assets/gifs.")

# 🎮 GTA VI Theme Main App
def gaming_app():
    apply_gta_vi_theme()
    st.title("💸 GTA VI Theme: Cluster Heist (K-Means)")

    st.markdown("**Vice City meets Finance** — Find hidden clusters in your financial empire.")

    data_source = st.radio("💾 Choose your data source", ("Upload CSV (Kragle)", "Yahoo Finance"))

    df = None
    if data_source == "Upload CSV (Kragle)":
        uploaded = st.file_uploader("Upload financial CSV file", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("💾 File uploaded successfully.")
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")
    else:
        tickers = st.text_input("📈 Enter stock tickers (comma separated)", value="AAPL,MSFT,TSLA,NVDA")
        if st.button("🚦 Download Stock Data"):
            prices = {}
            for symbol in tickers.split(','):
                symbol = symbol.strip().upper()
                try:
                    data = yf.download(symbol, period="6mo")
                    if not data.empty and 'Close' in data:
                        prices[symbol] = data['Close']
                        st.success(f"✅ {symbol} data fetched.")
                    else:
                        st.warning(f"⚠️ No valid 'Close' data for: {symbol}")
                except Exception as e:
                    st.warning(f"❌ Failed to fetch data for {symbol}: {e}")

            if prices:
                try:
                    df = pd.DataFrame(prices).dropna()
                    st.success("📊 Combined DataFrame created.")
                except Exception as e:
                    st.error(f"❌ Error creating DataFrame: {e}")
            else:
                st.error("❌ No valid stock data fetched. Please check your tickers.")

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

            try:
                st.image("assets/gifs/gta_footer.gif", use_column_width=True)
            except:
                st.warning("⚠️ Could not load footer image. Check if 'gta_footer.gif' exists in assets/gifs.")
        else:
            st.warning("⛔ Select at least 2 features to cluster.")
    else:
        st.warning("🧾 Upload data or fetch stock prices to start your cluster job.")
