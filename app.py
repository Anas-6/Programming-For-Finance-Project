import streamlit as st
from themes import zombie, futuristic, game_of_thrones, gaming
from PIL import Image
from show_welcome import show_welcome

# Set Streamlit page configuration
st.set_page_config(
    page_title="FinVerse - Financial Realms Reimagined",
    page_icon="🎨",
    layout="wide"
)

# Global Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Orbitron', sans-serif;
            background-color: #0e1117;
            color: #f0f2f6;
        }

        .block-container {
            padding-top: 2rem;
        }

        .stTitle > h1 {
            font-size: 3rem;
            color: #61dafb;
            margin-bottom: 0;
        }

        .stRadio label {
            display: block;
            padding: 0.75rem 1.5rem;
            margin: 0.25rem 0;
            background-color: #1c1f26;
            border-radius: 10px;
            cursor: pointer;
            transition: 0.3s;
            border: 1px solid transparent;
        }

        .stRadio label:hover {
            border: 1px solid #61dafb;
            background-color: #20232a;
        }

        .stRadio div[role="radiogroup"] > label[data-selected="true"] {
            background-color: #61dafb;
            color: #0e1117;
        }

        .sidebar .sidebar-content {
            background-color: #0e1117;
        }

        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Layout with Logo and Title
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/images/finverse_logo.png", use_container_width=True)
with col2:
    st.title("FinVerse: Financial Realms Reimagined")

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigate FinVerse")
page = st.sidebar.radio("Select a Theme", [
    "🏠 Welcome",
    "🧟 Zombie Realm",
    "🚀 Futuristic Zone",
    "🐉 Thrones Territory",
    "🎮 Gaming World"
])

# Footer credit
st.markdown("""
    <hr>
    <center>
        <sub>🚀 Developed by Team FinVerse | FAST NUCES | Spring 2025 - Programming for Finance</sub>
    </center>
""", unsafe_allow_html=True)

# Render selected theme
if page == "🏠 Welcome":
    show_welcome()
elif page == "🧟 Zombie Realm":
    zombie.zombie_app()
elif page == "🚀 Futuristic Zone":
    futuristic.futuristic_app()
elif page == "🐉 Thrones Territory":
    game_of_thrones.got_app()
elif page == "🎮 Gaming World":
    gaming.gaming_app()
