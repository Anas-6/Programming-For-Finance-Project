import streamlit as st
from themes import zombie, futuristic, game_of_thrones, gaming
from PIL import Image
from show_welcome import show_welcome



# Set Streamlit page configuration
st.set_page_config(
    page_title="FinVerse - Financial Realms Reimagined",
    page_icon=" 🎨 ",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
        }
        .css-18e3th9 {
            padding: 2rem 1rem;
        }
        .css-1d391kg {
            padding: 1rem;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Add project logo
col1, col2 = st.columns([1, 6])
with col1:
   st.image("https://github.com/Anas-6/Programming-For-Finance-Project/blob/main/assets/images/finverse_logo.png", width=90)


with col2:
    st.title("FinVerse: Financial Realms Reimagined")

# Sidebar Navigation
st.sidebar.title("🧭 Navigate FinVerse")
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
        <sub>Developed by Team FinVerse | FAST NUCES | Spring 2025 - Programming for Finance</sub>
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
    got.got_app()
elif page == "🎮 Gaming World":
    gaming.gaming_app()

