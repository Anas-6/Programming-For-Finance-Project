![finverse_logo](https://github.com/user-attachments/assets/c1de0c52-0421-4aea-8985-0e43b9eb6fb4)

# 💸 FinVerse: Financial Realms Reimagined

[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](https://programming-for-finance-project-dhzjkefzlidhmgdng9es4r.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)


FinVerse is a multi-themed financial machine learning application built using **Streamlit**, creatively integrating **Kragle financial datasets** and **Yahoo Finance data**. It allows users to explore, visualize, and model financial data using machine learning—**with a twist of imagination**.

> 🌐 **Live Streamlit App**: [Click to Launch](https://programming-for-finance-project-dhzjkefzlidhmgdng9es4r.streamlit.app/)

---

## 🤝 Acknowledgments
Instructor: Dr. Usama Arshad

Course: AF3005 – Programming for Finance

University: FAST NUCES, Spring 2025## 🎯 Objective

To develop a collaborative, interactive, and creative machine learning app using real-world financial data — with each group member contributing a **uniquely themed module**.

---

## 🧑‍🤝‍🧑 Group Members & Themes

| Name                        | Theme             | Model                  |
|------------------------|-------------------|------------------------|
| Matin Khan      (22i-9842)  | Zombie Realm      | Linear Regression      |
| Mohid Ayaz      (22i-9960)  | Futuristic Realm  | Logistic Regression    |
| Muhammad Anas   (22i-9808)  | Game of Thrones   | K-Means Clustering     |
| Muhammad Anas   (22i-9808)  | Gaming Realm      | Logistic Regression    |

---


## 📦 Features

- 🔄 Upload Data from  Kaggle or fetch stock data via **Yahoo Finance API**
- 📈 Apply **Machine Learning** models (Linear Regression, Logistic Regression, K-Means, Decision Tree)
- 🧑‍🎨 Select from **4 visually distinct themes**:
  - 🧟 Zombie Realm
  - 🚀 Futuristic Realm
  - 🐉 Game of Thrones Realm
  - 🎮 Gaming Realm
- 📊 Visualize correlations, trends, and clusters with **Plotly**, **Matplotlib**, and **Seaborn**
- 🌐 Fully deployed on **Streamlit Cloud**
- 📂 Complete open-source repo with all themes and modules



## 🧪 Machine Learning Models Used

- 🔵 **Linear Regression** (Predict Stock Price)
- 🟢 **Logistic Regression** (Predict Movement Direction)
- 🟣 **K-Means Clustering** (Group Stocks or Features)
- 🟠 **Logistic Regression** (Classify financial behaviors)

---

## 🖥️ Tech Stack

- **Frontend/UI**: Streamlit
- **ML & Data**: scikit-learn, pandas, numpy, yfinance
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud
- **Version Control**: GitHub

---

## 🗂️ Project Structure

├──Programming-For-Finance-Project
 ├── 📁 assets/
  ├── gifs/
  ├──images/
 ├── 📁 themes/
  ├── zombie.py
  ├── futuristic.py
  ├── game_of_thrones.py
  ├── gaming.py

 ├── app.py              
 ├── show_welcome.py     
 ├── requirements.txt 


---

## 🔌 Data Sources

- 📁 **Kragle Financial Datasets** 
- 🌐 **Yahoo Finance API** (via `yfinance`)

---

## Video



https://github.com/user-attachments/assets/be790f26-d373-4380-8852-fca7bbf9c681


## 🚀 Run Locally

```
# Clone the repository
git clone https://github.com/your-username/finverse.git
cd finverse

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

