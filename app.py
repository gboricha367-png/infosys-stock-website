import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Trading Dashboard", layout="wide")

# ---------------- STYLE ----------------

st.markdown("""
<style>
body {
    background-color:#ffe6f2;
}

.stMetric {
    background-color:#ffcce6;
    padding:20px;
    border-radius:12px;
    text-align:center;
}

h1, h2, h3 {
    color:#cc0066;
}

.sidebar .sidebar-content {
    background-color:#ffd6eb;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------

USERNAME = "admin"
PASSWORD = "infosys123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in == False:

    st.title("Infosys Stock Dashboard Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = yf.download("INFY.NS", period="5y", progress=False)
    df.dropna(inplace=True)
    return df

df = load_data()
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

# ---------------- INDICATORS ----------------

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["Daily Return"] = df["Close"].pct_change()

# ---------------- TITLE ----------------

st.title("Infosys Stock Analytics Dashboard")

# ---------------- METRICS ----------------

latest_price = float(df["Close"].iloc[-1])
previous_price = float(df["Close"].iloc[-2])

price_change = latest_price - previous_price
pct_change = (price_change / previous_price) * 100

high_52 = float(df["High"].tail(252).max())
low_52 = float(df["Low"].tail(252).min())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Price", round(latest_price,2), str(round(price_change,2)))
col2.metric("Daily % Change", str(round(pct_change,2)) + "%")
col3.metric("52 Week High", round(high_52,2))
col4.metric("52 Week Low", round(low_52,2))

st.divider()

# ---------------- PRICE CHART ----------------

st.subheader("Price Trend with Moving Averages")

fig1, ax1 = plt.subplots()

ax1.plot(df.index, df["Close"], label="Close Price")
ax1.plot(df.index, df["MA20"], label="20 Day MA")
ax1.plot(df.index, df["MA50"], label="50 Day MA")

ax1.legend()

plt.xticks(rotation=45)

st.pyplot(fig1)

st.divider()

# ---------------- VOLUME ----------------

st.subheader("Trading Volume")

fig2, ax2 = plt.subplots()

ax2.bar(df.index, df["Volume"])

plt.xticks(rotation=45)

st.pyplot(fig2)

st.divider()

# ---------------- RETURNS ----------------

st.subheader("Daily Returns")

fig3, ax3 = plt.subplots()

ax3.plot(df.index, df["Daily Return"])

plt.xticks(rotation=45)

st.pyplot(fig3)

st.divider()

# ---------------- MACHINE LEARNING ----------------

st.subheader("Machine Learning Price Prediction")

X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

col1, col2 = st.columns(2)

col1.metric("Model Error (MSE)", round(mse,2))
col2.metric("Model Accuracy (R²)", round(r2,4))

st.divider()

# ---------------- ACTUAL VS PREDICTED ----------------

st.subheader("Actual vs Predicted Price")

fig4, ax4 = plt.subplots()

ax4.plot(y_test.values, label="Actual Price")
ax4.plot(predictions, label="Predicted Price")

ax4.legend()

st.pyplot(fig4)

st.divider()

# ---------------- NEXT DAY PREDICTION ----------------

st.subheader("Next Day Closing Price Prediction")

next_day_features = X.tail(1)

next_day_prediction = model.predict(next_day_features)

predicted_price = float(next_day_prediction.flatten()[0])

st.metric("Predicted Next Closing Price", round(predicted_price,2))
