import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Dashboard", layout="wide")

# ---------------- LOGIN ----------------

USERNAME = "admin"
PASSWORD = "infosys123"

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"] == False:
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = yf.download("INFY.NS", period="5y", progress=False)
    df.dropna(inplace=True)
    return df

df = load_data()

# ---------------- INDICATORS ----------------

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

# ---------------- TITLE ----------------

st.title("Infosys Stock Market Dashboard")

# ---------------- METRICS ----------------

latest_price = float(df["Close"].iloc[-1])
previous_price = float(df["Close"].iloc[-2])

change = latest_price - previous_price
pct_change = (change / previous_price) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(latest_price,2), str(round(change,2)))
col2.metric("5Y High", round(float(df["High"].max()),2))
col3.metric("5Y Low", round(float(df["Low"].min()),2))

# ---------------- PRICE CHART ----------------

st.subheader("Price Trend")

fig, ax = plt.subplots()

ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, df["MA20"], label="20 Day MA")
ax.plot(df.index, df["MA50"], label="50 Day MA")

ax.legend()
plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------- MACHINE LEARNING ----------------

st.subheader("Machine Learning Model")

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

col1.metric("Mean Squared Error", round(mse,2))
col2.metric("R2 Score", round(r2,4))

# ---------------- ACTUAL VS PREDICTED ----------------

st.subheader("Actual vs Predicted Prices")

fig2, ax2 = plt.subplots()

ax2.plot(y_test.values, label="Actual")
ax2.plot(predictions, label="Predicted")

ax2.legend()

st.pyplot(fig2)

# ---------------- NEXT DAY PREDICTION ----------------

st.subheader("Next Day Prediction")

next_day_features = X.tail(1)
next_day_prediction = model.predict(next_day_features)

predicted_price = float(next_day_prediction.flatten()[0])

st.metric("Predicted Next Closing Price", round(predicted_price, 2))
