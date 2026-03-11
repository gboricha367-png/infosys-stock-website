import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Stock Dashboard", layout="wide")

# LOGIN SYSTEM

USERNAME = "admin"
PASSWORD = "infosys123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.title("Stock Analytics Platform")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error("Invalid username or password")

    st.stop()
```

# ---------------- STYLE ----------------

st.markdown("""

<style>
.stApp {
background-color:#f6f8fb;
}
</style>

""", unsafe_allow_html=True)

st.title("Infosys Stock Analytics Dashboard")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
data = yf.download("INFY.NS", period="5y", progress=False)
data.dropna(inplace=True)
return data

df = load_data()

# ---------------- INDICATORS ----------------

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

# ---------------- METRICS ----------------

latest_price = df["Close"].iloc[-1]
prev_price = df["Close"].iloc[-2]

change = latest_price - prev_price
pct = (change / prev_price) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(latest_price,2), f"{round(change,2)} ({round(pct,2)}%)")
col2.metric("Highest Price (5Y)", round(df["High"].max(),2))
col3.metric("Lowest Price (5Y)", round(df["Low"].min(),2))

st.divider()

# ---------------- PRICE CHART ----------------

st.subheader("Price Trend")

fig, ax = plt.subplots()

ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, df["MA20"], label="MA20")
ax.plot(df.index, df["MA50"], label="MA50")

ax.legend()

plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------- MACHINE LEARNING ----------------

st.subheader("Machine Learning Model")

X = df[['Open','High','Low','Volume']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

col1, col2 = st.columns(2)

col1.metric("Model MSE", round(mse,2))
col2.metric("Model R2 Score", round(r2,4))

# ---------------- NEXT DAY PREDICTION ----------------

st.subheader("Next Day Closing Price Prediction")

latest_data = X.tail(1)

next_day = model.predict(latest_data)

st.metric("Predicted Next Day Price", round(float(next_day[0]),2))

# ---------------- FORECAST ----------------

st.subheader("7 Day Forecast")

future_predictions = []

last_row = X.iloc[-1:].copy()

for i in range(7):
pred = model.predict(last_row)
future_predictions.append(float(pred))

future_dates = pd.date_range(
start=df.index[-1] + pd.Timedelta(days=1),
periods=7
)

forecast_df = pd.DataFrame({
"Date": future_dates,
"Predicted Price": future_predictions
})

st.dataframe(forecast_df)
