import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Dashboard", layout="wide")

# LOGIN

USERNAME = "admin"
PASSWORD = "infosys123"

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
st.title("Login")

```
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username == USERNAME and password == PASSWORD:
        st.session_state["logged_in"] = True
        st.rerun()
    else:
        st.error("Wrong username or password")

st.stop()
```

# DASHBOARD TITLE

st.title("Infosys Stock Analytics Dashboard")

# LOAD DATA

@st.cache_data
def load_data():
df = yf.download("INFY.NS", period="5y", progress=False)
df.dropna(inplace=True)
return df

df = load_data()

# INDICATORS

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

# METRICS

latest_price = df["Close"].iloc[-1]
previous_price = df["Close"].iloc[-2]

change = latest_price - previous_price
pct_change = (change / previous_price) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(latest_price,2), str(round(change,2)) + " (" + str(round(pct_change,2)) + "%)")
col2.metric("5Y High", round(df["High"].max(),2))
col3.metric("5Y Low", round(df["Low"].min(),2))

st.subheader("Price Chart")

fig, ax = plt.subplots()

ax.plot(df.index, df["Close"], label="Close")
ax.plot(df.index, df["MA20"], label="MA20")
ax.plot(df.index, df["MA50"], label="MA50")

ax.legend()

st.pyplot(fig)

# MACHINE LEARNING

st.subheader("Machine Learning Prediction")

X = df[["Open","High","Low","Volume"]]
y = df["Close"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

col1, col2 = st.columns(2)

col1.metric("Model MSE", round(mse,2))
col2.metric("Model R2 Score", round(r2,4))

st.subheader("Next Day Prediction")

next_day = model.predict(X.tail(1))

st.metric("Predicted Price", round(float(next_day[0]),2))
