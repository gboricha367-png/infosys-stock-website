import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Stock Dashboard", layout="wide")

# -----------------------

# LOGIN SYSTEM

# -----------------------

USERNAME = "admin"
PASSWORD = "infosys123"

if "logged_in" not in st.session_state:
st.session_state.logged_in = False

if not st.session_state.logged_in:

```
st.title("Stock Analytics Platform")

st.write("Login to access the dashboard")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):

    if username == USERNAME and password == PASSWORD:
        st.session_state.logged_in = True
        st.rerun()
    else:
        st.error("Invalid credentials")

st.stop()
```

# -----------------------

# MODERN UI STYLE

# -----------------------

st.markdown("""

<style>

.stApp {
background-color:#f6f8fb;
}

h1,h2,h3 {
color:#1e293b;
}

.block-container {
padding-top:2rem;
}

.metric-card{
background:white;
padding:20px;
border-radius:10px;
box-shadow:0px 2px 6px rgba(0,0,0,0.05);
}

</style>

""", unsafe_allow_html=True)

st.title("Infosys Stock Analytics Dashboard")

# -----------------------

# LOAD DATA

# -----------------------

@st.cache_data
def load_data():

```
df = yf.download("INFY.NS", period="5y", progress=False)
df.dropna(inplace=True)

return df
```

df = load_data()

# indicators

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

# -----------------------

# STOCK SUMMARY

# -----------------------

latest_close = float(df["Close"].iloc[-1])
previous_close = float(df["Close"].iloc[-2])

change = latest_close - previous_close
pct_change = (change / previous_close) * 100

high_52 = df["High"].rolling(252).max().iloc[-1]
low_52 = df["Low"].rolling(252).min().iloc[-1]

volume = df["Volume"].iloc[-1]

# -----------------------

# DASHBOARD METRICS

# -----------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Price", round(latest_close,2), f"{round(change,2)} ({round(pct_change,2)}%)")
col2.metric("52 Week High", round(high_52,2))
col3.metric("52 Week Low", round(low_52,2))
col4.metric("Volume", int(volume))
col5.metric("Market Trend","Bullish" if change>0 else "Bearish")

st.markdown("---")

# -----------------------

# TABS LIKE TRADING APPS

# -----------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Overview",
"Technical Charts",
"Data",
"Machine Learning Model"
]
)

# =============================

# OVERVIEW TAB

# =============================

with tab1:

```
col1, col2 = st.columns([2,1])

with col1:

    st.subheader("Price Movement")

    fig, ax = plt.subplots()

    ax.plot(df.index, df["Close"], label="Close Price")
    ax.plot(df.index, df["MA20"], label="MA20")
    ax.plot(df.index, df["MA50"], label="MA50")

    ax.legend()

    plt.xticks(rotation=45)

    st.pyplot(fig)

with col2:

    st.subheader("Company Information")

    st.write("""
```

Infosys Limited is an Indian multinational IT company that provides
business consulting, IT services and outsourcing.

This dashboard performs stock analysis using historical market
data and applies machine learning techniques for price prediction.
""")

# =============================

# TECHNICAL CHARTS

# =============================

with tab2:

```
st.subheader("Moving Average Analysis")

fig2, ax2 = plt.subplots()

ax2.plot(df.index, df["Close"], label="Close")
ax2.plot(df.index, df["MA50"], label="50 Day MA")
ax2.plot(df.index, df["MA200"], label="200 Day MA")

ax2.legend()

plt.xticks(rotation=45)

st.pyplot(fig2)

st.subheader("Volume Analysis")

fig3, ax3 = plt.subplots()

ax3.bar(df.index, df["Volume"])

plt.xticks(rotation=45)

st.pyplot(fig3)
```

# =============================

# DATA TAB

# =============================

with tab3:

```
st.subheader("Historical Data")

st.dataframe(df.tail(50))
```

# =============================

# MACHINE LEARNING

# =============================

with tab4:

```
st.subheader("Stock Price Prediction Model")

X = df[['Open','High','Low','Volume']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,shuffle=False
)

model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

col1,col2 = st.columns(2)

col1.metric("Model Error (MSE)",round(mse,2))
col2.metric("Model Accuracy (R2)",round(r2,4))

st.subheader("Actual vs Predicted")

fig4, ax4 = plt.subplots()

ax4.plot(y_test.values,label="Actual")
ax4.plot(predictions,label="Predicted")

ax4.legend()

st.pyplot(fig4)

st.subheader("Next Day Prediction")

latest_data = X.tail(1)

next_day = model.predict(latest_data)

st.metric("Predicted Closing Price", round(float(next_day[0]),2))

st.subheader("7 Day Forecast")

last_row = X.iloc[-1:].copy()

future_predictions = []

for i in range(7):

    pred = model.predict(last_row)
    future_predictions.append(float(pred))

future_dates = pd.date_range(
    start=df.index[-1]+pd.Timedelta(days=1),
    periods=7
)

forecast_df = pd.DataFrame({
"Date":future_dates,
"Predicted Price":future_predictions
})

st.dataframe(forecast_df)

fig5, ax5 = plt.subplots()

ax5.plot(forecast_df["Date"],forecast_df["Predicted Price"])

plt.xticks(rotation=45)

st.pyplot(fig5)
```
