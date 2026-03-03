import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Stock Prediction", layout="wide")

st.title("📈 Infosys Stock Price Prediction using Machine Learning")

# Load Data
df = pd.read_csv("INFY.NS.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

st.subheader("Raw Dataset")
st.write(df.tail())

# Historical Closing Price
st.subheader("Historical Closing Price Trend")
fig1 = plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
st.pyplot(fig1)

# Moving Average
st.subheader("50-Day Moving Average")
df['MA50'] = df['Close'].rolling(50).mean()
fig2 = plt.figure(figsize=(10,5))
plt.plot(df['Close'], label="Close")
plt.plot(df['MA50'], label="MA50")
plt.legend()
st.pyplot(fig2)

# Machine Learning Section
st.subheader("Machine Learning Model")

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Model Evaluation")
st.write("Mean Squared Error:", mse)
st.write("R2 Score:", r2)

# Actual vs Predicted
st.subheader("Actual vs Predicted Prices")
fig3 = plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
st.pyplot(fig3)

st.markdown("---")
st.markdown("Developed as part of Machine Learning Academic Project.")
