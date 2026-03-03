import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Stock Prediction", layout="wide")

st.title("📈 Infosys Stock Price Prediction using Machine Learning")

# Sidebar Navigation
page = st.sidebar.selectbox(
    "Select Section",
    ["Project Overview", "Data Visualization", "Machine Learning Model"]
)

# Load Data
df = pd.read_csv("INFY.NS.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

if page == "Project Overview":
    st.header("Project Overview")
    st.write("""
    This project predicts the closing stock price of Infosys Ltd.
    using Machine Learning (Linear Regression).

    Steps followed:
    - Data Collection
    - Data Cleaning
    - Feature Selection
    - Model Training
    - Model Evaluation
    - Visualization
    """)

elif page == "Data Visualization":
    st.header("Historical Data Analysis")

    st.subheader("Closing Price Trend")
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(df['Close'])
    plt.title("Closing Price Over Time")
    st.pyplot(fig1)

    st.subheader("50-Day Moving Average")
    df['MA50'] = df['Close'].rolling(50).mean()
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(df['Close'], label="Close")
    plt.plot(df['MA50'], label="MA50")
    plt.legend()
    st.pyplot(fig2)

elif page == "Machine Learning Model":
    st.header("Model Training & Prediction")

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write("Mean Squared Error:", mse)
    st.write("R2 Score:", r2)

    st.subheader("Actual vs Predicted Prices")
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    st.pyplot(fig3)

    # 7-Day Future Prediction
    st.subheader("📅 7-Day Future Prediction")

    last_row = X.tail(1).values
    future_predictions = []

    for i in range(7):
        next_pred = model.predict(last_row)[0]
        future_predictions.append(next_pred)

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=7
    )

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close Price": future_predictions
    })

    st.write(future_df)

    fig4 = plt.figure(figsize=(10,5))
    plt.plot(future_df["Date"], future_df["Predicted Close Price"])
    plt.xticks(rotation=45)
    st.pyplot(fig4)

st.markdown("---")
st.markdown("Developed as part of Machine Learning Academic Project.")
