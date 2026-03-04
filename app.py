import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Infosys Stock Dashboard", layout="wide")

st.title("📊 Infosys Ltd Stock Market Analysis Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------

@st.cache_data
def load_data():
    df = yf.download("INFY.NS", period="5y")
    df.dropna(inplace=True)
    return df

df = load_data()

if df.empty:
    st.error("No data available. Please try again later.")
    st.stop()

# ----------------------------
# SIDEBAR
# ----------------------------

page = st.sidebar.selectbox(
    "Select Section",
    ["Home", "Data Overview", "Visualizations", "Machine Learning Model"]
)

# ----------------------------
# HOME
# ----------------------------

if page == "Home":
    st.subheader("About Infosys Ltd")
    st.write("""
    This dashboard performs historical stock analysis and price prediction 
    for Infosys Ltd using Linear Regression.
    """)

# ----------------------------
# DATA OVERVIEW
# ----------------------------

elif page == "Data Overview":
    st.subheader("Recent Stock Data")
    st.dataframe(df.tail())

# ----------------------------
# VISUALIZATIONS
# ----------------------------

elif page == "Visualizations":

    st.subheader("Closing Price Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(df.index, df["Close"])
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Trading Volume Over Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index, df["Volume"])
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ----------------------------
# MACHINE LEARNING MODEL
# ----------------------------

elif page == "Machine Learning Model":

    st.subheader("Model Training & Evaluation")

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

    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", round(mse, 2))
    col2.metric("R2 Score", round(r2, 4))

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Prices")
    fig3, ax3 = plt.subplots()
    ax3.plot(y_test.values, label="Actual")
    ax3.plot(y_pred, label="Predicted")
    ax3.legend()
    st.pyplot(fig3)

    # ----------------------------
    # SAFE 7-DAY FORECAST
    # ----------------------------

    st.subheader("📅 7-Day Future Prediction")

    last_row = X.iloc[-1:].copy()
    future_predictions = []

    for _ in range(7):
        pred_array = model.predict(last_row)
        pred_value = np.squeeze(pred_array)   # <-- SAFELY extract scalar
        future_predictions.append(float(pred_value))

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=7
    )

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close Price": future_predictions
    })

    st.dataframe(future_df)

    fig4, ax4 = plt.subplots()
    ax4.plot(future_df["Date"], future_df["Predicted Close Price"])
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.metric(
        "Average 7-Day Forecast Price",
        round(future_df["Predicted Close Price"].mean(), 2)
    )
