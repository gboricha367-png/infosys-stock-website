import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# ----------------------------
# Sidebar Navigation
# ----------------------------
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "EDA", "Machine Learning Model"]
)

# ----------------------------
# Load Data (Infosys)
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

# ============================
# HOME PAGE
# ============================
if page == "Home":
    st.title("📈 Stock Price Prediction Dashboard")
    
    st.markdown("""
    ### Project Overview
    
    This project uses **Machine Learning (Linear Regression)** 
    to analyze historical stock price data and predict 
    the next 7 days closing prices.
    
    ### Features:
    - Exploratory Data Analysis (EDA)
    - Model Training & Evaluation
    - Actual vs Predicted Comparison
    - 7-Day Future Forecast
    - Download Forecast Option
    
    ---
    
    ⚠️ Predictions are generated using historical data 
    and are for academic demonstration purposes only.
    """)

# ============================
# EDA PAGE
# ============================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Raw Data")
    st.write(df.tail())

    st.subheader("Closing Price Trend")
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df.index, df["Close"])
    ax1.set_title("Historical Closing Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    st.pyplot(fig1)

    st.subheader("Volume Trend")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(df.index, df["Volume"])
    ax2.set_title("Trading Volume")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    st.pyplot(fig2)

# ============================
# MACHINE LEARNING PAGE
# ============================
elif page == "Machine Learning Model":

    st.title("🤖 Model Training & Prediction")

    # Features & Target
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Evaluation")
    st.write("Mean Squared Error:", round(mse, 4))
    st.write("R² Score:", round(r2, 4))

    # Actual vs Predicted Graph
    st.subheader("Actual vs Predicted Prices")
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(y_test.values, label="Actual")
    ax3.plot(y_pred, label="Predicted")
    ax3.legend()
    st.pyplot(fig3)

    # ----------------------------
    # 7-Day Future Prediction
    # ----------------------------
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

    # Download Button
    st.download_button(
        label="📥 Download 7-Day Forecast CSV",
        data=future_df.to_csv(index=False),
        file_name="7_day_forecast.csv",
        mime="text/csv"
    )

    # ----------------------------
    # Dashboard Metrics
    # ----------------------------
    st.subheader("📌 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Last Close", round(df['Close'].iloc[-1], 2))
    col2.metric("Highest Close", round(df['Close'].max(), 2))
    col3.metric("Lowest Close", round(df['Close'].min(), 2))

    # ----------------------------
    # Combined Graph
    # ----------------------------
    st.subheader("📈 Historical + 7-Day Forecast")

    fig4, ax4 = plt.subplots(figsize=(10,5))
    ax4.plot(df.index, df['Close'], label="Historical Close")
    ax4.plot(future_df["Date"], future_df["Predicted Close Price"], 
             linestyle="--", label="7-Day Forecast")
     col4.metric("Forecast Avg", round(future_df['Predicted Close Price'].mean(), 2))
    ax4.axvline(df.index[-1], linestyle=":", label="Today")

    ax4.legend()
    st.pyplot(fig4)

    st.markdown("""
    **Note:**  
    Linear Regression was used to predict future closing prices 
    based on historical features (Open, High, Low, Volume).  
    Predictions are static and meant for academic demonstration.
    """)
