import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Stock Dashboard", layout="wide")

st.title("📊 Infosys Ltd Stock Market Analysis")

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
# SIDEBAR NAVIGATION
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
    This dashboard provides stock analysis and price prediction for Infosys Ltd.
    
    It includes:
    - Historical Data
    - Price Visualization
    - Linear Regression Model
    - 7-Day Forecast
    """)

# ----------------------------
# DATA OVERVIEW
# ----------------------------

elif page == "Data Overview":
    st.subheader("Raw Stock Data")
    st.dataframe(df.tail())

# ----------------------------
# VISUALIZATION
# ----------------------------

elif page == "Visualizations":

    st.subheader("Closing Price Over Time")
    fig1 = plt.figure()
    plt.plot(df.index, df["Close"])
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Trading Volume Over Time")
    fig2 = plt.figure()
    plt.plot(df.index, df["Volume"])
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

    st.subheader("Actual vs Predicted Prices")
    fig3 = plt.figure()
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    st.pyplot(fig3)

    # ----------------------------
    # 7-DAY FUTURE PREDICTION
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

    st.dataframe(future_df)

    fig4 = plt.figure()
    plt.plot(future_df["Date"], future_df["Predicted Close Price"])
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    avg_forecast = future_df["Predicted Close Price"].mean()
    st.metric("Average 7-Day Forecast Price", round(avg_forecast, 2))
