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
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd

    st.header("📊 Model Training & Prediction")

    # --- Features & Target ---
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Model Evaluation ---
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write("Mean Squared Error:", round(mse, 4))
    st.write("R² Score:", round(r2, 4))

    # --- Actual vs Predicted Graph ---
    st.subheader("Actual vs Predicted Prices")
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red")
    plt.xlabel("Time Index")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    st.pyplot(fig1)

    # --- 7-Day Future Prediction ---
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

    # Show table
    st.write(future_df)

    st.download_button(
    label="📥 Download 7-Day Forecast CSV",
    data=future_df.to_csv(index=False),
    file_name="7_day_forecast.csv",
    mime="text/csv"
)

    # --- Metrics Dashboard ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Close", round(df['Close'].iloc[-1], 2))
    col2.metric("Highest Close", round(df['Close'].max(), 2))
    col3.metric("Lowest Close", round(df['Close'].min(), 2))
    col4.metric("7-Day Forecast Avg", round(future_df['Predicted Close Price'].mean(), 2))

    # --- Combined Historical + Future Graph ---
    st.subheader("📈 Historical + 7-Day Forecast")
    plt.figure(figsize=(10,5))

    # Historical
    plt.plot(df.index, df['Close'], label="Historical Close", color="blue")
    # Forecast
    plt.plot(future_df["Date"], future_df["Predicted Close Price"], 
             label="7-Day Forecast", color="orange", linestyle="--")
    # Today line
    plt.axvline(df.index[-1], color="gray", linestyle=":", label="Today")

    # Labels & legend
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Historical + 7-Day Forecast")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # --- Academic Explanation (for viva) ---
    st.markdown("""
    **Note:**  
    - The Linear Regression model is trained on historical features (Open, High, Low, Volume).  
    - 7-Day forecast uses the last available feature set to predict forward.  
    - Predictions are static and meant for academic demonstration.  
    """)
    st.pyplot(fig4)

st.markdown("---")
st.markdown("Developed as part of Machine Learning Academic Project.")
