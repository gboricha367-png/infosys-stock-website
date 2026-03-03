import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Infosys Stock Prediction", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("INFY.NS.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

data = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Home",
     "Historical Analysis",
     "Technical Indicators",
     "Model Prediction",
     "Download Results",
     "About"]
)

if page == "Home":
    st.title("Stock Price Prediction Using Machine Learning")
    st.subheader("Infosys Ltd")
    st.write("This web application predicts stock prices using Linear Regression.")

elif page == "Historical Analysis":
    st.title("Historical Closing Price Trend")
    fig = plt.figure(figsize=(12,6))
    plt.plot(data['Close'])
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    st.pyplot(fig)

elif page == "Technical Indicators":
    st.title("50-Day Moving Average")
    data['MA50'] = data['Close'].rolling(50).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label="Close Price")
    plt.plot(data['MA50'], label="50-Day MA")
    plt.legend()
    st.pyplot(fig)

elif page == "Model Prediction":
    st.title("Linear Regression Prediction")

    X = data[['Open','High','Low','Volume']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    fig = plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    st.pyplot(fig)

    st.write("R2 Score:", r2_score(y_test, predictions))
    st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))

elif page == "Download Results":
    st.title("Download Prediction Results")

    X = data[['Open','High','Low','Volume']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": predictions
    })

    st.download_button(
        "Download CSV",
        results.to_csv(index=False),
        "prediction_results.csv",
        "text/csv"
    )

elif page == "About":
    st.title("About Project")
    st.write("""
    Project: Stock Price Prediction  
    Company: Infosys Ltd  
    Model Used: Linear Regression  
    Developed by: Grishma Boricha  
    """)
