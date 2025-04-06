import os
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="AI Indian Stock Analysis",
    page_icon="📈",
    layout="wide"
)

st.markdown('<h1 class="main-header">🚀 AI Indian Stock Analysis Chatbot</h1>', unsafe_allow_html=True)

def get_stock_data(symbol,period="1y"):
    stock = yf.Ticker(symbol + ".NS")
    data = stock.history(period=period)
    
    if data.empty:
        print("No Stock Data found!")
        return None
    
    data = data[['Close']]
    return data

def prepare_data(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data["Close"].iloc[i:i+time_steps].values)
        y.append(data["Close"].iloc[i+time_steps])
    
    return np.array(X), np.array(y)

def train_model(data):
    X,y = prepare_data(data)
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression()
    model.fit(x_train.reshape(x_train.shape[0], -1),y_train)
    
    return model

def predict_future(model,data):
    last_10_days = data[-10:].values.reshape(1, -1)
    predicted_price = model.predict(last_10_days)[0]

    return predicted_price


def get_stock_price(symbol: str):
    """Fetch Indian stock price from Yahoo Finance with error handling"""
    try:
        stock = yf.Ticker(symbol + ".NS")
        if not stock.info or stock.info is None:
            return {"symbol": symbol, "price": "N/A", "error": "Invalid stock symbol or data not available."}
        latest_price = stock.info.get('regularMarketPrice', "N/A")
        if latest_price == "N/A":
            return {"symbol": symbol, "price": "N/A", "error": "Price data not available."}
        return {"symbol": symbol, "price": latest_price}
    except Exception as e:
        return {"symbol": symbol, "price": "N/A", "error": f"Unable to fetch stock data: {str(e)}"}

def get_stock_history(symbol: str):
    """Fetch historical stock data for plotting"""
    try:
        stock = yf.Ticker(symbol + ".NS")
        data = stock.history(period="1mo") 
        if data.empty:
            return None
        return data
    except Exception as e:
        return None

def analyze_stock_trends(symbol: str):
    prompt = f"Analyze the stock trend for {symbol} (Indian NSE stock) and give a market forecast."
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in trend analysis: {str(e)}"

def stock_sentiment_analysis(symbol: str):
    prompt = f"Perform sentiment analysis on recent news related to {symbol} (Indian NSE stock)."
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"


# ------------------ FRONTEND (STREAMLIT) ------------------
symbol = st.text_input("Enter NSE Stock Symbol:", placeholder="e.g., RELIANCE, TATAMOTORS, INFY", help="Enter the stock symbol without .NS")



col1, col2 , _ = st.columns([1.2, 1.2, 1.6])


with col1:
    if st.button("🔍 Analyze Stock"):
        if not symbol:
            st.error("Please enter a stock symbol")
        else:
            with st.spinner("Analyzing stock data..."):
                price_data = get_stock_price(symbol.upper().strip())

                if "error" in price_data:
                    st.error(price_data["error"])
                else:
                    st.success(f"Successfully retrieved data for {symbol}")
                    st.metric(label="Current Stock Price", value=f"₹{price_data['price']:,.2f}")

                    # **📊 Graph Plot**
                    stock_data = get_stock_history(symbol)
                    if stock_data is not None:
                        st.subheader("📉 Stock Price Trend (Last 30 Days)")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(stock_data.index, stock_data["Close"], marker="o", linestyle="-", color="blue", label="Closing Price")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price (₹)")
                        ax.set_title(f"{symbol} Stock Price Trend (Last 30 Days)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                    else:
                        st.error("Stock price history not available.")

                    tab1, tab2 = st.tabs(["📈 Market Analysis", "📰 Sentiment Analysis"])
                    
                    with tab1:
                        with st.spinner("Generating market analysis..."):
                            trend_analysis = analyze_stock_trends(symbol)
                            st.write(trend_analysis)
                    
                    with tab2:
                        with st.spinner("Analyzing market sentiment..."):
                            sentiment = stock_sentiment_analysis(symbol)
                            st.write(sentiment)
with col2:        
    if st.button("📊 Predict Future Price"):
        with st.spinner("Predicting stock price..."):
            stock_data = get_stock_data(symbol)
            
            if stock_data is not None:
                model = train_model(stock_data)
                prediction = predict_future(model, stock_data)
                
                st.success(f"📈 Predicted Next Day Price: ₹{prediction:.2f}")
            else:
                st.error("Stock data not available.")

st.markdown('<div class="credits">Made with ❤️ by Dhruv Bafna (Jain)</div>', unsafe_allow_html=True)
