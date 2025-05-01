import os
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import plotly.graph_objects as go

# Download NLTK data
nltk.download('vader_lexicon')

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="AI Indian Stock Analysis",
    page_icon="📈",
    layout="wide"
)

st.sidebar.header("Chart Settings")
time_period = st.sidebar.selectbox("Select Time Period for Chart", ["1mo", "3mo", "6mo", "1y"], index=2)

# Function to fetch stock data
@st.cache_data
def get_stock_data(symbol, period="1mo"):
    stock = yf.Ticker(symbol + ".NS")
    hist = stock.history(period=period)
    info = stock.info
    return hist, info


def stock_sentiment_analysis(symbol):
    prompt = f"""
    Perform a concise sentiment analysis (50-100 words) on recent news for the Indian NSE stock {symbol}. 
    Summarize the sentiment as Positive, Negative, or Neutral, and provide a brief explanation based on market perception, 
    earnings, or events. Include a sentiment score between -1 (very negative) and 1 (very positive) in the format 'Score: X.X'.
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"

# Function to parse sentiment score xzsdew3
def parse_sentiment_score(analysis_text):
    try:
        score_match = re.search(r'Score:\s*([-]?[0-1]\.[0-9])', analysis_text, re.IGNORECASE)
        if score_match:
            return float(score_match.group(1))
        
        analysis_lower = analysis_text.lower()
        if 'positive' in analysis_lower:
            return 0.5
        elif 'negative' in analysis_lower:
            return -0.5
        elif 'neutral' in analysis_lower:
            return 0.0
        
        sia = SentimentIntensityAnalyzer()
        vader_score = sia.polarity_scores(analysis_text)['compound']
        return vader_score
    except:
        return 0.0

# Function to visualize sentiment
def visualize_sentiment(sentiment_text, symbol):
    score = parse_sentiment_score(sentiment_text)
    st.subheader("Market Sentiment")
    st.write(f"**Sentiment for {symbol}:**")
    st.write(sentiment_text)
    st.write(f"**Parsed Sentiment Score:** {score:.2f}")
    color = 'green' if score > 0 else 'red' if score < 0 else 'gray'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment for {symbol}", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0.5], 'color': "lightgray"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
def get_ai_analysis(symbol, stock_info):
    prompt = f"""
    You are a professional stock market analyst. Provide a concise analysis (100-150 words) of the stock {symbol} based on the following data:
    - Current Price: {stock_info.get('regularMarketPrice', 'N/A')}
    - P/E Ratio: {stock_info.get('trailingPE', 'N/A')}
    - Market Cap: {stock_info.get('marketCap', 'N/A')}
    - 52-Week High: {stock_info.get('fiftyTwoWeekHigh', 'N/A')}
    - 52-Week Low: {stock_info.get('fiftyTwoWeekLow', 'N/A')}
    - Book Value: {stock_info.get('bookValue', 'N/A')}
    - Dividend Yield: {stock_info.get('dividendYield', 'N/A')*100 if stock_info.get('dividendYield') else 'N/A'}%
    Provide insights on its valuation, growth potential, and risks.
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Unable to generate AI analysis: {str(e)}"

# Main app
st.title("AI-Powered Indian Stock Analysis 📊")
st.markdown("Analyze Indian stocks with real-time data, AI insights, and sentiment analysis.")

# Stock symbol input on main page with unique key
stock_symbol = st.text_input("Enter Stock Symbol", key="stock_symbol_input")

# Fetch and display data only if a valid stock symbol is provided
if stock_symbol:
    try:
        hist, stock_info = get_stock_data(stock_symbol, time_period)
        if hist.empty:
            st.error("No historical data available for this stock. Try a different symbol or time period.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}. Please try another symbol .")
        st.stop()


    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Key Metrics")
        metrics = {
            "Current Price": f"₹{stock_info.get('regularMarketPrice', 'N/A')}",
            "P/E Ratio": stock_info.get('trailingPE', 'N/A'),
            "Market Cap": f"₹{stock_info.get('marketCap', 'N/A')/1e7:.2f} Cr",
            "52-Week High": f"₹{stock_info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52-Week Low": f"₹{stock_info.get('fiftyTwoWeekLow', 'N/A')}",
            "Book Value": f"₹{stock_info.get('bookValue', 'N/A')}",
            "Dividend Yield": f"{stock_info.get('dividendYield', 0)*100:.2f}%"
        }
        for key, value in metrics.items():
            st.metric(key, value)

        # Sentiment Analysis with Visualization
        sentiment_text = stock_sentiment_analysis(stock_symbol)
        visualize_sentiment(sentiment_text, stock_symbol)

    # Stock Chart and Analysis (col1)
    with col1:
        # Monthly Stock Price Chart
        st.subheader("Stock Price Chart")
        fig, ax = plt.subplots()
        ax.plot(hist.index, hist['Close'], label="Close Price", color="blue")
        ax.set_title(f"{stock_symbol} Price ({time_period})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Monthly Profit/Loss Table
        st.subheader("Monthly Profit/Loss")
        monthly_closes = hist['Close'].resample('ME').last()
        monthly_returns = monthly_closes.pct_change() * 100
        monthly_df = pd.DataFrame({
            "Month": monthly_closes.index.strftime("%Y-%m"),
            "Close Price (₹)": monthly_closes.values,
            "Return (%)": monthly_returns.values
        })
        monthly_df = monthly_df.dropna(subset=['Return (%)'])
        if not monthly_df.empty:
            st.table(monthly_df)
        else:
            st.info(f"No monthly returns available for {time_period}. Try a longer time period (e.g., 3mo or 6mo).")

        # AI Stock Analysis
        st.subheader("AI Stock Analysis")
        analysis = get_ai_analysis(stock_symbol, stock_info)
        st.write(analysis)

# Footer
st.markdown("---")
st.markdown('<div class="credits">Made with ❤️ by Dhruv Bafna (Jain)</div>', unsafe_allow_html=True)
