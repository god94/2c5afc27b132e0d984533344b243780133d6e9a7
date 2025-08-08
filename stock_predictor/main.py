import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse

def fetch_stock_data(ticker_symbol):
    """
    Fetches historical stock data for a given ticker symbol.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="5y")
    return hist

def fetch_news(ticker_symbol):
    """
    Fetches news for a given ticker symbol.
    """
    ticker = yf.Ticker(ticker_symbol)
    return ticker.news

def analyze_sentiment(news_list):
    """
    Performs sentiment analysis on a list of news articles.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    if news_list:
        for news in news_list:
            title = news.get('content', {}).get('title', '')
            summary = news.get('content', {}).get('summary', '')
            text_to_analyze = title + ". " + summary
            sentiment = analyzer.polarity_scores(text_to_analyze)
            sentiments.append({
                'title': title,
                'sentiment': sentiment
            })
    return sentiments

def preprocess_stock_data(data):
    """
    Normalizes the 'Close' price of the stock data.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    close_price = data['Close'].values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)
    data['Normalized Close'] = scaled_close
    return data

def create_dataset_for_prediction(stock_data, sentiments):
    """
    Creates a dataset for generating historical predictions.
    """
    stock_data['Price_Up'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)

    avg_sentiment = 0
    if sentiments:
        compound_sentiments = [s['sentiment']['compound'] for s in sentiments if 'sentiment' in s and 'compound' in s['sentiment']]
        if compound_sentiments:
            avg_sentiment = np.mean(compound_sentiments)

    stock_data['Avg_Sentiment'] = avg_sentiment

    stock_data = stock_data.dropna()

    X = stock_data[['Normalized Close', 'Avg_Sentiment']]
    y = stock_data['Price_Up']

    return X, y, stock_data.index

def generate_historical_predictions(model, X):
    """
    Generates predictions for a historical dataset.
    """
    return model.predict(X)

def save_predictions_to_csv(data, predictions, index, outfile):
    """
    Saves the data and predictions to a CSV file.
    """
    output_df = pd.DataFrame({
        'Date': index,
        'Close': data.loc[index]['Close'],
        'Predicted_Direction': ['UP' if p == 1 else 'DOWN' for p in predictions]
    })
    output_df.to_csv(outfile, index=False)

def main(args):
    """
    Main function to run the stock prediction application.
    """
    ticker_symbol = args.ticker
    outfile = args.outfile

    print(f"Fetching data for {ticker_symbol}...")
    data = fetch_stock_data(ticker_symbol)

    if data.empty:
        print(f"Could not fetch stock data for {ticker_symbol}.")
        return

    print("Preprocessing data...")
    processed_data = preprocess_stock_data(data.copy())

    news = fetch_news(ticker_symbol)
    sentiments = analyze_sentiment(news)

    X, y, prediction_index = create_dataset_for_prediction(processed_data, sentiments)

    if X.empty:
        print("Could not create a dataset for the model.")
        return

    print("Training baseline model...")
    model = LogisticRegression()
    model.fit(X, y)

    print("Generating historical predictions...")
    historical_predictions = generate_historical_predictions(model, X)

    print(f"Saving predictions to {outfile}...")
    save_predictions_to_csv(data, historical_predictions, prediction_index, outfile)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Predictor - CSV Generator")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., XOM)")
    parser.add_argument("--outfile", type=str, required=True, help="Output CSV file path (e.g., xom_predictions.csv)")
    args = parser.parse_args()
    main(args)
