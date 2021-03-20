import yfinance as yf
import numpy as np
import scipy as sp
import pandas as pd
from talib import RSI,BBANDS,MACD


def getTickerPriceData(tickers,period='5d',interval='1d'):
    #Getting Ticker Price Data (Open,High,Close,etc)
    ticker_df = yf.download(tickers=tickers,period=period,interval=interval)
    return ticker_df

def makeTickerDfSignals(ticker_data_df,interval='1d',short_window=9,long_window=21):
    #Add computational signals to the ticker dataframe

    # Day Length Trade Intervals:
    #if ticker_data_df.index.name == 'Date':
    signals_df = ticker_data_df.loc[:,['Close']].copy()
    # Set the `date` column as the index
    #signals_df.set_index(signals_df.index.name, drop=True, inplace=True)

    # Calculate Daily Returns
    signals_df['Daily_Return'] = signals_df['Close'].dropna().pct_change()
    signals_df.dropna(inplace=True)

    # Generate the short and long moving averages (short window and long window days, respectively)
    signals_df['SMA%s'%short_window] = signals_df['Close'].rolling(window=short_window).mean()
    signals_df['SMA%s'%long_window] = signals_df['Close'].rolling(window=long_window).mean()
    #print(signals_df.head())
    # Initialize the new `Signal` column
    signals_df['Signal'] = 0.0

    signals_df.dropna(inplace=True)
    # Generate the trading signal (1 or 0) to when the short window is less than the long
    # Note: Use 1 when the SMA50 is less than SMA100 and 0 for when it is not.
    signals_df['Signal'][short_window:] = np.where(
        signals_df["SMA%s" % short_window][short_window:] > signals_df["SMA%s" % long_window][short_window:], 1.0,
        0.0)

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long window close prices, respectively
    signals_df['fast_close_%s'%short_window] = signals_df['Close'].ewm(halflife=short_window).mean()
    signals_df['slow_close_%s'%long_window] = signals_df['Close'].ewm(halflife=long_window).mean()

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    signals_df['fast_vol'] = signals_df['Daily_Return'].ewm(halflife=short_window).std()
    signals_df['slow_vol'] = signals_df['Daily_Return'].ewm(halflife=long_window).std()


    # Calculate the points in time at which a position should be taken, 1 or -1
    signals_df["Entry/Exit"] = signals_df["Signal"].diff()

    # RSI Indicator
    signals_df['RSI'] = RSI(ticker_data_df['Close'], timeperiod=14)

    # MACD Indicator
    macd, macdsignal, macdhist = MACD(ticker_data_df['Close'], fastperiod=short_window, slowperiod=long_window, signalperiod=6)
    signals_df['MACD'] = macd
    signals_df['MACD_Sig'] = macdsignal

    #else:
        #print("Intra-Day trading will be developed!")
        #signals_df = [[]]

    return signals_df