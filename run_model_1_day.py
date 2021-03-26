#!/usr/bin/env python
# coding: utf-8

# In[70]:


from pathlib import Path
from tensorflow.keras.models import model_from_json


# In[71]:


import numpy as np
import pandas as pd
import hvplot.pandas
from pathlib import Path
import hvplot

import datetime
from datetime import date, timedelta
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
get_ipython().run_line_magic('matplotlib', 'inline')

load_dotenv()


# In[72]:


# load json and create model
file_path = Path("model_1_day.json")
with open(file_path, "r") as json_file:
    model_json = json_file.read()
loaded_model_1_day = model_from_json(model_json)

# load weights into new model
file_path = "model_1_day.h5"
loaded_model_1_day.load_weights(file_path)


# In[73]:


def fetch_data():
    """Fetches the latest prices."""
   # Set Alpaca API key and secret

    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    alpaca = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    api_version="v2")
    today = date.today()
    start = date.today() - timedelta(days=600)

    # Set the tickers & timeframe
    tickers = ["SPY"]

    timeframe = "1D"

    #Get Closing prices for past 60 days

    df = alpaca.get_barset(
        tickers,
        timeframe,
        start = start,
        end = today
    ).df
    #figure out another way to clean and prep the data here - hard coded for now 
    df_clean = df['SPY']

    # Display sample data
    return df_clean


# In[74]:


data = fetch_data()
data = data.reset_index()
data = data.set_index('time', drop=True)
data['close of tomorrow'] = data['close'].shift(-1)
data


# In[75]:


#Calculate additional data to use as features - add to dataframe



def generate_ema_macd__RSI_signals(df):
    
    def EWMA(data, ndays):
        EMA = pd.Series(data['close'].ewm(span = ndays, min_periods = ndays - 1).mean(),
                     name = 'EWMA_' + str(ndays))
        data = data.join(EMA)
        return data

    def computeRSI (data, time_window):
        diff = data.diff(1).dropna()        # diff in one field(one day)

        #this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[ diff>0 ]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[ diff < 0 ]

        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

        rs = abs(up_chg_avg/down_chg_avg)
        rsi = 100 - 100/(1+rs)
        return rsi


    #Calculate RSI
    df['RSI'] = computeRSI (df['close'], 14)
    # Construct a buy and sell trading signals buy -1 sell - 0 -- Add more sophistication to this -- I wanted to have a signal that says buy when RSI is between 50 -70 or oversold and sell when overbought and RSI is between 30-50
    df['RSI Overbought'] = np.where(df['RSI'] > 70, 1.0, 0.0)
    df['RSI Oversold'] = np.where(df['RSI'] < 30, 1.0, 0.0)
    df['RSI Long'] = np.where(((df['RSI'] > 50) & (df['RSI'] < 70)) | (df['RSI'] < 30), 1.0, 0.0)
    df['RSI Short'] = np.where(((df['RSI'] > 30) & (df['RSI'] < 50)) | (df['RSI'] > 70), 1.0, 0.0)
    
    # Set short and long windows
    short_window = 5
    long_window = 13

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    df['EMA 5'] = df['close'].ewm(halflife=short_window).mean()
    df['EMA 13'] = df['close'].ewm(halflife=long_window).mean()

    # Construct a crossover trading signal
    df['crossover_long_5_13'] = np.where(df['EMA 5'] > df['EMA 13'], 1.0, 0.0)
    df['crossover_short_5_13'] = np.where(df['EMA 5'] < df['EMA 13'], -1.0, 0.0)
    df['crossover_signal_5_13'] = df['crossover_long_5_13'] + df['crossover_short_5_13']

     # Set short and long windows for 9 & 21 EMA
    short_window_9 = 9
    long_window_21 = 21

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    df['EMA 9'] = df['close'].ewm(halflife=short_window_9).mean()
    df['EMA 21'] = df['close'].ewm(halflife=long_window_21).mean()

    # Construct a crossover trading signal
    df['crossover_long_9_21'] = np.where(df['EMA 9'] > df['EMA 21'], 1.0, 0.0)
    df['crossover_short_9_21'] = np.where(df['EMA 9'] < df['EMA 21'], -1.0, 0.0)
    df['crossover_signal_9_21'] = df['crossover_long_9_21'] + df['crossover_short_9_21']
    
    # Set short and long windows for 55 & 200 EMA
    short_window_55 = 55
    long_window_200 = 200

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    df['EMA 55'] = df['close'].ewm(halflife=short_window_55).mean()
    df['EMA 200'] = df['close'].ewm(halflife=long_window_200).mean()

    # Construct a crossover trading signal
    df['crossover_long_55_200'] = np.where(df['EMA 55'] > df['EMA 200'], 1.0, 0.0)
    df['crossover_short_55_200'] = np.where(df['EMA 55'] < df['EMA 200'], -1.0, 0.0)
    df['crossover_signal_55_200'] = df['crossover_long_55_200'] + df['crossover_short_55_200']

    # Set short and long windows for 200 & 800 EMA
    short_window_200 = 200
    long_window_800 = 800

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    df['EMA 200'] = df['close'].ewm(halflife=short_window_200).mean()
    df['EMA 800'] = df['close'].ewm(halflife=long_window_800).mean()

    # Construct a crossover trading signal
    df['crossover_long_200_800'] = np.where(df['EMA 200'] > df['EMA 800'], 1.0, 0.0)
    df['crossover_short_200_800'] = np.where(df['EMA 200'] < df['EMA 800'], -1.0, 0.0)
    df['crossover_signal_200_800'] = df['crossover_long_200_800'] + df['crossover_short_200_800']
    
    # Calculate MACD
    
    EMA_8 = df['close'].ewm(halflife=8).mean()
    EMA_22 = df['close'].ewm(halflife=22).mean()
    macd = EMA_8 - EMA_22
    df['MACD H'] = macd
    df['MACD Signal'] = np.where(df['MACD H'] > 0, 1.0, 0.0)
    
    return df


# In[76]:


data = generate_ema_macd__RSI_signals(data)
data = data.dropna(subset= ['EMA 9', 'EMA 21', 'EMA 5', 'EMA 13', 'RSI','MACD H','EMA 55'])


# In[77]:


#define feauture and target values
X = data[['open', 'high', 'low', 'close', 'volume','EMA 9', 'EMA 21', 'EMA 5', 'EMA 13', 'RSI','MACD H','EMA 55']]#'EMA 800', 'EMA 200']]
y = data[['close of tomorrow']]


X = np.array(X)
y = np.array(y)


# In[78]:


# Use 70% of the data for training and the remaineder for testing
split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:-1]
y_train = y[: split]
y_test = y[split:-1]


# In[79]:


# Use the MinMaxScaler to scale data between 0 and 1.
scaler = MinMaxScaler()
X_scaler = scaler.fit(X)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
y_scaler = scaler.fit(y)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)


# In[80]:


# Reshape the features data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[81]:


X


# In[82]:


# Make predictions using the loaded model data X_test
predicted = loaded_model_1_day.predict(X_test)
trained_predicted = loaded_model_1_day.predict(X_test)

# Recover the original prices instead of the scaled version
predicted_prices = scaler.inverse_transform(predicted)
train_predicted_prices = scaler.inverse_transform(trained_predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
train_real_prices = scaler.inverse_transform(y_train.reshape(-1,1))


# In[83]:


# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Actual": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = data.index[-len(real_prices): ]) 

# Show the DataFrame's head
stocks.tail()



# In[84]:


final_prediction = round(stocks['Predicted'].values[-1], 2)
chart = stocks.plot(title="Actual Vs. Predicted 1 Day SPY Prices")


# In[85]:


# Plot the real vs predicted prices as a line chart
output = print(f'This model predicts the price of SPY will be ${final_prediction} at the close of tomorrow', stocks.plot(title="Actual Vs. Predicted 1 Day SPY Prices"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




