from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def calc_atr(df, period=14):
    """
    Calculate the Average True Range (ATR)
    """
    high_low = df['High'] - df['Low']
    high_close = df['High'] - df['Close'].shift(1)
    low_close = df['Low'] - df['Close'].shift(1)
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    atr = true_range.ewm(com=period, min_periods=period).mean()
    return atr


def calc_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI)
        Values above .7 indicate the asset is in overbought territory
        Values below .3 indicate the asset is in oversold territory
    """
    delta = df['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    average_gain = up.rolling(window=period).mean()
    average_loss = abs(down.rolling(window=period).mean())

    rs = average_gain / average_loss
    rsi = (100 - (100 / (1 + rs)))/100
    
    return rsi


def calc_adx(df, period=14):
    """
    Calculate the Average Directional Index (ADX)
        [0-25): Weak trend
        [25-50): Strong trend
        [50-75): Very strong trend
        [75-100]: Extremely strong trend
    """
    H = df['High'] - df['High'].shift(1)
    L = df['Low'].shift(1) - df['Low']
    
    pdm = np.where((H > L) & (H > 0), H, 0)
    ndm = np.where((H < L) & (L > 0), L, 0)
    
    pdi = (pdm/df['atr']).ewm(com=period, min_periods=period).mean()
    ndi = (ndm/df['atr']).ewm(com=period, min_periods=period).mean()

    adx = (abs(pdi-ndi) / (pdi+ndi)).ewm(com=period, min_periods=period).mean()

    return adx


def historical(ticker):
    now = datetime.now()
    start = now - timedelta(days=58)

    stock = ticker

    df = yf.download(stock, end=now.strftime('%Y-%m-%d'), start=start.strftime('%Y-%m-%d'), interval='5m', progress=False)['Adj Close']
    stock_ticker = yf.Ticker(stock)
    stock_hist = stock_ticker.history(end=now.strftime('%Y-%m-%d'), start=start.strftime('%Y-%m-%d'), interval='5m');
    stock_hist.reset_index(inplace=True)
    stock_hist.set_index('Datetime', inplace=True)
    stock_hist = pd.concat([stock_hist, df], axis=1)
    stock_hist['returns'] = stock_hist['Adj Close'].pct_change()
    stock_hist['atr'] = calc_atr(stock_hist)
    stock_hist['rsi'] = calc_rsi(stock_hist)
    stock_hist['adx'] = calc_adx(stock_hist)
    stock_hist['s'] = stock_hist.index.map(pd.Timestamp.timestamp)
    stock_hist['day sin'] = np.sin(stock_hist['s']*(2*np.pi / 60/60/24))
    stock_hist.ffill(inplace=True)
    stock_hist.dropna(inplace=True)
    stock_hist.reset_index(inplace=True)
    stock_hist.drop(['s', 'Datetime'], axis=1, inplace=True)
    
    return stock_hist.to_dict()


def get_data():
    with open('tickers.txt', 'r') as file:
        lines = file.readlines()
    
    portfolio = {}

    for line in lines:
        key, value = line.strip().split('=', 1)
        list_name = key.strip()
        items = value.strip().strip('[]').replace("'", "").split(', ')
        portfolio[list_name.replace(' ', '')] = items

    portfolio_hist = {}

    pbar = tqdm(total=sum([len(x) for x in portfolio.values()]))
    columns = iter(portfolio.keys())
    for eq in portfolio.values():
        aux = pd.DataFrame()
        for stck in eq:
            df = historical(stck)
            aux[stck] = df
            pbar.update(1)
        multi_indexed_df = pd.concat({(i, j): pd.Series(v) for i, d in aux.to_dict().items() for j, v in d.items()}, axis=0)
        hist = pd.DataFrame(multi_indexed_df.unstack().transpose())
        #hist = hist.swaplevel(0, 1, axis=1)
        hist = hist.sort_index(axis=1)
        portfolio_hist[next(columns)] = hist.dropna() 
    pbar.close()
    
    return portfolio, portfolio_hist 
