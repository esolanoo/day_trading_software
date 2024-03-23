from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def random_day():
    random_day =  datetime.now() - timedelta(days=random.randint(0, 55))
    return random_day
    

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
    stock_hist['s'] = stock_hist.index.map(pd.Timestamp.timestamp)
    stock_hist['day sin'] = np.sin(stock_hist['s']*(2*np.pi / 60/60/24))
    stock_hist.ffill(inplace=True)
    stock_hist.dropna(inplace=True)
    stock_hist.reset_index(inplace=True)
    stock_hist.drop(['s', 'Datetime', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
    if 'Capital Gains' in stock_hist.columns:
        stock_hist.drop('Capital Gains', axis=1, inplace=True)
    
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


def preprocess(DF, window=6):
    df = DF.copy()
    targets = df.loc[:, ['High', 'Low', 'Close', 'Adj Close']].to_numpy()
    df = df.to_numpy()
    X = []
    Y = []

    for i in range(len(df)-window):
        r = [x for x in df[i:i+window]]
        X.append(r)
        Y.append(targets[i+window])

    X = np.array(X)
    Y = np.array(Y)
    
    idx = 7*int(X.shape[0]/10)
    idx2 = 9*int(X.shape[0]/10)
    X_train = X[:idx, :, :]
    Y_train = Y[:idx, :]
    X_test =  X[idx:idx2, :, :]
    Y_test =  Y[idx:idx2, :]
    X_val =  X[idx2:, :, :]
    Y_val =  Y[idx2:, :]
    
    x_means = [np.mean(X_train[:, :, i]) for i in range(X_train.shape[2])]
    x_stds = np.array([np.std(X_train[:, :, i]) for i in range(X_train.shape[2])])

    X_train_p = (X_train-x_means)/x_stds
    X_test_p = (X_test-x_means)/x_stds
    X_val_p = (X_val-x_means)/x_stds
        
    y_means = x_means[:4]
    y_stds = x_stds[:4]

    Y_train_p = (Y_train-y_means)/y_stds
    Y_test_p = (Y_test-y_means)/y_stds
    Y_val_p = (Y_val-y_means)/y_stds
    
    return [X_train_p, X_test_p, X_val_p, x_means, x_stds], [Y_train_p, Y_test_p, Y_val_p, y_means, y_stds]


def ML_data(portfolio, portfolio_dict):
    X, Y = {}, {}

    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            X[tckr], Y[tckr] = preprocess(portfolio[eq][tckr], window=20)
    
    return X, Y