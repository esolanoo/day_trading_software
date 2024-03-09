import pandas as pd
import numpy as np


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


def  KPIs(df):
    """
    Calculate the Compound Annual Growth Rate (CAGR) and the annualized volatility
    """
    returns = df['Adj Close'].pct_change()
    periods = len(returns)
    intervals_per_year = 252 * 78   # 78 inputs per day (NY Stock exchange is open 6.5 hr per day)
    
    cagr = (df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) ** (intervals_per_year / periods) - 1
    volatility = returns.std() * np.sqrt(intervals_per_year / periods)
    
    return cagr, volatility


def tckr_Sortino(df, rfr=0.02):
    """
    Calculate the Sortino Ratio
    """
    cagr, neg_vol = KPIs(df)
    return (cagr- rfr) / neg_vol


def Measurements(portfolio):
    measurements = pd.DataFrame()
    for eq in portfolio.keys():
        for stck in portfolio[eq]:
            measurements[stck] = [portfolio[eq][stck]['adx'].iloc[-1], portfolio[eq][stck]['rsi'].iloc[-1], tckr_Sortino(portfolio[eq][stck])]      
    measurements = measurements.transpose()
    measurements.columns = ['adx', 'rsi', 'sortino']
    return measurements
    