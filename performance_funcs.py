import pandas as pd
import numpy as np


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
    