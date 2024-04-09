import pandas as pd
import numpy as np


def calc_atr(df, period=14):
    """
    Calculate the Average True Range (ATR)
    """
    high_low = df['high'] - df['low']
    high_close = df['high'] - df['close'].shift(1)
    low_close = df['low'] - df['close'].shift(1)
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr


def calc_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI)
    """
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    average_gain = up.rolling(window=period).mean()
    average_loss = abs(down.rolling(window=period).mean())

    rs = average_gain / average_loss
    rsi = 100*(100 - (100 / (1 + rs)))

    return rsi


def calc_adx(df, period=14):
    """
    Calculate the Average Directional Index (ADX)
        [0-25): Weak trend
        [25-50): Strong trend
        [50-75): Very strong trend
        [75-100]: Extremely strong trend
    """
    H = df['high'] - df['high'].shift(1)
    L = df['low'].shift(1) - df['low']
    
    pdm = np.where((H > L) & (H > 0), H, 0)
    ndm = np.where((H < L) & (L > 0), L, 0)
    
    pdi = ((pdm/df['atr']).ewm(alpha=1/period, adjust=False).mean() / df['atr']) * 100
    ndi = ((ndm/df['atr']).ewm(alpha=1/period, adjust=False).mean() / df['atr']) * 100

    adx = ((abs(pdi-ndi) / (pdi+ndi))*100).ewm(alpha=1/period, adjust=False).mean()

    return adx


def  KPIs(df):
    """
    Calculate the Compound Annual Growth Rate (CAGR) and the annualized volatility
    """
    returns = df['adj close'].pct_change()
    periods = len(returns)
    intervals_per_year = 252 * 78   # 78 inputs per day (NY Stock exchange is open 6.5 hr per day)
    
    cagr = (df['adj close'].iloc[-1] / df['adj close'].iloc[0]) ** (intervals_per_year / periods) - 1
    volatility = returns.std() * np.sqrt(intervals_per_year / periods)
    
    return cagr, volatility


def tckr_Sortino(df, rfr=0.00):
    """
    Calculate the Sortino Ratio
    """
    returns = df['returns']
    expected_return = returns.mean()
    negative_returns = returns[returns < rfr]
    downside_deviation = np.sqrt((negative_returns**2).mean())
    sortino_ratio = (expected_return - rfr) / downside_deviation
    return sortino_ratio


def add_trading_signals(df):
    # Define your conditions for Buy, Hold and Sell
    conditions = [
        (df['adx'] > .25),  # Buy condition
        (df['rsi'] > .01),  # Sell condition
        (df['rsi'].between(.03, .05)) & (df['adx'] <= .25)  # Hold condition
    ]

    choices = ['Buy', 'Sell', 'Hold']

    df['signal'] = np.select(conditions, choices, default='Hold')

    return df


def Measurements(joint):
    measurements = {}
    for tckr in list(joint.keys()):
        df = pd.DataFrame(joint[tckr])
        df['atr'] = calc_atr(df)
        df['rsi'] = calc_rsi(df)
        df['adx'] = calc_adx(df)
        df.dropna(axis=1, inplace=True)
        df['sortino'] = tckr_Sortino(df)
        measurements[tckr] = df.iloc[-1, -4:]
    measurements = pd.DataFrame(measurements)
    measurements = measurements.transpose()
    measurements.columns = ['atr', 'rsi', 'adx', 'sortino']
    measurements = add_trading_signals(measurements)
    return measurements

