from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *
from alpaca_funcs import *

from datetime import datetime, time
import pytz
import warnings



warnings.filterwarnings('ignore')

def nyc_stock_market_open():
    nyc_timezone = pytz.timezone("America/New_York")
    current_time = datetime.now(nyc_timezone).time()
    market_open = time(9, 30)
    market_close = time(16, 0)

    return market_open <= current_time < market_close


window = 20
learning_rate = 0.00005
print('Getting the data...')
portfolio_dict, portfolio = get_data()

X, Y = ML_data(portfolio, portfolio_dict, window)

print('Populating models...')
# models, checkpoints = Create_Models(portfolio_dict, window, learning_rate)
models, checkpoints = PopulateModels(portfolio_dict, window, learning_rate)

if not(nyc_stock_market_open()):
    print('Training models...')
    Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=5)
    
    
print('Making predictions...')
predictions_all, next_prediction, joint = Predictions(X, Y, models, window)

# n = 0
# for tckr in models.keys():
#     n += models[tckr].evaluate(X[tckr][2], Y[tckr][2], verbose=0)
# print(f'Average succes of predictions  (1-MAPE): {100-n/len(models.keys())}')

# tckr = ''
# while tckr!='Q':
#     tckr = input("Enter ticker to display or 'q' to end: ")
#     tckr = tckr.upper()
#     if tckr in models.keys():
#         print('Whole data')
#         Plot_Predictions(predictions_all, tckr)
#         print()
#         print(next_prediction[tckr])
    
meassurements = Measurements(joint)

df = showPortfolio()

for tckr in meassurements.index.tolist():
    if tckr not in portfolio_dict['crypto'] + portfolio_dict['futures']:
        if meassurements.loc[tckr, 'signal']=='Buy' and AccountBalance()>bid_price(tckr):
            placeOrder(tckr, 1, True)
        if meassurements.loc[tckr, 'signal']=='Sell' and df.loc[tckr, 'volume'] > 0:
            placeOrder(tckr, 1, False)