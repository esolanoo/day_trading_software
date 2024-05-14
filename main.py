from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *
from alpaca_funcs import *

from datetime import datetime
from datetime import time as t
import time
import pytz
import os
import warnings



warnings.filterwarnings('ignore')

def nyc_stock_market_open():
    nyc_timezone = pytz.timezone("America/New_York")
    current_time = datetime.now(nyc_timezone).time()
    market_open = t(9, 30)
    market_close = t(15, 45)

    return market_open <= current_time < market_close

def Trade(train=False, evaluate=False, trade=True):
    window = 20
    learning_rate = 0.00005
    print('Getting the data...')
    portfolio_dict, portfolio = get_data()

    X, Y = ML_data(portfolio, portfolio_dict, window)

    print('Populating models...')
    # models, checkpoints = Create_Models(portfolio_dict, window, learning_rate)
    models, checkpoints = PopulateModels(portfolio_dict, window, learning_rate)

    if train:
        print('Training models...')
        Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=5)
        
        
    print('Making predictions...')
    predictions_all, next_prediction, joint = Predictions(X, Y, models, window)

    if evaluate:
        n = 0
        for tckr in models.keys():
            n += models[tckr].evaluate(X[tckr][2], Y[tckr][2], verbose=0)
        print(f'Average succes of predictions  (1-MAPE): {100-n/len(models.keys())}')

        tckr = ''
        while tckr!='Q':
            tckr = input("Enter ticker to display or 'q' to end: ")
            tckr = tckr.upper()
            if tckr in models.keys():
                print('Whole data')
                Plot_Predictions(predictions_all, tckr)
                print()
                print(next_prediction[tckr])
        
    if trade:
        meassurements = Measurements(joint)
        df = showPortfolio()
        for tckr in meassurements.index.tolist():
            tckr_alpaca = tckr.replace('-', '/')
            if tckr not in portfolio_dict['futures']:
                if meassurements.loc[tckr, 'signal']=='Buy' and AccountBalance()>bid_price(tckr):
                    placeOrder(tckr_alpaca, 1, True)
                elif meassurements.loc[tckr, 'signal']=='Sell' and tckr in df['symbol']:
                    placeOrder(tckr_alpaca, 1, False)
    

def main():
    initial_investment = 6000
    trained_today = True
    while True: 
        os.system('cls')
        AccountPerformance(initial_investment)
        if not(nyc_stock_market_open()):
            if trained_today:
                SellAll()
                trained_today = False
            else:
                Trade(train=True, evaluate=False, trade=False)
                trained_today = True
                while not(nyc_stock_market_open):
                    pass 
        else: 
            start_time = time.time()
            Trade()
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 240:  # 4 minutes
                wait_time = 240 - elapsed_time
                print(f'Waiting for next market candles ({wait_time} seconds)...')
                time.sleep(wait_time)
                

if __name__ == "__main__":
    main()
    