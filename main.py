from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *
from alpaca_funcs import *

from datetime import datetime
from datetime import time as t
import random
import time
import pytz
import os
import warnings
from IPython.display import display



warnings.filterwarnings('ignore')

def nyc_stock_market_open():
    nyc_timezone = pytz.timezone("America/New_York")
    current_time = datetime.now(nyc_timezone).time()
    market_open = t(9, 30)
    market_close = t(15, 45)

    return market_open <= current_time < market_close

def Trade(train=False, evaluate=False, trade=True, epochs=5):
    window = 12
    learning_rate = 0.00005
    print('Getting the data...')
    portfolio_dict, portfolio = get_data()
    expensive, expensive_tckr = most_expensive(portfolio, portfolio_dict)
    X, Y = ML_data(portfolio, portfolio_dict, window)

    print('Populating models...')
    models, checkpoints = PopulateModels(portfolio_dict, window, learning_rate)

    if train:
        print('Training models...')
        Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=epochs)
        
        
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
        aux = meassurements.reset_index()
        aux['index'] = aux['index'].apply(lambda x: x.replace('-', ''))
        aux = df.set_index('symbol').join(aux.set_index('index'))
        for tckr in meassurements.index.tolist():
            if tckr not in portfolio_dict['futures']:
                tckr_order = tckr.replace('-', '')
                try:
                    if tckr != expensive_tckr:
                        qty = random.randint(1, expensive//bid_price(tckr))+1
                        price = qty*bid_price(tckr)*1.03
                    else:
                        qty = 1
                        price = expensive
                    p=''
                    if meassurements.loc[tckr, 'signal']=='Buy' and AccountBalance()>price:
                        print(f'Buy {qty} value of {tckr_order} for {bid_price(tckr)} USD')
                        p = 'buying'
                        placeOrder(tckr_order, qty, True)
                    elif tckr_order in aux.index and aux.loc[tckr_order, 'signal']=='Sell':
                        p = 'selling'
                        crypto = tckr in portfolio_dict['crypto']
                        qty = float(aux.loc[tckr_order, 'volume'])
                        val = float(aux.loc[tckr_order, 'value'])
                        if qty>0:
                            print(f"Sell {qty} values of {tckr} for {qty*val} USD")
                            placeOrder(tckr_order, qty, False, crypto)
                    else:
                        pass
                except:
                    print(f'error {p} {tckr_order}') 
    

def main():
    initial_investment = 10000
    while True: 
        os.system('cls')
        AccountPerformance(initial_investment)
        if nyc_stock_market_open():
            start_time = time.time()
            Trade()
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 240:  # 4 minutes
                wait_time = 240 - elapsed_time
                print(f'Waiting for next market candles ({int(wait_time)} seconds)...')
                time.sleep(wait_time)
             
        else: 
            print('Market clossing')
            print('Selling everything\n')
            SellAll()
            print('Begining training after close')
            Trade(train=True, evaluate=False, trade=False)
            print('Trained!')
            print('Waiting for next market open hour\n')
            while not(nyc_stock_market_open):
                pass
                

if __name__ == "__main__":
    main()
    