from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *

import warnings

warnings.filterwarnings('ignore')

window = 20
learning_rate = 0.00005
print('Getting the data...')
portfolio_dict, portfolio = get_data()
X, Y = ML_data(portfolio, portfolio_dict, window)

print('Populating models...')
# models, checkpoints = Create_Models(portfolio_dict, window, learning_rate)
models, checkpoints = PopulateModels(portfolio_dict, window, learning_rate)

train = ''
while not(train=='y' or train=='n'):
    train = input('Train? [y/n]: ')
    
if train=='y':
    print('Training models...')
    Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=15)
 

print('Making predictions...')
predictions_all, next_prediction, joint = Predictions(X, Y, models, window)

n = 0
for tckr in models.keys():
    n += models[tckr].evaluate(X[tckr][2], Y[tckr][2], verbose=0)
print(f'Average succes of predictions (1-MAPE): {100-n/len(models.keys())}')
    
meassurements = Measurements(joint)
meassurements.to_csv('meassurements.csv')
"""
tckr = ''
while tckr!='Q':
    tckr = input("Enter ticker to display or 'q' to end: ")
    tckr = tckr.upper()
    if tckr in models.keys():
        print('Whole data')
        Plot_Predictions(predictions_all, tckr)
        print()
        print(next_prediction[tckr])
"""