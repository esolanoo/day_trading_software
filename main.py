from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *

import warnings

warnings.filterwarnings('ignore')

print('Getting the data...')
portfolio_dict, portfolio = get_data()
X, Y = ML_data(portfolio, portfolio_dict)

train = ''
while not(train=='y' or train=='n'):
    train = input('Train? [y/n]: ')

models, checkpoints = Create_Models(portfolio_dict, learning_rate=0.0001)
if train=='y':
    print('Training models...')
    Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=100)
else:
    print('Populating models...')
    models = PopulateModels(portfolio_dict)

print('Making predictions...')
predictions_test, predictions_validation = Predictions(X, Y, models)

tckr = ''
while tckr!='Q':
    tckr = input("Enter ticker to display or 'q' to end: ")
    tckr = tckr.upper()
    if tckr in models.keys():
        print('Test data')
        Plot_Predictions(predictions_test, tckr)
        print()
        print('Validation data')
        Plot_Predictions(predictions_validation, tckr)