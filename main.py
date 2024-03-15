from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *

import warnings

warnings.filterwarnings('ignore')

portfolio_dict, portfolio = get_data()
X, Y = ML_data(portfolio, portfolio_dict)
train = True

models, checkpoints = Create_Models(portfolio_dict, learning_rate=0.005)
if train:
    Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=3)
else:
    models = PopulateModels(portfolio_dict)