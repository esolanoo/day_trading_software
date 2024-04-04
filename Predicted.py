from data_funcs import *
from performance_funcs import *
from ML_funcs import *
from plot_funcs import *

import warnings

warnings.filterwarnings('ignore')

window = 20
learning_rate = 0.0005


portfolio_dict, portfolio = get_data()
X, Y = ML_data(portfolio, portfolio_dict, window)
models, checkpoints = PopulateModels(portfolio_dict, window, learning_rate)
predictions_all, next_prediction, joint = Predictions(X, Y, models, window)


measurements = Measurements(joint)
pd.DataFrame(measurements).to_csv('measurements.csv')
