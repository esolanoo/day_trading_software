import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import warnings


warnings.filterwarnings('ignore')


def Create_Models(portfolio_dict):
    models, checkpoints = {}, {}

    for eq in portfolio_dict:
        for stck in portfolio_dict[eq]: 
            models[stck] = Sequential()
            models[stck].add(InputLayer((6, 8)))
            models[stck].add(LSTM(64))
            models[stck].add(Dense(8, 'relu'))
            models[stck].add(Dense(4, 'linear'))
            checkpoints[stck] = ModelCheckpoint('{}/'.format(stck), save_best_only=True)
    
    return models, checkpoints

def Train_Models(portfolio_dict, X, Y, models, checkpoints, epochs=10, learning_rate=0.001):
    for eq in portfolio_dict:
        for stck in portfolio_dict[eq]:
                print(stck)
                models[stck].compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), 
                                     metrics=[RootMeanSquaredError()])
                models[stck].fit(X[stck][0], Y[stck][0], validation_data=(X[stck][1], Y[stck][1]), 
                                 epochs=epochs, callbacks=[checkpoints[stck]])