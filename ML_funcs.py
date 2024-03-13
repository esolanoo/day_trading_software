import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from pathlib import Path

import warnings


warnings.filterwarnings('ignore')


def Create_Models(portfolio_dict):
    models, checkpoints = {}, {}

    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]: 
            models[tckr] = Sequential()
            models[tckr].add(InputLayer((6, 8)))
            models[tckr].add(LSTM(64))
            models[tckr].add(Dense(8, 'relu'))
            models[tckr].add(Dense(4, 'linear'))
            checkpoints[tckr] = ModelCheckpoint('{}/'.format(tckr), save_best_only=True)
    
    return models, checkpoints

def Train_Models(portfolio_dict, X, Y, models, checkpoints, epochs=10, learning_rate=0.0001):
    pbar = tqdm(total=sum([len(x) for x in portfolio_dict.values()]))
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            models[tckr].compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), 
                                 metrics=[RootMeanSquaredError()])
            models[tckr].fit(X[tckr][0], Y[tckr][0], validation_data=(X[tckr][1], Y[tckr][1]), 
                             epochs=epochs, callbacks=[checkpoints[tckr]])
            pbar.update(1)
    pbar.close()


def Predictions(X, Y, models):
    preds = {}
    for tckr in list(X.keys()):
        y = models[tckr].predict(X[tckr][2]) # Validation data
        y_p = y*Y[tckr][4]+Y[tckr][3]
        y_real = Y[tckr][2]*Y[tckr][4]+Y[tckr][3]
        data = {
            'Actual High': y_real[:, 0],
            'Predicted High': y_p[:, 0],
            'Actual Low': y_real[:, 1],
            'Predicted Low': y_p[:, 1],
            'Actual Close': y_real[:, 2],
            'Predicted Close': y_p[:, 2],
            'Actual Adj Close': y_real[:, 3],
            'Predicted Adj Close': y_p[:, 3]
        }
        preds[tckr] = pd.DataFrame(data=data)
    return preds


def PopulateModels(portfolio_dict):
    models = {}
    pbar = tqdm(total=sum([len(x) for x in portfolio_dict.values()]))
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            str_path = 'D:\Documentos\ECID\ProyectoTerminal\DayTradingSoftware\day_trading_software\{}\saved_model.pb'.format(tckr)
            path = Path(str_path)
            models[tckr] = tf.keras.models.load_model(path)
            pbar.update(1)
    pbar.close()
            
    return models

