import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import InputLayer, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from tqdm import tqdm
import os

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


warnings.filterwarnings('ignore')


def Single_Model():
    model = Sequential([
        InputLayer((20, 8)),     # Modify if window changed
        LSTM(64, activation='tanh', return_sequences=True),
        LSTM(64),
        Dense(8, 'relu'),
        Dense(4, 'linear')
    ])
    return model


def Create_Models(portfolio_dict, learning_rate=0.001):
    models, checkpoints = {}, {}
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:               
            models[tckr] = Single_Model()
            checkpoints[tckr] = ModelCheckpoint("{}/cpt.weights.h5".format(tckr), save_weights_only=True, verbose=0)
            models[tckr].compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    return models, checkpoints


def PopulateModels(portfolio_dict):
    models = {}
    pbar = tqdm(total=sum([len(x) for x in portfolio_dict.values()]))
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            models[tckr] = Single_Model()
            models[tckr].load_weights("{}/cpt.weights.h5".format(tckr))
            pbar.update(1)
    pbar.close()
            
    return models


def Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs=10):
    pbar = tqdm(total=sum([len(x) for x in portfolio_dict.values()]))
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            print(tckr)
            models[tckr].fit(X[tckr][0], Y[tckr][0], validation_data=(X[tckr][1], Y[tckr][1]), epochs=epochs, callbacks=[checkpoints[tckr]])
            pbar.update(1)
            print()
    pbar.close()


def Predictions(X, Y, models):
    preds_test, preds_validation = {}, {}
    for tckr in list(X.keys()):
        y = models[tckr].predict(X[tckr][1]) # Test data
        y_p = y*Y[tckr][4]+Y[tckr][3]
        y_real = Y[tckr][1]*Y[tckr][4]+Y[tckr][3]
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
        preds_test[tckr] = pd.DataFrame(data=data)
        
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
        preds_validation[tckr] = pd.DataFrame(data=data)
    return preds_test, preds_validation




