import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import InputLayer, LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import MeanAbsolutePercentageError
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import os

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


warnings.filterwarnings('ignore')


def Single_Model(window):
    model = Sequential([
        InputLayer((window, 8)),     # Modify if window changed
        LSTM(64, activation='tanh', return_sequences=True),
        LSTM(64),
        Dense(8, 'relu'),
        Dense(4, 'linear')
    ])
    return model


def Create_Models(portfolio_dict, window, learning_rate):
    models, checkpoints = {}, {}
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:               
            models[tckr] = Single_Model(window=window)
            checkpoints[tckr] = ModelCheckpoint("{}/cpt.weights.h5".format(tckr), save_weights_only=True, verbose=0)
            models[tckr].compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(learning_rate=learning_rate))
    return models, checkpoints


def PopulateModels(portfolio_dict, window, learning_rate):
    models, checkpoints = {}, {}
    pbar = tqdm(total=sum([len(x) for x in portfolio_dict.values()]))
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            models[tckr] = Single_Model(window=window)
            models[tckr].load_weights("{}/cpt.weights.h5".format(tckr))
            checkpoints[tckr] = ModelCheckpoint("{}/cpt.weights.h5".format(tckr), save_weights_only=True, verbose=0)
            models[tckr].compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(learning_rate=learning_rate))
            pbar.update(1)
    pbar.close()
    return models, checkpoints


def Fit_Models(portfolio_dict, X, Y, models, checkpoints, epochs):
    pbar = tqdm(total=sum([len(x) for x in portfolio_dict.values()]))
    for eq in portfolio_dict:
        for tckr in portfolio_dict[eq]:
            print(tckr)
            if models[tckr].evaluate(X[tckr][1],Y[tckr][1], verbose=0)<=1:
                print( f'Model MAPE sufficiently low: {models[tckr].evaluate(X[tckr][1],Y[tckr][1], verbose=0)}')
            else: 
                early_stopping = EarlyStopping(monitor='val_loss', patience=int(epochs/3), restore_best_weights=True)
                try:
                    models[tckr].fit(X[tckr][0], Y[tckr][0], validation_data=(X[tckr][1], Y[tckr][1]), epochs=epochs, 
                                    callbacks=[checkpoints[tckr], early_stopping], verbose=0)
                except OSError as e:
                    if e.errno == 22:
                        print("\nUnable to synchronously create file. Passing...")
                    pass
            pbar.update(1)
            print()
    pbar.close()


def Predictions(X, Y, models, window):
    preds_all = {}
    pred_next = {}
    joint = {}
    pbar = tqdm(total=len([x for x in models.keys()])*2)
    for tckr in list(X.keys()):
        y = models[tckr].predict(X[tckr][2], verbose=0)
        pbar.update(1)
        y_real = Y[tckr][2]
        data = {
            'Actual High': y_real[:, 0],
            'Predicted High': y[:, 0],
            'Actual Low': y_real[:, 1],
            'Predicted Low': y[:, 1],
            'Actual Close': y_real[:, 2],
            'Predicted Close': y[:, 2],
            'Actual Adj Close': y_real[:, 3],
            'Predicted Adj Close': y[:, 3]
        }
        
        preds_all[tckr] = pd.DataFrame(data=data)
        pred_next[tckr] = models[tckr].predict(X[tckr][3].reshape(1, window, 8), verbose=0)
        
        joint[tckr] = preds_all[tckr][['Predicted High', 'Predicted Low', 'Predicted Close', 'Predicted Adj Close']]
        joint[tckr].loc[len(joint[tckr])] = pred_next[tckr][0]
        joint[tckr].columns = ['high', 'low', 'close', 'adj close']
        joint[tckr]['returns'] = joint[tckr]['adj close'].pct_change()
        joint[tckr] = joint[tckr].dropna()
        pbar.update(1)
    pbar.close()
    return preds_all, pred_next, joint




