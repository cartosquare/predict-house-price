# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
from hyperopt import fmin, tpe, Trials
## keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split

import time

def mean_absolute_percentage_error(y_true, y_pred):
    # convert pd.series to numpy.ndarray
    y_true = y_true.as_matrix()

    dim = y_true.shape
    mape = np.abs((y_true - y_pred) / y_true).sum() / float(dim[0])
    return mape


def run_model():
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))

    ## output layer
    model.add(Dense(1))
    model.add(Activation('linear'))

    ## loss
    model.compile(loss='mean_squared_error', optimizer="adam")

    ## train
    model.fit(X_train, y_train, nb_epoch=10, batch_size=32, validation_split=0, verbose=0)

    ## prediction
    pred = model.predict(X_test, verbose=0)
    pred.shape = (X_test.shape[0],)

    score = mean_absolute_percentage_error(y_test, pred)
    print score


def score(param):
    hidden_units = int(param["hidden_units"])

    ## regression with keras' deep neural networks
    model = Sequential()

    ## hidden layers
    hidden_layers = int(param['hidden_layers'])
    first_layer = True
    while hidden_layers > 0:
        # Dense layer
        if first_layer:
            model.add(Dense(hidden_units, input_dim=X_train.shape[1], init='glorot_uniform'))
        else:
            model.add(Dense(hidden_units, init='glorot_uniform'))

        # batch normal
        if param["batch_norm"]:
            model.add(BatchNormalization(input_shape=(hidden_units,)))

        # Activation layer
        if param["hidden_activation"] == "prelu":
            model.add(PReLU(input_shape=(hidden_units,)))
        else:
            model.add(Activation(param['hidden_activation']))

        # dropout layer
        if first_layer:
            first_layer = False
            model.add(Dropout(param["input_dropout"]))
        else:
            model.add(Dropout(param["hidden_dropout"]))

        hidden_layers -= 1

    ## output layer
    model.add(Dense(1, init='glorot_uniform'))
    model.add(Activation('linear'))

    ## loss
    model.compile(loss='mean_squared_error', optimizer="adam")

    ## train
    model.fit(X_train, y_train, nb_epoch=int(param['nb_epoch']), batch_size=int(param['batch_size']), validation_split=0, verbose=0)

    ## prediction
    pred = model.predict(X_test, verbose=0)
    pred.shape = (X_test.shape[0],)

    score = mean_absolute_percentage_error(y_test, pred)

    print score

    return score


def optimize(trials):
    space = {
        "batch_norm": hp.choice("batch_norm", [True, False]),
        "hidden_units": hp.choice("hidden_units", [64, 128, 256, 512]),
        "hidden_layers": hp.choice("hidden_layers", [1, 2, 3, 4]),
        "input_dropout": hp.quniform("input_dropout", 0, 0.9, 0.1),
        "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.9, 0.1),
        "hidden_activation": hp.choice("hidden_activation", ["relu", "prelu", "sigmoid"]),
        "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
        "nb_epoch": hp.choice("nb_epoch", [10, 20, 30, 40])
    }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=200)

    return best

# loading data
data_file = './price_train.csv'
data = pd.read_csv(data_file, sep=';')

# fill missing value
data['greening_rate'] = data["greening_rate"].fillna(data["greening_rate"].median())
data['year'] = data["year"].fillna(data["year"].median())

# split train data and label
predictors = ['district', 'loop_location', 'building_type', 'greening_rate', 'year', 'has_subway', 'is_hutong']
train = data[predictors]
labels = data['price']

# vectorize
train = train.T.to_dict().values()
vec = DictVectorizer()
train = vec.fit_transform(train)

## to array
train = train.toarray()

## scale
scaler = StandardScaler()
train = scaler.fit_transform(train)

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.25, random_state=42)
'''
## to array
X_train = X_train.toarray()
X_test = X_test.toarray()

## scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print X_train.shape[1]
print X_test.shape[1]

# run_model()
param = {
    'hidden_units': 128, 'hidden_activation': 'prelu', 'batch_size': 128, 'input_dropout': 0.3, 'hidden_dropout': 0.3, 'hidden_layers': 4, 'nb_epoch': 30, 'batch_norm': False
}

start = time.time()
score(param)
end = time.time()
print('Time elapsed: %f' % (end - start))

'''
# Trials object where the history of search will be stored
trials = Trials()

best_param = optimize(trials)
print best_param

parameters = ['hidden_units', 'hidden_layers', 'input_dropout', 'hidden_dropout', 'hidden_activation', 'batch_size', 'nb_epoch']

cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(40, 7))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.05, 0.5])

learn_fig = './learn.png'
savefig(learn_fig)
plt.show()
'''

