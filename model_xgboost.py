# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from hyperopt import hp
from hyperopt import fmin, tpe, Trials
from sklearn import cross_validation
from sklearn.svm import SVR
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold


def mean_absolute_percentage_error(preds, dtrain):
    labels = dtrain.get_label()
    dim = labels.shape

    error_val = np.abs((labels - preds) / labels).sum() / float(dim[0])
    return 'error', error_val


def run_model(param):
    '''
    param = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'colsample_bytree': 0.9,
        'min_child_weight': 0.0,
        'subsample': 0.5,
        'eta': 0.01,
        'max_depth': 5,
        'gamma': 2.0
    }

    param = {
        'booster': 'gblinear',
        'objective': 'reg:linear',
        'lambda_bias': 100.0,
        'alpha': 0.054,
        'eta': 0.0033,
        'lambda': 403.0
    }
    '''
    param['min_child_weight'] = int(param['min_child_weight'])
    param['max_depth'] = int(param['max_depth'])
    num_rounds = int(param['num_round'])

    # Create Train and Test DMatrix
    xgtest = xgb.DMatrix(X_test, label=y_test)
    xgtrain = xgb.DMatrix(X_train, label=y_train)

    # train
    watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
    xgb.train(param, xgtrain, num_rounds, watchlist, feval=mean_absolute_percentage_error)


def score(param):
    param['min_child_weight'] = int(param['min_child_weight'])
    param['max_depth'] = int(param['max_depth'])
    num_rounds = int(param['num_round'])

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    res = xgb.cv(param, xgtrain, num_rounds, nfold=5, metrics={'error'}, seed=0, feval=mean_absolute_percentage_error)

    mape = res.tail(1).iloc[0][0]
    print mape
    return mape


def optimize(trials):
    # {'epsilon': 0.0020306829526341645, 'C': 99.47584472767942, 'gamma': 0.09951456444237791, 'kernel': 0}
    space = {
        'booster': 'gblinear',
        'objective': 'reg:linear',
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'lambda': hp.quniform('lambda', 0, 5, 0.05),
        'alpha': hp.quniform('alpha', 0, 0.5, 0.005),
        'lambda_bias': hp.quniform('lambda_bias', 0, 3, 0.1),
        'num_round': hp.quniform('num_round', 10, 500, 1),
        'silent': 1
    }

    space2 = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eta': hp.quniform('eta', 0.0001, 0.1, 0.0001),
        'gamma': hp.quniform('gamma', 0, 2, 0.1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
        'num_round': hp.quniform('num_round', 10, 2000, 10),
        'silent': 1
    }

    best = fmin(score, space2, algo=tpe.suggest, trials=trials, max_evals=200)

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

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.25, random_state=42)

print X_train.shape
print X_test.shape

# run_model()

# Trials object where the history of search will be stored
trials = Trials()

best_param = optimize(trials)

run_model(best_param)
print best_param

parameters = ['eta', 'gamma', 'min_child_weight', 'max_depth', 'subsample', 'colsample_bytree', 'num_round']
# parameters = ['eta', 'lambda', 'alpha', 'lambda_bias', 'num_round']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(40, 7))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.1, 0.17])

learn_fig = './learn.png'
savefig(learn_fig)
plt.show()
