# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from hyperopt import hp
from hyperopt import fmin, tpe, Trials
from sklearn import cross_validation
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split


def mean_absolute_percentage_error(y_true, y_pred):
    # convert pd.series to numpy.ndarray
    y_true = y_true.as_matrix()

    dim = y_true.shape
    mape = np.abs((y_true - y_pred) / y_true).sum() / float(dim[0])
    return 'error', mape

loss_func = make_scorer(mean_absolute_percentage_error, greater_is_better=True)
score_func = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


def run_model():
    # kernel='rbf', C=70000, gamma=0.01, epsilon=0.0005
    # {'epsilon': 0.0019, 'C': 100000.0, 'gamma': 0.01, 'kernel': 0}
    # {'epsilon': 0.0086, 'C': 150000.0, 'gamma': 0.008}
    # {'epsilon': 0.0017, 'C': 160000.0, 'gamma': 0.001}
    svr = SVR(kernel='rbf', C=160000, gamma=0.00001, epsilon=0.001)

    # 使用交叉验证方式训练模型并取得错误率
    scores = cross_validation.cross_val_score(svr, X_train, y_train, cv=4, n_jobs=-1, scoring=score_func)

    # 输出交叉验证平均错误率
    mean_score = np.abs(np.mean(scores))
    print('cross validate score is %f' % (mean_score))

    # 训练模型
    svr.fit(X_train, y_train)

    # 预测
    y_pred = svr.predict(X_test)
    error = mean_absolute_percentage_error(y_test, y_pred)
    print('predict error %f' % (error))


def score(param):
    # cross-validate to get weight
    print param
    svr = SVR(kernel=param['kernel'], C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'])

    # scoring='mean_absolute_error'
    scores = cross_validation.cross_val_score(svr, X_train, y_train, cv=4, n_jobs=-1, scoring=score_func)

    mean_score = np.abs(np.mean(scores))
    print('cross validate score is %f' % (mean_score))

    return mean_score


def optimize(trials):
    # {'epsilon': 0.0020306829526341645, 'C': 99.47584472767942, 'gamma': 0.09951456444237791, 'kernel': 0}
    space = {
        # 'C': hp.loguniform('C', np.log(1), np.log(100)),
        # 'gamma': hp.loguniform('gamma', np.log(0.001), np.log(0.1)),
        # 'epsilon': hp.loguniform('epsilon', np.log(0.001), np.log(0.1)),
        'C': hp.quniform('C', 50000, 650000, 10000),
        'gamma': hp.quniform('gamma', 0.001, 0.1, 0.001),
        'epsilon': hp.quniform('epsilon', 0.0001, 0.01, 0.0001),
        'kernel': 'rbf'
    }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=150)

    return best

# loading data
data_file = '/Users/xuxiang/Documents/house_price/price_train.csv'
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
print best_param

parameters = ['epsilon', 'C', 'gamma']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(40, 7))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.1, 0.4])

learn_fig = '/Users/xuxiang/Documents/house_price/learn.png'
savefig(learn_fig)
plt.show()
