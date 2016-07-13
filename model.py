# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

## sklearn
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

## xgboost
import xgboost as xgb

## keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils


# 检测预测准确度的函数
def mean_absolute_percentage_error(y_true, y_pred):
    # convert pd.series to numpy.ndarray
    y_true = y_true.as_matrix()

    dim = y_true.shape
    mape = np.abs((y_true - y_pred) / y_true).sum() / float(dim[0])
    return mape

score_func = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


def mean_absolute_percentage_error2(preds, dtrain):
    labels = dtrain.get_label()
    dim = labels.shape

    error_val = np.abs((labels - preds) / labels).sum() / float(dim[0])
    return 'error', error_val


def dnn_score(param):
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
    model.fit(X_train_dnn, y_train_dnn, nb_epoch=int(param['nb_epoch']), batch_size=int(param['batch_size']), validation_split=0, verbose=0)

    ## prediction
    pred = model.predict(X_test_dnn, verbose=0)
    pred.shape = (X_test_dnn.shape[0],)
    print 'score: '
    score = mean_absolute_percentage_error(y_test_dnn, pred)

    print score

    return pred

# 加载数据
data_file = './price_train.csv'
data = pd.read_csv(data_file, sep=';')

# 填充缺失值
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
train_array = train.toarray()

## scale
scaler = StandardScaler()
dnn_train = scaler.fit_transform(train_array)

# split train and test data
X_train_dnn, X_test_dnn, y_train_dnn, y_test_dnn = train_test_split(dnn_train, labels, test_size=0.25, random_state=42)

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.25, random_state=42)

# 支持向量机模型
svr = SVR(kernel='rbf', C=650000, gamma=0.008, epsilon=0.0086)

# 使用交叉验证方式训练模型并取得错误率
scores = cross_validation.cross_val_score(svr, X_train, y_train, cv=5, n_jobs=-1, scoring=score_func)

# 输出平均错误率
mean_score = np.abs(np.mean(scores))
print('cross validate score is %f' % (mean_score))

# 训练模型
svr.fit(X_train, y_train)

# 预测
y_pred = svr.predict(X_test)
error = mean_absolute_percentage_error(y_test, y_pred)
print('svm predict error %f' % (error))

# xgboost 模型
param = {
    # 'lambda_bias': 1.2, 'alpha': 0.002, 'eta': 0.3, 'num_round': 376.0, 'lambda': 0.0
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'lambda_bias': 2.7, 'alpha': 0.019, 'eta': 0.98, 'num_round': 10.0, 'lambda': 0.35
}
num_rounds = int(param['num_round'])

# Create Train and Test DMatrix
xgtest = xgb.DMatrix(X_test, label=y_test)
xgtrain = xgb.DMatrix(X_train, label=y_train)

# train
watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
xgbmodel = xgb.train(param, xgtrain, num_rounds, watchlist, feval=mean_absolute_percentage_error2, verbose_eval=False)
y_pred_xgb = xgbmodel.predict(xgtest)
error_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
print('xgboost predict error %f' % (error_xgb))

#
param2 = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    # 'colsample_bytree': 0.3, 'min_child_weight': 4, 'subsample': 1.0, 'eta': 0.09, 'num_round': 970, 'max_depth': 5, 'gamma': 0.2
    'colsample_bytree': 0.2, 'min_child_weight': 0, 'subsample': 1.0, 'eta': 0.0384, 'num_round': 1760, 'max_depth': 8, 'gamma': 0.4
}

xgbmodel = xgb.train(param, xgtrain, num_rounds, watchlist, feval=mean_absolute_percentage_error2, verbose_eval=False)
y_pred_xgb_tree = xgbmodel.predict(xgtest)
error_xgb_tree = mean_absolute_percentage_error(y_test, y_pred_xgb_tree)
print('xgboost predict error %f' % (error_xgb_tree))

# dnn
param = {
    'hidden_units': 128, 'hidden_activation': 'prelu', 'batch_size': 128, 'input_dropout': 0.3, 'hidden_dropout': 0.3, 'hidden_layers': 4, 'nb_epoch': 30, 'batch_norm': False
}
dnn_pred = dnn_score(param)

avg_pred = (3 * y_pred + y_pred_xgb + y_pred_xgb_tree + 10 * dnn_pred) / 15.0
error_weighted = mean_absolute_percentage_error(y_test, avg_pred)
print('weighted predict error %f' % (error_weighted))
