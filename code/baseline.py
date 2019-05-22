import pandas as pd 
import numpy as np 
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

rootdir = "../dota-2-prediction"
train_path = os.path.join(rootdir, "train.csv")
test_path = os.path.join(rootdir, "test.csv")

train_dt = pd.read_csv(train_path, header=0, sep=",")
test_dt = pd.read_csv(test_path, header=0, sep=",")

"""describe = train_dt.describe()
for c in train_dt:
    #print(c)
    print(describe[c])"""

def load_data(path, sep=","):
    """data = np.genfromtxt(path, delimiter=sep, dtype=str)
    print(data[:5,:])
    data = data[1:,:]
    X = data[1:,:-1].astype(np.float)
    Y = data[1:,-1].astype(np.float)
    return (X,Y)"""
    data = pd.read_csv(path, header=0, sep=sep)
    #print(data.head(1))
    data = data.fillna(value=0.0)
    data_arr = data.values
    #print(data_arr.shape, data_arr[:3])
    X = data_arr[:,:-1]
    Y = data_arr[:,-1]

    return (X,Y)

def raw_feats_all():
    X, Y = load_data(train_path)
    N = X.shape[0]
    np.random.seed(1333)
    np.random.shuffle(X)
    np.random.seed(1333)
    np.random.shuffle(Y)

    train_X = X[:int(N*0.8),:]
    train_Y = Y[:int(N*0.8)]
    test_X = X[int(N*0.8):,:]
    test_Y = Y[int(N*0.8):]

    model = XGBClassifier()
    model.fit(train_X, train_Y)

    y_pred = model.predict(test_X)
    print(y_pred[:5])

    fpr, tpr, threshold = roc_curve(test_Y, y_pred, pos_label=1)
    auc_result = auc(fpr, tpr)
    print(auc_result)

if __name__ == "__main__":
    raw_feats_all()
