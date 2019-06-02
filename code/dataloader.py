import numpy as np
import pandas as pd
from hero_feats import hero_roles_feat


def feature_preprocess(raw_data):
    def sum_features(data):
        return np.stack(
            [
                data[:, i] + data[:, i + 8] + data[:, i + 16] + data[:, i + 24] + data[:, i + 32]
                for i in range(2, 9)
            ]
            +
            [
                data[:, i] + data[:, i + 8] + data[:, i + 16] + data[:, i + 24] + data[:, i + 32]
                for i in range(42, 49)
            ]
            , axis=1)
    
    def sub_features(data):
        return np.stack([data[:, i] - data[:, i + 40] for i in range(1, 41)], axis=1)
    
    def sum_sub_features(data):
        tdata = sum_features(data)
        return np.stack([tdata[:, i] - tdata[:, i + 7] for i in range(7)], axis=1)
    
    def hero_features(data):
        tdata = []
        for i in range(data.shape[0]):
            tline = [-1.] * 14 * 112
            for ti in range(5):
                hero_id = int(data[i][ti * 8 + 1] - 1)
                assert hero_id <= 112
                tline[hero_id * 7: (hero_id + 1) * 7] = data[i][ti * 8 + 2: (ti + 1) * 8 + 1]
            for ti in range(5):
                hero_id = int(data[i][40 + ti * 8 + 1] - 1)
                assert hero_id <= 112
                tline[7 * 112 + hero_id * 7: 7 * 112 + (hero_id + 1) * 7] = data[i][
                                                                            40 + ti * 8 + 2: 40 + (ti + 1) * 8 + 1]
            tdata.append(np.array(tline))
        tdata = np.array(tdata)
        return np.array(tdata)
    
    return np.concatenate([raw_data, sub_features(raw_data), sum_features(raw_data), sum_sub_features(raw_data),
                           hero_features(raw_data)], axis=1)


def get_reverse_x(X):
    tx = X.copy()
    for i in range(tx.shape[0]):
        tx[i][1:41], tx[i][41:81] = X[i][41:81], X[i][1:41]
        tx[i][85:93], tx[i][93:101] = X[i][93:101], X[i][85:93]
        if X[i][82] < 0.5:
            tx[i][82] = 1.0
        else:
            tx[i][82] = 0.0
        
        if X[i][83] <= 4.5:
            tx[i][83] = tx[i][83] + 5
        else:
            tx[i][83] = tx[i][83] - 5
        
        if X[i][84] <= 4.5:
            tx[i][84] = tx[i][84] + 5
        else:
            tx[i][84] = tx[i][84] - 5
    return tx


def get_reverse_y(Y):
    ty = Y.copy()
    for i in range(ty.shape[0]):
        if Y[i] < 0.5:
            ty[i] = 1.0
        else:
            ty[i] = 0.0
    return ty


def load_data(path, hero_path=None, sep=",", test=False, base=False):
    data = pd.read_csv(path, header=0, sep=sep)
    data = data.fillna(value=0.0)
    data_arr = data.values
    if base:
        return data_arr[:, :-1], data_arr[:, -1]
    if test:
        if hero_path is not None:
            data_arr = hero_roles_feat(data_arr, hero_path)
        return feature_preprocess(data_arr)
    X = data_arr[:, :-1]
    Y = data_arr[:, -1]
    X = np.concatenate([X, get_reverse_x(X)], axis=0)
    if hero_path is not None:
        X = hero_roles_feat(X, hero_path)
    Y = np.concatenate([Y, get_reverse_y(Y)], axis=0)
    return feature_preprocess(X), Y
