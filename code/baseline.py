import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
from sklearn.metrics import roc_curve, auc
from hero_feats import HeroAttrs

rootdir = "../dota-2-prediction"
train_path = os.path.join(rootdir, "train.csv")
test_path = os.path.join(rootdir, "test.csv")
hero_path = os.path.join(rootdir, "hero_names.json")

train_dt = pd.read_csv(train_path, header=0, sep=",")
test_dt = pd.read_csv(test_path, header=0, sep=",")


def feature_preprocess(data):
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
    
    return np.concatenate([data, sub_features(data), sum_features(data), sum_sub_features(data), hero_features(data)],
                          axis=1)


def hero_roles_feat(data):
    hero_attr_md = HeroAttrs(hero_path)
    id_map_feats = hero_attr_md.make_feat()
    
    # hero_feats = np.zeros((data.shape[0], 24*10), dtype=float)
    # hero_feats = np.zeros((data.shape[0], 23*10), dtype=float)
    hero_feats = np.zeros((data.shape[0], 9 * 10), dtype=float)
    for i in range(data.shape[0]):
        for j in range(10):
            hero_idx = data[i, 1 + j * 8]
            # hero_feats[i,j*24:(j+1)*24] = id_map_feats[hero_idx]
            # hero_feats[i,j*23:(j+1)*23] = np.concatenate((id_map_feats[hero_idx][:2], id_map_feats[hero_idx][3:]))
            hero_feats[i, j * 9:(j + 1) * 9] = id_map_feats[hero_idx][15:]
    
    # print("before add hero attrs: ", data.shape)
    # print("after add hero attrs: ", data.shape)
    return np.concatenate((data, hero_feats), axis=1)


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


def load_data(path, sep=",", test=False):
    """data = np.genfromtxt(path, delimiter=sep, dtype=str)
    print(data[:5,:])
    data = data[1:,:]
    X = data[1:,:-1].astype(np.float)
    Y = data[1:,-1].astype(np.float)
    return (X,Y)"""
    data = pd.read_csv(path, header=0, sep=sep)
    # print(data.head(1))
    data = data.fillna(value=0.0)
    data_arr = data.values
    # print(data_arr.shape, data_arr[:3])
    if test:
        data_arr = hero_roles_feat(data_arr)
        return feature_preprocess(data_arr)
    X = data_arr[:, :-1]
    Y = data_arr[:, -1]
    X = np.concatenate([X, get_reverse_x(X)], axis=0)
    X = hero_roles_feat(X)
    Y = np.concatenate([Y, get_reverse_y(Y)], axis=0)
    return (feature_preprocess(X), Y)


def raw_feats_all(X, Y):
    # print(X.shape)
    # X = X[:300]
    # Y = Y[:300]
    N = X.shape[0]
    np.random.seed(1333)
    np.random.shuffle(X)
    np.random.seed(1333)
    np.random.shuffle(Y)
    
    train_X = X[:int(N * 0.8), :]
    train_Y = Y[:int(N * 0.8)]
    test_X = X[int(N * 0.8):, :]
    test_Y = Y[int(N * 0.8):]
    
    model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000
    )
    model.fit(train_X[:, 1:], train_Y,
              eval_set=[(test_X[:, 1:], test_Y)],
              eval_metric='logloss',
              early_stopping_rounds=50)
    
    y_pred = model.predict_proba(test_X[:, 1:])[:, 1]
    print(y_pred[:5])
    
    fpr, tpr, threshold = roc_curve(test_Y, y_pred, pos_label=1)
    auc_result = auc(fpr, tpr)
    print(auc_result)
    
    print('Feature importances:', list(model.feature_importances_))
    print('hero attrs feature importances: ', list(model.feature_importances_)[101:191])
    
    print('Ploting feature importance...')
    ax = lgb.plot_importance(model, max_num_features=20)
    try:
        plt.show()
    except:
        print("Can't display the picture")
    # print('Plot the 1st tree...')
    # ax = lgb.plot_tree(model, tree_index=1, figsize=(20, 8), show_info=['split_gain'])
    # plt.show()
    
    return model


def generate_solution(*models):
    print(test_path)
    X = load_data(test_path, test=True)
    print(X.shape)
    Ys = []
    for model in models:
        Ys.append(model.predict_proba(X[:, 1:])[:, 1])
        print(Ys[-1].shape)
    Ys = tuple(Ys)
    print(len(Ys))
    print(Ys[0].shape)
    ans = np.mean(Ys, axis=0)
    print(ans.shape)
    data = {
        'match_id': [int(x_row[0]) for x_row in X],
        'radiant_win': ans
    }
    data = pd.DataFrame(data)
    print('Generating submission file...')
    data.to_csv('../submission.csv', index=False)
    print('Done.')


if __name__ == "__main__":
    data_X, data_Y = load_data(train_path)
    model1 = raw_feats_all(data_X, data_Y)
    model2 = raw_feats_all(data_X, data_Y)
    model3 = raw_feats_all(data_X, data_Y)
    model4 = raw_feats_all(data_X, data_Y)
    model5 = raw_feats_all(data_X, data_Y)
    model6 = raw_feats_all(data_X, data_Y)
    model7 = raw_feats_all(data_X, data_Y)
    model8 = raw_feats_all(data_X, data_Y)
    model9 = raw_feats_all(data_X, data_Y)
    generate_solution(model1, model2, model3, model4, model5, model6, model7, model8, model9)
