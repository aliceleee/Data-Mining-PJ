import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import sys
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from dataloader import load_data


def split_data(X, Y, t=0.8, seed=1333):
    N = X.shape[0]
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)
    
    train_x = X[:int(N * t), :]
    train_y = Y[:int(N * t)]
    test_x = X[int(N * t):, :]
    test_y = Y[int(N * t):]
    
    return train_x, train_y, test_x, test_y


def feats_all(X, Y, show=True):
    train_x, train_y, test_x, test_y = split_data(X, Y)
    
    model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000
    )
    model.fit(train_x[:, 1:], train_y,
              eval_set=[(test_x[:, 1:], test_y)],
              eval_metric='logloss',
              early_stopping_rounds=50)
    
    y_pred = model.predict_proba(test_x[:, 1:])[:, 1]
    print("First 5 predicts:", y_pred[:5])
    
    fpr, tpr, threshold = roc_curve(test_y, y_pred, pos_label=1)
    auc_result = auc(fpr, tpr)
    print("auc =", auc_result)
    
    if show:
        print('Feature importances:', list(model.feature_importances_))
        print('hero attrs feature importances: ', list(model.feature_importances_)[101:191])
        
        print('Ploting feature importance...')
        ax = lgb.plot_importance(model, max_num_features=20)
        try:
            plt.show()
        except:
            print("Can't display the picture")
    
    return model


def baseline(X, Y):
    train_x, train_y, test_x, test_y = split_data(X, Y)
    
    model = XGBClassifier()
    model.fit(train_x, train_y)
    
    y_pred = model.predict(test_x)
    print("First 5 predicts:", y_pred[:5])
    
    fpr, tpr, threshold = roc_curve(test_y, y_pred, pos_label=1)
    auc_result = auc(fpr, tpr)
    print("auc =", auc_result)
    
    return model


def generate_solution(test_path, hero_path, *models):
    X = load_data(test_path, hero_path, test=True)
    Ys = []
    if isinstance(models[0], tuple):
        models = models[0]
    for model in models:
        Ys.append(model.predict_proba(X[:, 1:])[:, 1])
    Ys = tuple(Ys)
    ans = np.mean(Ys, axis=0)
    data = {
        'match_id': [int(x_row[0]) for x_row in X],
        'radiant_win': ans
    }
    data = pd.DataFrame(data)
    print('Generating submission file...')
    data.to_csv('../submission.csv', index=False)
    print('Done.')


if __name__ == "__main__":
    rootdir = "../dota-2-prediction"
    train = os.path.join(rootdir, "train.csv")
    test = os.path.join(rootdir, "test.csv")
    hero = os.path.join(rootdir, "hero_names.json")
    if len(sys.argv) > 1 and sys.argv[1] == '-b':
        data_X, data_Y = load_data(train, base=True)
        model0 = baseline(data_X, data_Y)
    elif len(sys.argv) > 1 and sys.argv[1] == '-s':
        data_X, data_Y = load_data(train)
        model = feats_all(data_X, data_Y, False)
        generate_solution(test, None, model)
    elif len(sys.argv) > 1 and sys.argv[1] == '-a':
        data_X, data_Y = load_data(train, hero)
        try:
            N = int(sys.argv[2])
        except:
            N = 1
        models = []
        for _ in range(N):
            models.append(feats_all(data_X, data_Y, False))
        models = tuple(models)
        generate_solution(test, hero, models)
    else:
        # 这是一个使用三个模型集成学习的例子，中间还会输出不同特征的重要程度
        data_X, data_Y = load_data(train, hero)
        model1 = feats_all(data_X, data_Y)
        model2 = feats_all(data_X, data_Y)
        model3 = feats_all(data_X, data_Y)
        generate_solution(test, hero, model1, model2, model3)
