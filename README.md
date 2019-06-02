# Data Mining PJ

Code for kaggle competition: https://www.kaggle.com/c/dota-2-prediction

## First Version -- 0.614

我们将所有的columns作为features，作为一个二分类任务送给XGBoost进行分类。

## Second Version -- 0.721

(+4%) 我们将输出的submission的二分类0/1输出，变为概率输出。

(+6%) 为了让英雄的使用情况体现在Feature中，我们将双方的10个英雄展开到 10 * 112 (英雄个数) * 7 (英雄feature) 维度上。

(+1%) 我们把交换双方后的数据也加入训练集中，使得训练数据翻倍。

## Third Version -- 0.723

(+0.2%) 对比赛中的每个角色使用的英雄添加onehot的roles特征

## Forth Version -- 0.725

(+0.2%) 使用3-9个相同的第三版的模型进行集成学习，结果逐渐接近0.725。

## 比赛结果

最终提交了使用9个相同的模型进行预测的代码。比赛过程中可见的 Public Score 为 0.72499； 最终结果的 Private Score 为 0.72380，排名第三。

## 运行代码

在 code 文件夹下运行 `pip install -r requirements` 安装依赖包；
运行 `python work.py` 加参数运行代码。

```shell
# 运行 First Version 的基线程序
python work.py -b

# 运行 Second Version 的程序
python work.py -s

# 运行 Third Version 和 Forth Version 的程序（N 为集成模型的数量，不写默认为1）
python work.py -a (N)
```
