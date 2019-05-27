# Data Mining PJ

Code for kaggle competition: https://www.kaggle.com/c/dota-2-prediction

## First Version -- 0.614

我们简单得将所有的columns作为feature，并作为一个二分类任务送给XGBoost进行分类。

## Second Version -- 0.721

(+4%) 我们将输出的submission的二分类0/1输出，变为概率输出。

(+6%) 为了让英雄的使用情况体现在Feature中，我们将双方的10个英雄展开到 10 * 112 (英雄个数) * 7 (英雄feature) 维度上。

(+1%) 我们把交换双方后的数据也加入训练集中，使得训练数据翻倍。
