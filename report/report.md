# 实验报告

## 小组分工

Alice

## Kaggle比赛说明 成绩截图

在比赛最终70%的数据集的排行榜上，我们最好的模型排名第三。

![privateLeaderboard](./privateLeaderboard.png)

在比赛过程中30%的数据集的排行榜上，我们最好的模型排名第四。

![publicLeaderboard](./publicLeaderboard.png)

## 代码和目录结构，运行文档

代码分为三个部分，均放置在 code 文件夹内。

+ work.py 入口程序，配置了运行选项

+ dataloader.py 包含数据读取和特征提取的代码

+ hero_feats.py 包含对英雄特征的提取代码

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

## 问题描述 数据描述 数据可视化

cky

## 算法

### 决策树 XGBoost LightGBM

Alice

### 特征提取

Alice

### Ensemble

zyn

### tricks

cky

## 运行结果和可视化(决策树可视化)

cky 或 zyn
