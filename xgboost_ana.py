#!/usr/bin/python3
#
# xgboost 建模 shap 分析
#

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shap

plt.style.use('seaborn')
# 中文绘图
matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('data/carbon.csv')

# 选择特征
cols = ['人口',  # 人口
        '总GDP', '农林消费部门GDP', '能源供应部门GDP', '工业消费部门GDP', '交通消费部门GDP', '建筑消费部门GDP',  # GDP
        '农林消费部门能源消费量', '能源供应部门能源消费量', '工业消费部门能源消费量', '交通消费部门能源消费量', '建筑消费部门能源消费量', '居民生活能源消费量', # 能耗
         # 碳排放因子  数据不太好，暂时不考虑
        '人均GDP', '化石能源占比', '能源强度', '能源碳排强度', '碳排强度',
        '人口增长率', 'GDP增长率', '人均GDP增长率', '能源强度增长率', '能源碳排强度增长率'
        ]

# cols = ['总GDP', '人口', '能源消费总量']

# 训练模型
model = xgb.XGBRegressor(max_depth=8, learning_rate=0.05, n_estimators=150)
model.fit(data[cols], data['碳排放总量'].values)

# # 使用 feature importance 解释 xgboost
plt.figure(figsize=(10, 10))
plt.bar(range(len(cols)), model.feature_importances_)
plt.xticks(range(len(cols)), cols, rotation=-45, fontsize=14)
plt.title('特征重要程度', fontsize=14)
plt.show()

##############################################33
# 使用 shap 解释 xgboost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data[cols])
data['pred'] = model.predict(data[cols])

## 单个样本 shap 值   基于 JS  这个得在 jupyter 里运行
j = 0  # 样本序号
player_explainer = pd.DataFrame()
player_explainer['feature'] = cols
player_explainer['feature_value'] = data[cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]

# 单个绘图
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j])


## 总体分析
shap.summary_plot(shap_values, data[cols])
# 绝对值影响程度
shap.summary_plot(shap_values, data[cols], plot_type="bar")


## 部分依赖图
for temp in cols:
    shap.dependence_plot(temp, shap_values, data[cols], interaction_index=None, show=False)



## 多个变量交互分析
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[cols])
shap.summary_plot(shap_interaction_values, data[cols], max_display=4)

## 两个变量交互影响
shap.dependence_plot('GDP总量', shap_values, data[cols], interaction_index='人均能源消耗', show=False)