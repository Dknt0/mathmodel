#!/usr/bin/python3
#
# 预测模型
#

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shap
import os

booster_select = 'gbtree'
objective_select = 'reg:squarederror'
max_depth_select = 6
n_estimators_select = 500
learning_rate_select = 0.05

# 读取数据
data = pd.read_csv('data/carbon.csv')
modelLabel = ['农林消费部门',
              '能源供应部门',
              '工业消费部门',
              '交通消费部门',
              '建筑消费部门',
              '居民生活']

######################################
'''
能源消费预测模型 按部门划分
'''
features_engConsModel = [['人口', '农林消费部门GDP', '农林消费部门能源强度'],  # 农林消费部门
                         ['人口', '能源供应部门GDP', '能源供应部门能源强度'],  # 能源供应部门
                         ['人口', '工业消费部门GDP', '工业消费部门能源强度'],  # 工业消费部门
                         ['人口', '交通消费部门GDP', '交通消费部门能源强度'],  # 交通消费部门
                         ['人口', '建筑消费部门GDP', '建筑消费部门能源强度'],  # 建筑消费部门
                         ['人口', '总GDP', '总能源强度'],  # 居民生活消费
                         ]

target_engConsModel = ['农林消费部门能源消费量',
                       '能源供应部门能源消费量',
                       '工业消费部门能源消费量',
                       '交通消费部门能源消费量',
                       '建筑消费部门能源消费量',
                       '居民生活能源消费量',
                       ]

# 训练能源消费预测模型  共6个
engConsModel = [0, 0, 0, 0, 0, 0]
def train_engConsModel():
    for i in range(6):
        engConsModel[i] = xgb.XGBRegressor(max_depth=max_depth_select, learning_rate=learning_rate_select,
                                           n_estimators=n_estimators_select,
                                           booster=booster_select, objective=objective_select)
        engConsModel[i].fit(data[features_engConsModel[i]], data[target_engConsModel[i]].values)

## 能耗预测 xgboost 模型 shap 解释
def shap_engConsModel():
    shap_values_engCons = [0, 0, 0, 0, 0, 0]
    shap_interaction_values_engCons = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        # shap 分析
        explainer = shap.TreeExplainer(engConsModel[i])
        shap_values_engCons[i] = explainer.shap_values(data[features_engConsModel[i]])
        data['pred_engCons' + modelLabel[i]] = engConsModel[i].predict(data[features_engConsModel[i]])
        # 影响程度总体分析 绘图
        shap.summary_plot(shap_values_engCons[i], data[features_engConsModel[i]])
        # 影响程度绝对值总体分析 绘图
        shap.summary_plot(shap_values_engCons[i], data[features_engConsModel[i]], plot_type="bar")
        # # 多变量交互分析
        shap_interaction_values_engCons[i] = shap.TreeExplainer(engConsModel[i]).shap_interaction_values(data[features_engConsModel[i]])
        shap.summary_plot(shap_interaction_values_engCons[i], data[features_engConsModel[i]], max_display=8)
        # 关闭窗口
        plt.close()


########################################
'''
碳排放因子预测模型 按部门划分
'''
features_cFactorModel = [['农林消费部门化石能源占比', '非化石能源产热电占比'],  # 农林消费部门
                         ['能源供应部门化石能源占比', '非化石能源产热电占比'],  # 能源供应部门
                         ['工业消费部门化石能源占比', '非化石能源产热电占比'],  # 工业消费部门
                         ['交通消费部门化石能源占比', '非化石能源产热电占比'],  # 交通消费部门
                         ['建筑消费部门化石能源占比', '非化石能源产热电占比'],  # 建筑消费部门
                         ['居民生活消费化石能源占比', '非化石能源产热电占比'],  # 居民生活消费
                         ]

target_cFactorModel = ['农林消费部门碳排放因子',
                       '能源供应部门碳排放因子',
                       '工业消费部门碳排放因子',
                       '交通消费部门碳排放因子',
                       '建筑消费部门碳排放因子',
                       '居民生活消费碳排放因子',
                       ]

# 训练能源消费预测模型  共6个
cFactorModel = [0, 0, 0, 0, 0, 0]
def train_cFactorModel():
    for i in range(6):
        cFactorModel[i] = xgb.XGBRegressor(max_depth=max_depth_select, learning_rate=learning_rate_select,
                                           n_estimators=n_estimators_select,
                                           booster=booster_select, objective=objective_select)
        cFactorModel[i].fit(data[features_cFactorModel[i]], data[target_cFactorModel[i]].values)


## 碳因子预测 xgboost 模型 shap 解释
def shap_cFactorModel():
    shap_values_cFactor = [0, 0, 0, 0, 0, 0]
    shap_interaction_values_cFactor = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        # shap 分析
        explainer = shap.TreeExplainer(cFactorModel[i])
        shap_values_cFactor[i] = explainer.shap_values(data[features_cFactorModel[i]])
        data['pred_cFactor' + modelLabel[i]] = cFactorModel[i].predict(data[features_cFactorModel[i]])
        # 影响程度总体分析 绘图
        shap.summary_plot(shap_values_cFactor[i], data[features_cFactorModel[i]])
        # 影响程度绝对值总体分析 绘图
        shap.summary_plot(shap_values_cFactor[i], data[features_cFactorModel[i]], plot_type="bar")
        # # 多变量交互分析
        shap_interaction_values_cFactor[i] = shap.TreeExplainer(cFactorModel[i]).shap_interaction_values(data[features_cFactorModel[i]])


if __name__ == '__main__':
    train_engConsModel()
    train_cFactorModel()
    shap_engConsModel()
    shap_cFactorModel()
