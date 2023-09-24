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

# 绘图格式设置
plt.style.use('classic')
matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False

imgPathPrefix = "/home/dknt/Projects/mathmodel/src/img2/"
os.system('mkdir ' + imgPathPrefix + '碳因子模型分析')
os.system('mkdir ' + imgPathPrefix + '能消模型分析')
os.system('mkdir ' + imgPathPrefix + '总碳排放模型分析')

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
                         ['人口', '总GDP', '能源强度'],  # 居民生活消费
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
for i in range(6):
    engConsModel[i] = xgb.XGBRegressor(max_depth=8, learning_rate=0.05, n_estimators=150)
    engConsModel[i].fit(data[features_engConsModel[i]], data[target_engConsModel[i]].values)


# 预测结果
engConsTrainPredict = [0, 0, 0, 0, 0, 0]
engConsTrainGT = [0, 0, 0, 0, 0, 0]
engConsErr = [0, 0, 0, 0, 0, 0]

# 绘图
for i in range(6):
    name_prefix = '能消模型分析/能消预测_'
    # 特征重要程度绘图
    # # 使用 feature importance 解释 xgboost
    plt.figure(figsize=(8, 8))
    plt.bar(range(len(features_engConsModel[i])), engConsModel[i].feature_importances_)
    plt.xticks(range(len(features_engConsModel[i])), features_engConsModel[i], rotation=0, fontsize=15)
    plt.title(modelLabel[i] + '能源消费预测模型 特征重要程度', fontsize=15)
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + '特征重要程度_' + modelLabel[i])  # 存图
    # plt.show()
    plt.clf()

    # 预测结果绘图
    engConsTrainPredict[i] = engConsModel[i].predict(data[features_engConsModel[i]])
    engConsTrainGT[i] = data[target_engConsModel[i]].values
    plt.plot(data['year'].values, engConsTrainPredict[i], 'bo-')  # 预测值
    plt.plot(data['year'].values, engConsTrainGT[i], 'ro--')  # GroundTruth
    plt.xlabel('时间 - 年')
    plt.ylabel('能源消费量 - 万tec')
    plt.title(modelLabel[i] + '能源消费预测结果')
    plt.legend(labels=['预测值', '真实值'], loc='best')
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + '能源消费预测结果_' + modelLabel[i])  # 存图
    # plt.show()
    plt.clf()

    # 拟合误差图表
    engConsErr[i] = engConsTrainPredict[i] - engConsTrainGT[i]
    plt.plot(data['year'].values, engConsErr[i], 'r*--')  # 误差
    plt.xlabel('时间 - 年')
    plt.ylabel('能源消费预测误差 - 万tec')
    plt.title(modelLabel[i] + '能源消费预测误差')
    plt.legend(labels=['预测误差'], loc='best')
    plt.grid(True)
    plt.ylim((-150, 150))
    plt.savefig(imgPathPrefix + name_prefix + '能源消费预测误差_' + modelLabel[i])  # 存图
    plt.clf()
    # 关闭窗口
    plt.close()

## 能耗预测 xgboost 模型 shap 解释
shap_values_engCons = [0, 0, 0, 0, 0, 0]
shap_interaction_values_engCons = [0, 0, 0, 0, 0, 0]
for i in range(6):
    # shap 分析
    name_prefix = '能消模型分析/能消预测_'
    explainer = shap.TreeExplainer(engConsModel[i])
    shap_values_engCons[i] = explainer.shap_values(data[features_engConsModel[i]])
    data['pred_engCons' + modelLabel[i]] = engConsModel[i].predict(data[features_engConsModel[i]])
    # 影响程度总体分析 绘图
    shap.summary_plot(shap_values_engCons[i], data[features_engConsModel[i]])
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + 'shap影响程度总体分析_' + modelLabel[i])  # 存图
    plt.clf()
    # 影响程度绝对值总体分析 绘图
    shap.summary_plot(shap_values_engCons[i], data[features_engConsModel[i]], plot_type="bar")
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + 'shap影响程度绝对值总体分析_' + modelLabel[i])  # 存图
    plt.clf()
    # # 多变量交互分析
    # shap_interaction_values_engCons[i] = shap.TreeExplainer(engConsModel[i]).shap_interaction_values(data[features_engConsModel[i]])
    # shap.summary_plot(shap_interaction_values_engCons[i], data[features_engConsModel[i]], max_display=8)
    # plt.savefig(imgPathPrefix + name_prefix + 'shap多变量交互分析分析_' + modelLabel[i])  # 存图
    # plt.clf()
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
for i in range(6):
    cFactorModel[i] = xgb.XGBRegressor(max_depth=8, learning_rate=0.05, n_estimators=150)
    cFactorModel[i].fit(data[features_cFactorModel[i]], data[target_cFactorModel[i]].values)


# 绘图
# 预测结果
cFactorTrainPredict = [0, 0, 0, 0, 0, 0]
cFactorTrainGT = [0, 0, 0, 0, 0, 0]
cFactorErr = [0, 0, 0, 0, 0, 0]

# 绘图
for i in range(6):
    name_prefix = '碳因子模型分析/碳因子预测_'
    # 特征重要程度绘图
    # # 使用 feature importance 解释 xgboost
    plt.figure(figsize=(8, 8))
    plt.bar(range(len(features_cFactorModel[i])), cFactorModel[i].feature_importances_)
    plt.xticks(range(len(features_cFactorModel[i])), features_cFactorModel[i], rotation=0, fontsize=15)
    plt.title(modelLabel[i] + '碳排放因子预测模型 特征重要程度', fontsize=15)
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + '特征重要程度_' + modelLabel[i])  # 存图
    # plt.show()
    plt.clf()

    # 预测结果绘图
    cFactorTrainPredict[i] = cFactorModel[i].predict(data[features_cFactorModel[i]])
    cFactorTrainGT[i] = data[target_cFactorModel[i]].values
    plt.plot(data['year'].values, cFactorTrainPredict[i], 'bo-')  # 预测值
    plt.plot(data['year'].values, cFactorTrainGT[i], 'ro--')  # GroundTruth
    plt.xlabel('时间 - 年')
    plt.ylabel('碳排放因子 - tCO2/tce')
    plt.title(modelLabel[i] + '碳排放因子预测结果')
    plt.legend(labels=['预测值', '真实值'], loc='best')
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + '碳排放因子预测结果_' + modelLabel[i])  # 存图
    # plt.show()
    plt.clf()

    # 拟合误差图表
    cFactorErr[i] = cFactorTrainPredict[i] - cFactorTrainGT[i]
    plt.plot(data['year'].values, cFactorErr[i], 'r*--')  # 误差
    plt.xlabel('时间 - 年')
    plt.ylabel('碳排放因子预测误差 - tCO2/tce')
    plt.title(modelLabel[i] + '碳排放因子预测误差')
    plt.legend(labels=['预测误差'], loc='best')
    plt.grid(True)
    plt.ylim((-0.1, 0.1))
    plt.savefig(imgPathPrefix + name_prefix + '碳排放因子预测误差_' + modelLabel[i])  # 存图
    plt.clf()
    # 关闭窗口
    plt.close()

## 碳因子预测 xgboost 模型 shap 解释
shap_values_cFactor = [0, 0, 0, 0, 0, 0]
shap_interaction_values_cFactor = [0, 0, 0, 0, 0, 0]
for i in range(6):
    # shap 分析
    name_prefix = '碳因子模型分析/碳因子预测_'
    explainer = shap.TreeExplainer(cFactorModel[i])
    shap_values_cFactor[i] = explainer.shap_values(data[features_cFactorModel[i]])
    data['pred_cFactor' + modelLabel[i]] = cFactorModel[i].predict(data[features_cFactorModel[i]])
    # 影响程度总体分析 绘图
    shap.summary_plot(shap_values_cFactor[i], data[features_cFactorModel[i]])
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + 'shap影响程度总体分析_' + modelLabel[i])  # 存图
    plt.clf()
    # 影响程度绝对值总体分析 绘图
    shap.summary_plot(shap_values_cFactor[i], data[features_cFactorModel[i]], plot_type="bar")
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + 'shap影响程度绝对值总体分析_' + modelLabel[i])  # 存图
    plt.clf()
    # # 多变量交互分析
    shap_interaction_values_cFactor[i] = shap.TreeExplainer(cFactorModel[i]).shap_interaction_values(data[features_cFactorModel[i]])
    shap.summary_plot(shap_interaction_values_cFactor[i], data[features_cFactorModel[i]], max_display=8)
    plt.savefig(imgPathPrefix + name_prefix + 'shap多变量交互分析分析_' + modelLabel[i])  # 存图
    plt.clf()
    # 关闭窗口
    plt.close()

# 总碳排放预测绘图
carbEmisPredict = np.zeros([7, 11])  # 碳排放预测  第七行为总量
carbEmisGT = np.zeros([1, 11])  # GT

# 预测各部门碳排放
for i in range(6):  # 部门
    for j in range(11):  # 年份
        engConsData_temp = [[data[features_engConsModel[i][0]][j],  # 人口
                             data[features_engConsModel[i][1]][j],  # 部门GDP
                             data[features_engConsModel[i][2]][j]]]  # 部门能源强度
        engConsIndex_temp = [features_engConsModel[i][0],
                             features_engConsModel[i][1],
                             features_engConsModel[i][2]]
        engConsInput_temp = pd.DataFrame(engConsData_temp, [0], engConsIndex_temp)
        cFactorData_temp = [[data[features_cFactorModel[i][0]][j],  # 部门化石能源占比
                             data[features_cFactorModel[i][1]][j]]]  # 非化石能源产热电占比
        cFactorIndex_temp = [features_cFactorModel[i][0],
                             features_cFactorModel[i][1]]
        cFactorInput_temp = pd.DataFrame(cFactorData_temp, [0], cFactorIndex_temp)
        carbEmisPredict[i][j] = engConsModel[i].predict(engConsInput_temp) * cFactorModel[i].predict(cFactorInput_temp)

# 计算总量
for j in range(11):
    sum_temp = 0.0
    for i in range(6):
        sum_temp += carbEmisPredict[i][j]
    carbEmisPredict[6][j] = sum_temp

carbEmisGT = data['碳排放总量'].values
carbEmisGT += data['能源供应部门碳排放量'].values

name_prefix = '总碳排放模型分析/'
plt.plot(data['year'].values, carbEmisPredict[6], 'bo-')  # 预测值
plt.plot(data['year'].values, carbEmisGT, 'ro--')  # GroundTruth
plt.xlabel('时间 - 年')
plt.ylabel('碳排放量- 万tCO2')
plt.title('碳排放总量预测结果')
plt.legend(labels=['预测值', '真实值'], loc='best')
plt.grid(True)
plt.savefig(imgPathPrefix + name_prefix + '碳排放总量预测结果')  # 存图
# plt.show()
plt.clf()
plt.close()

'''
预测

'''


'''
1

'''
