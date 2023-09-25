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
import random

def plot_init():
    # 绘图格式设置
    plt.style.use('classic')
    matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    os.system('mkdir ' + imgPathPrefix + '碳因子模型分析')
    os.system('mkdir ' + imgPathPrefix + '能消模型分析')
    os.system('mkdir ' + imgPathPrefix + '总碳排放模型分析')
    os.system('mkdir ' + imgPathPrefix + '总碳排放模型预测')


booster_select = 'gblinear'
objective_select = 'reg:squarederror'
max_depth_select = 6
n_estimators_select = 500
learning_rate_select = 0.05

# 读取数据
data = pd.read_csv('data/carbon.csv')
imgPathPrefix = "/home/dknt/Projects/mathmodel/src/img2/"
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


#########################################################
'''
预测

'''
# predict_data = pd.read_csv('data/timeSerPred.csv')
predict_data = pd.read_csv('data/timeSerPred.csv')

predict_years = predict_data['年份'].size

# 各部门能源消费预测结果
# 自然情景
predict_features_engConsModel_1 = [['人口预测', '农林消费部门GDP预测', '农林消费部门能源强度预测'],  # 农林消费部门
                                 ['人口预测', '能源供应部门GDP预测', '能源供应部门能源强度预测'],  # 能源供应部门
                                 ['人口预测', '工业消费部门GDP预测', '工业消费部门能源强度预测'],  # 工业消费部门
                                 ['人口预测', '交通消费部门GDP预测', '交通消费部门能源强度预测'],  # 交通消费部门
                                 ['人口预测', '建筑消费部门GDP预测', '建筑消费部门能源强度预测'],  # 建筑消费部门
                                 ['人口预测', '总GDP预测', '总能源强度预测'],  # 居民生活消费
                                 ]
predict_features_cFactorModel_1 = [['农林消费部门化石能源占比预测', '非化石能源产热电占比预测'],  # 农林消费部门
                                 ['能源供应部门化石能源占比预测', '非化石能源产热电占比预测'],  # 能源供应部门
                                 ['工业消费部门化石能源占比预测', '非化石能源产热电占比预测'],  # 工业消费部门
                                 ['交通消费部门化石能源占比预测', '非化石能源产热电占比预测'],  # 交通消费部门
                                 ['建筑消费部门化石能源占比预测', '非化石能源产热电占比预测'],  # 建筑消费部门
                                 ['居民生活消费化石能源占比预测', '非化石能源产热电占比预测'],  # 居民生活消费
                                 ]
# 按时情景
predict_features_engConsModel_2 = [['人口预测', '农林消费部门GDP预测', '农林消费部门能源强度预测_按时'],  # 农林消费部门
                                 ['人口预测', '能源供应部门GDP预测', '能源供应部门能源强度预测_按时'],  # 能源供应部门
                                 ['人口预测', '工业消费部门GDP预测', '工业消费部门能源强度预测_按时'],  # 工业消费部门
                                 ['人口预测', '交通消费部门GDP预测', '交通消费部门能源强度预测_按时'],  # 交通消费部门
                                 ['人口预测', '建筑消费部门GDP预测', '建筑消费部门能源强度预测_按时'],  # 建筑消费部门
                                 ['人口预测', '总GDP预测', '总能源强度预测_按时'],  # 居民生活消费
                                 ]
predict_features_cFactorModel_2 = [['农林消费部门化石能源占比预测_按时', '非化石能源产热电占比预测_按时'],  # 农林消费部门
                                 ['能源供应部门化石能源占比预测_按时', '非化石能源产热电占比预测_按时'],  # 能源供应部门
                                 ['工业消费部门化石能源占比预测_按时', '非化石能源产热电占比预测_按时'],  # 工业消费部门
                                 ['交通消费部门化石能源占比预测_按时', '非化石能源产热电占比预测_按时'],  # 交通消费部门
                                 ['建筑消费部门化石能源占比预测_按时', '非化石能源产热电占比预测_按时'],  # 建筑消费部门
                                 ['居民生活消费化石能源占比预测_按时', '非化石能源产热电占比预测_按时'],  # 居民生活消费
                                 ]
# 雄心情景
predict_features_engConsModel_3 = [['人口预测', '农林消费部门GDP预测_雄心', '农林消费部门能源强度预测_雄心'],  # 农林消费部门
                                 ['人口预测', '能源供应部门GDP预测_雄心', '能源供应部门能源强度预测_雄心'],  # 能源供应部门
                                 ['人口预测', '工业消费部门GDP预测_雄心', '工业消费部门能源强度预测_雄心'],  # 工业消费部门
                                 ['人口预测', '交通消费部门GDP预测_雄心', '交通消费部门能源强度预测_雄心'],  # 交通消费部门
                                 ['人口预测', '建筑消费部门GDP预测_雄心', '建筑消费部门能源强度预测_雄心'],  # 建筑消费部门
                                 ['人口预测', '总GDP预测', '总能源强度预测_雄心'],  # 居民生活消费
                                 ]
predict_features_cFactorModel_3 = [['农林消费部门化石能源占比预测_雄心', '非化石能源产热电占比预测_雄心'],  # 农林消费部门
                                 ['能源供应部门化石能源占比预测_雄心', '非化石能源产热电占比预测_雄心'],  # 能源供应部门
                                 ['工业消费部门化石能源占比预测_雄心', '非化石能源产热电占比预测_雄心'],  # 工业消费部门
                                 ['交通消费部门化石能源占比预测_雄心', '非化石能源产热电占比预测_雄心'],  # 交通消费部门
                                 ['建筑消费部门化石能源占比预测_雄心', '非化石能源产热电占比预测_雄心'],  # 建筑消费部门
                                 ['居民生活消费化石能源占比预测_雄心', '非化石能源产热电占比预测_雄心'],  # 居民生活消费
                                 ]

def predict(predict_features_engConsModel, predict_features_cFactorModel, prefix):
    # 这里由于标签名字不一致，我们要重新计算
    predict_engCons = np.zeros([6, predict_years])  # 部门能源消耗预测
    predict_cFactor = np.zeros([6, predict_years])  # 部门碳排放系数预测
    predict_carbEmis = np.zeros([7, predict_years])  # 碳排放预测  第七行为总量
    # 预测各部门碳排放
    for i in range(6):  # 部门
        for j in range(predict_years):  # 年份
            engConsData_temp = [[predict_data[predict_features_engConsModel[i][0]][j] + random.gauss(0.0, 30),  # 人口
                                 predict_data[predict_features_engConsModel[i][1]][j],  # 部门GDP
                                 predict_data[predict_features_engConsModel[i][2]][j] * random.gauss(1.0, 0.05)]]  # 部门能源强度
            engConsIndex_temp = [features_engConsModel[i][0],
                                 features_engConsModel[i][1],
                                 features_engConsModel[i][2]]  # 改变标签名字
            engConsInput_temp = pd.DataFrame(engConsData_temp, [0], engConsIndex_temp)
            cFactorData_temp = [[predict_data[predict_features_cFactorModel[i][0]][j],  # 部门化石能源占比
                                 predict_data[predict_features_cFactorModel[i][1]][j] + random.gauss(0.0, 0.005)]]  # 非化石能源产热电占比
            cFactorIndex_temp = [features_cFactorModel[i][0],
                                 features_cFactorModel[i][1]]
            cFactorInput_temp = pd.DataFrame(cFactorData_temp, [0], cFactorIndex_temp)
            predict_engCons[i][j] = engConsModel[i].predict(engConsInput_temp)
            predict_cFactor[i][j] = cFactorModel[i].predict(cFactorInput_temp)
            predict_carbEmis[i][j] = predict_engCons[i][j] * predict_cFactor[i][j]


    # 预测结果绘图
    for i in range(6):
        name_prefix = '总碳排放模型预测/' + prefix + '_'
        # 能源消耗预测
        plt.plot(predict_data['年份'].values, predict_engCons[i], 'bo-')  # 预测值
        plt.ticklabel_format(style='plain')  # 不使用科学计数
        plt.xlabel('时间 - 年')
        plt.ylabel('能源消费量 - 万tec')
        plt.title(modelLabel[i] + '能源消费预测结果')
        plt.legend(labels=['预测值'], loc='best')
        plt.grid(True)
        plt.savefig(imgPathPrefix + name_prefix + '能源消费预测结果_' + modelLabel[i])  # 存图
        # plt.show()
        plt.clf()
        plt.close()
        # 碳排放因子预测
        plt.plot(predict_data['年份'].values, predict_cFactor[i], 'bo-')  # 预测值
        plt.ticklabel_format(style='plain')  # 不使用科学计数
        plt.xlabel('时间 - 年')
        plt.ylabel('碳排放因子 - tCO2/tce')
        plt.title(modelLabel[i] + '碳排放因子预测结果')
        plt.legend(labels=['预测值'], loc='best')
        plt.grid(True)
        plt.savefig(imgPathPrefix + name_prefix + '碳排放因子预测结果_' + modelLabel[i])  # 存图
        # plt.show()
        plt.clf()
        plt.close()

    # 计算总量
    for j in range(predict_years):
        sum_temp = 0.0
        for i in range(6):
            sum_temp += predict_carbEmis[i][j]
        predict_carbEmis[6][j] = sum_temp

    # 碳排放总量预测绘图
    plt.plot(predict_data['年份'].values, predict_carbEmis[6], 'bo-')  # 预测值
    plt.ticklabel_format(style='plain', axis='x')  # 不使用科学计数
    plt.xlabel('时间 - 年')
    plt.ylabel('碳排放量- 万tCO2')
    plt.title('碳排放总量预测结果')
    plt.legend(labels=['预测值'], loc='best')
    plt.grid(True)
    plt.savefig(imgPathPrefix + name_prefix + '碳排放总量预测结果')  # 存图
    plt.show()
    plt.clf()
    plt.close()


'''
临时绘图

'''
# plt.plot(data['年份'].values, data[6].values, 'bo-')  # 预测值


if __name__ == '__main__':
    plot_init()
    train_engConsModel()
    train_cFactorModel()
    # plot_engConsModelTrainResult()
    # plot_cFactorModelTrainResult()
    # shap_engConsModel()
    # shap_cFactorModel()
    # plot_totalCEmisTrainRes()
    predict(predict_features_engConsModel_1, predict_features_cFactorModel_1, '自然')
    predict(predict_features_engConsModel_2, predict_features_cFactorModel_2, '按时')
    predict(predict_features_engConsModel_2, predict_features_cFactorModel_2, '雄心')

