import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot  as plt
from matplotlib import font_manager
myfont = font_manager.FontProperties(fname=r".\utils\NotoSansCJK-Black.ttc")
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):
    pass

from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder
from utils.append_external_data import concat_externaldata
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from sklearn.svm import SVR
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , partial
import datetime
from lightgbm import log_evaluation, early_stopping



def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def MAPE(y, y_pred):
    return sum(abs(y_pred-y)/y)/len(y)

def Score(y, y_pred):
    y=np.expm1(y)
    y_pred=np.expm1(y_pred)
    hit_rate = np.around(np.sum(np.where(abs((y_pred-y)/y)<.1,1,0))/len(y),decimals=4)*10000
    MAPE = 1-np.sum(abs((y_pred-y)/y))/len(y)
    return hit_rate+MAPE

def Score_MAPE(y, y_pred):
    y=np.expm1(y)
    y_pred=np.expm1(y_pred)
#     hit_rate = np.around(np.sum(np.where(abs((y_pred-y)/y)<.1,1,0))/len(y),decimals=4)*10000
    MAPE = 1-np.sum(abs((y_pred-y)/y))/len(y)
    return MAPE

def Score_type(y, y_pred, h_type):
    y = np.expm1(y)
    y_pred = np.where(h_type>0,np.expm1(y_pred),0)
    hit_rate = np.around(np.sum(np.where(abs((y_pred-y)/y)<.1,1,0))/len(y),decimals=4)*10000
    MAPE = 1-np.sum(abs((y_pred-y)/y))/len(y)
    return hit_rate+MAPE

def Score_acc(y, y_pred, h_type):
    y = np.expm1(y)
    y_pred = np.where(h_type>0,np.expm1(y_pred),0)
    type_total = sum(np.where(h_type>0,1,0))
    hit_rate = np.sum(np.where(abs((y_pred-y)/y)<.1,1,0))/type_total
#     MAPE = 1-np.sum(abs((y_pred-y)/y))/len(y)
    return hit_rate#+MAPE

def Score2(y, y_pred):
    y=np.expm1(y)
    y_pred=np.expm1(y_pred)
#     hit_rate = np.around(np.sum(np.where(abs((y_pred-y)/y)<.1,1,0))/len(y),decimals=4)*10000
#     MAPE = 1-np.sum(abs((y_pred-y)/y))/len(y)
    hit_rate = np.sum(np.where(abs((y_pred-y)/y)<.1,1,0))/len(y)
    MAPE = np.around(1-np.sum(abs((y_pred-y)/y))/len(y),decimals=4)*10000
    return hit_rate+MAPE

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def check_result(search_result,param_space_list_2,n,plot_val=True,plot_tid=True,output_best = True):
    search_result.sort(key=lambda x : x["result"]["loss"]) # sort by search result
    
    for k in range(n):
        print("\nTop"+str(k+1)+" result :")
        print("val : ",search_result[k]["result"]["loss"])
        for i,j in search_result[k]["misc"]["vals"].items():
            print(i,param_space_list_2[i][j[0]])
    
    if plot_tid:
        print("\n")
        for i,j in param_space_list_2.items():
            if type(j) == type([]):
#                 print(i,j)
                f, ax = plt.subplots(1)
                xs = [param_space_list_2[i][t['misc']['vals'][i][0]] for t in search_result] 
                ys = [t["tid"] for t in search_result]
                ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
                ax.set_title('$t$ $vs$ '+i, fontsize=18)
                ax.set_xlabel(i, fontsize=16)
                ax.set_ylabel('$t$', fontsize=16)
                
    if plot_val:
        print("\n")
        for i,j in param_space_list_2.items():
            if type(j) == type([]):
                print(i,j)
                f, ax = plt.subplots(1)
                xs = [param_space_list_2[i][t['misc']['vals'][i][0]] for t in search_result] 
                ys = [t['result']['loss'] for t in search_result]
                ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
                ax.set_title('$val$ $vs$ '+i, fontsize=18)
                ax.set_xlabel(i, fontsize=16)
                ax.set_ylabel('$val$', fontsize=16)
    if output_best:
        best_result_param = param_space_list_2.copy()
        for i,j in search_result[0]["misc"]["vals"].items():
            best_result_param[i] = param_space_list_2[i][j[0]]
        return best_result_param
    
param_space_list = {
"boosting_type" : 'gbdt',
"objective" : 'regression',
"metric" : 'mape',
"learning_rate" : 0.005, 
"n_estimators" : 30000,
"num_leaves" : [250,300,350,375,400,425,450,500,550], 
"max_depth" : [-1], #[-1,4,5,6,7,8], 
"max_bin" : [ 250, 255, 300, 350,375, 400,425, 450, 500, 550, 600],
"min_data_in_leaf" : [0,1,2,3], 
"bagging_fraction" : [0.7,0.72,0.74,0.76,0.77,0.78,0.79,0.8,0.82,0.84,0.86], 
"bagging_freq" : [4,5,6,7, 10,12, 15], 
"feature_fraction" : [0.4,0.5,0.55,0.6,0.65,0.7,0.8], 
"feature_fraction_seed" : 1111, 
"bagging_seed" : 1111, 
"reg_lambda" : [1e-3,0.0,0.03,0.05,0.07,0.1,0.15],
"reg_alpha" : [1e-3,0.0,0.03,0.05,0.07,0.1,0.15],
# "min_split_gain" : [1e-5,1e-3,0.0,0.1,0.2,0.3,0.4,0.5],
"min_sum_hessian_in_leaf" : [0,1,2,3,4,5,6,7,8,9,10],
# "device": "cuda"
}

param_space_hyper = {
"boosting_type" : 'gbdt',
"objective" : 'regression',
"metric" : 'mape',
"learning_rate" : 0.005, 
"n_estimators" : 30000,
"num_leaves" :  hp.choice("num_leaves",[50,300,350,375,400,425,450,500,550]), 
"max_depth" : hp.choice("max_depth"  ,[-1]), 
"max_bin" :  hp.choice("max_bin" ,[ 250, 255, 300, 350,375, 400,425, 450, 500, 550, 600]),
"min_data_in_leaf" : hp.choice( "min_data_in_leaf" ,[0,1,2,3]), 
"bagging_fraction" : hp.choice("bagging_fraction" ,[0.7,0.72,0.74,0.76,0.77,0.78,0.79,0.8,0.82,0.84,0.86]), 
"bagging_freq" :  hp.choice("bagging_freq" ,[4,5,6,7, 10,12, 15]), 
"feature_fraction" :  hp.choice("feature_fraction" ,[0.4,0.5,0.55,0.6,0.65,0.7,0.8]), 
"feature_fraction_seed" : 1111, 
"bagging_seed" : 1111, 
"reg_lambda" :  hp.choice("reg_lambda" ,[1e-3,0.0,0.03,0.05,0.07,0.1,0.15]),
"reg_alpha" : hp.choice( "reg_alpha" ,[1e-3,0.0,0.03,0.05,0.07,0.1,0.15]),
# "min_split_gain" :  hp.choice("min_split_gain" ,[1e-5,1e-3,0.0,0.1,0.2,0.3,0.4,0.5]),
"min_sum_hessian_in_leaf" : hp.choice("min_sum_hessian_in_leaf" ,[0,1,2,3,4,5,6,7,8,9,10]),
# "device": "cuda"
}


def lgb_fine_tune(argsDict):
    
    model_lgb = lgb.LGBMRegressor(**argsDict, early_stopping_rounds=150, verbose=1)
    
    starttime = datetime.datetime.now()
    model_lgb.fit(X_train,train_y, eval_set=valid,eval_metric='mape')
    
    endtime = datetime.datetime.now()
    print ("Step_time:{}".format(endtime - starttime))
    lgb_valid_pred = model_lgb.predict(X_valid)
    lgb_train_pred = model_lgb.predict(train)
    rmsle_train = rmsle(y_train, lgb_train_pred)
    rmsle_valid = rmsle(valid_y, lgb_valid_pred)
    Score_train = Score(y_train, lgb_train_pred)
    Score_valid = Score(valid_y, lgb_valid_pred) 
    Score_MAPE_valid = Score_MAPE(valid_y, lgb_valid_pred) 
    mape_train = MAPE(y_train, lgb_train_pred)
    mape_valid = MAPE(valid_y, lgb_valid_pred)
    val = Score_MAPE_valid
    
    print("rmsle_train(val) = %.4f, rmsle_valid = %.4f, mape_valid = %.4f, Score_train = %.4f, Score_valid = %.4f \n"%(rmsle_train, rmsle_valid, Score_MAPE_valid*1000, Score_train,Score_valid))
    
    return {'loss': -val, 'status': STATUS_OK}



if __name__ == '__main__':
    train = pd.read_csv('./dataset/training_data.csv', sep=",")
    test = pd.read_csv('./dataset/public_dataset.csv', sep=",")
    # print(all_data.info())
    
    ntrain = train.shape[0]
    ntest = test.shape[0]
    all_data = pd.concat((train, test)).reset_index(drop=True)
#     print(all_data.columns)

    """New feature"""
    ### is_Top_floor
    all_data['top_floor'] = np.where((all_data['總樓層數'] == all_data['移轉層次']), 1, 0)
    ### new_town = city + town
    all_data['new_town'] = all_data['縣市'].apply(str) + '_' + all_data['鄉鎮市區'].apply(str)
    ### is_2_floor
    all_data['is_2_floor'] = np.where(all_data['移轉層次'] == 2, 1, 0)
    ### is_4_floor
    all_data['is_4_floor'] = np.where(all_data['移轉層次'] == 4, 1, 0)
    ### is_13_floor
    all_data['is_13_floor'] = np.where(all_data['移轉層次'] == 13, 1, 0)
    ### old_house (age >= 20)
    all_data['old_house'] = np.where(all_data['屋齡'] >= 20, 1, 0)
    ### high_floor
    all_data['percentage_floor'] = all_data['移轉層次'] / all_data['總樓層數']
    all_data['percentage_floor'] = np.where(all_data['percentage_floor'] >= 2/3, 'high', np.where(all_data['percentage_floor'] >= 1/3, 'median', 'low'))
    all_data.loc[(all_data['建物型態'] == '公寓(5樓含以下無電梯)') | (all_data['建物型態'] == '透天厝'), 'percentage_floor'] = 'None'
    
    """One hot encoding"""
    all_data["使用分區"] = all_data["使用分區"].fillna("None")
    all_data.drop(['備註', 'ID'], axis=1, inplace=True)
    one_hot_cols = ['主要建材', '縣市', '總樓層數', '主要用途', '車位個數', '建物型態', 'percentage_floor']
    all_data = pd.get_dummies(all_data, columns=one_hot_cols)
#     print(all_data.select_dtypes(include=['object']))

    category_col = ['鄉鎮市區', '路名', '使用分區', 'new_town']
    for c in category_col:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values)) 
        all_data[c] = lbl.transform(list(all_data[c].values))
    
    all_data['單價'] = np.log1p(all_data['單價'])
    all_data = concat_externaldata(all_data, "./dataset/external_data/ATM資料.csv", 'ATM資料', radius=300.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/大學基本資料.csv", '大學基本資料', radius=3000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/公車站點資料.csv", '公車站點資料', radius=300.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/火車站點資料.csv", '火車站點資料', radius=3000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/金融機構基本資料.csv", '金融機構基本資料', radius=1000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/便利商店.csv", '便利商店', radius=300.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/高中基本資料.csv", '高中基本資料', radius=3000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/國小基本資料.csv", '國小基本資料', radius=3000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/國中基本資料.csv", '國中基本資料', radius=3000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/捷運站點資料.csv", '捷運站點資料', radius=300.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/郵局據點資料.csv", '郵局據點資料', radius=1000.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/腳踏車站點資料.csv", '腳踏車站點資料', radius=300.0)
    all_data = concat_externaldata(all_data, "./dataset/external_data/醫療機構基本資料.csv", '醫療機構基本資料', radius=3000.0, mining_name='hospital')
    
    train = all_data[:ntrain]
    test = all_data[ntrain:].drop(['單價'], axis=1)
    
    train = train.drop(train[(train['土地面積']>10)].index)
    train = train.drop(train[(train['移轉層次']>=35)].index)
    train = train.drop(train[(train['建物面積']>=8)].index)
    train = train.drop(train[(train['車位面積']>=7)].index)
    train = train.drop(train[(train['主建物面積']>= 7)].index)
    train = train.drop(train[(train['陽台面積']>= 7)].index)
    train = train.drop(train[(train['附屬建物面積']>= 10)].index)
    
    y_train = train['單價']
    train = train.drop(['單價'], axis=1)
    X_train, X_valid, train_y, valid_y = train_test_split(train, y_train, test_size=0.1, random_state=49)#42
    
    valid=[(X_valid, valid_y)]
    
    trials = Trials()

    algo = partial(tpe.suggest,n_startup_jobs=1)
    best = fmin(lgb_fine_tune,param_space_hyper,algo=algo,max_evals=1200, trials=trials)
    print(best)
    
    aa = check_result(trials.trials,param_space_list,n=4,plot_val=False,plot_tid=False,output_best = True)
    
    model_lgb = lgb.LGBMRegressor(**aa, early_stopping_rounds=150, verbose=0)
    model_lgb.fit(X_train,train_y, eval_set=valid,eval_metric='mape')
    
    lgb_valid_pred = model_lgb.predict(X_valid)
    lgb_train_pred = model_lgb.predict(train)
    print(rmsle(valid_y, lgb_valid_pred))
    print(MAPE(valid_y, lgb_valid_pred))
    print(Score(valid_y, lgb_valid_pred))
    print(Score(y_train, lgb_train_pred))