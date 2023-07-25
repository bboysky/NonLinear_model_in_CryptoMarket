import os,json,requests,time,argparse,importlib

from datetime import datetime,timedelta

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from multiprocessing import Process
from multiprocessing import Pool

import xgboost as xgb
import lightgbm as lgb

from MiGo.backtest import light_backtest_engine

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from MiGo import set_jupyter_default_env
set_jupyter_default_env()
from MiGo.client import InternalClient

import warnings
warnings.filterwarnings('ignore')

c = InternalClient()

'''
Migo method
'''
def str_stamp(str_getting):
    timeArray = time.strptime(str_getting, "%Y%m%d")
    str_getting_stamp = int(time.mktime(timeArray))//86400 * 86400
    return str_getting_stamp

def get_close(symbol_list,start,end,type_):
    
    sym_data_info = c.get_valid_kline_symbol_info()
    tmp_dict = {}
    close_df = list()

    for sym in symbol_list:
        # print(sym)
        if not sym in sym_data_info.keys() or str_stamp(end)<sym_data_info[sym]['ts_min']:
            continue
        if str_stamp(start)>sym_data_info[sym]['ts_min']:
            start_ = start
        else:
            start_ = time.strftime("%Y-%m-%d",time.localtime(sym_data_info[sym]['ts_min']))
        tmp_dict[sym] = c.get_kline(sym,start_,end).set_index("datetime")[type_]
        
    close_df = pd.DataFrame(tmp_dict)

    return close_df

def calculate_factor_value(factor_class, params_setting, start, end,symbol_list):
    
    def one_para(factor_class, params_setting,symbol_list,start,end):
        _l = list()
        fac_dict = {}
        for symbol_name in symbol_list:
            task_setting = dict(data_symbol = symbol_name, params = params_setting, version="v1",timeout=60)
            sym_data_info = c.get_valid_kline_symbol_info()
            if not symbol_name in sym_data_info.keys() or str_stamp(end)<sym_data_info[symbol_name]['ts_min']:
                continue
            if str_stamp(start)>sym_data_info[symbol_name]['ts_min']:
                start_ = start
            else:
                start_ = time.strftime("%Y-%m-%d",time.localtime(sym_data_info[symbol_name]['ts_min']))
            factor_data = factor_class.core_logic(start_,end,task_setting).set_index("datetime")['value']
            fac_dict[symbol_name] = factor_data.loc[start:end]

        return pd.DataFrame(fac_dict)

    factor_dict = dict()
    
    for param_id, params_setting in params_setting.items():
        factor_dict[param_id] = one_para(factor_class, params_setting, symbol_list, start, end)
        
    return factor_dict



def cta_sig_v2(df,price,profit,loss,hold_n,type_):
    
    num = int(hold_n*5)
    for sym in df.columns:
        count = 0
#         print(sym)
        sym_array = df[sym].values[:-num]
        index_ = np.where(sym_array==type_)[0]
        kline_array = price[sym].values
        for i in index_:
            
            check_kline = sym_array[i+1:i+1+num]
            index_inbar = np.where(check_kline==type_)[0]
            if len(index_inbar)==0:
                num_check = num
                tmp=(kline_array[i+1:i+num_check]/kline_array[i+1]-1) * sym_array[i]
                tmp[(tmp<-loss) | (tmp>profit)] = 1
                tmp[-1] = 1
                len_sig = np.where(tmp==1)[0]
                first_one_index = len_sig[0]+1
                df[sym][i:][first_one_index] = 0
                if len(len_sig)>1:
                    count+=1
            else:
                num_check = index_inbar[0]
                tmp=(kline_array[i+1:i+num_check]/kline_array[i+1]-1) * sym_array[i]
                tmp[(tmp<-loss) | (tmp>profit)] = 1
                len_sig = np.where(tmp==1)[0]
                if len(len_sig)>=1:
                    first_one_index = len_sig[0]+1
                    df[sym][i:][first_one_index] = 0
                    count+=1
        # exit_count = count/len(index_)
#         print(sym,exit_count)
    return df

'''
判断特征矩阵有没有inf和null
'''
def check_factor_inf_null(df,check_inf,check_null):
    status_=0
    if check_inf:
        is_inf = ((df == np.inf).sum().max()) + ((df == -1*np.inf).sum().max())#统计缺失最多的品种
        if is_inf>0:
            ratio = is_inf/df.shape[0]  # 占比
            if ratio>0.01:
                print("存在inf值，品种最大占比多：%s,请清洗数据"%ratio)
            else:
                print("存在inf值，品种最大占比：%s"%ratio)
            status_ = 1
    if check_null:
        is_null = ((df.isnull()).sum().max())
        if is_null>0:
            ratio = is_null/df.shape[0]  # 占比
            if ratio<0.01:
                print("存在null值，品种最大占比：%s"%ratio)
            else:
                print("存在null值，品种最大占比多：%s,请清洗数据"%ratio)
            status_ = 1
    return status_


def TimeSeries_ic(df,price,N):

    pct = price.pct_change(N).shift(-N)
    pct = pct.reindex_like(df)
    person_s = df.corrwith(pct)
    return person_s.mean()

def cumsum_min(s):
    return s.cumsum().min()
def cumsum_max(s):
    return s.cumsum().max()

# longshort
def sig_2_TragetPos(sig,q_ratio):
    sig_rank_bottom = sig.rank(axis=1,ascending=True)
    sig_rank_top = sig.rank(axis=1,ascending=False)

    num = int(sig.shape[1]*q_ratio)+1

    sig_rank_bottom = np.sign(sig_rank_bottom - num)
    sig_rank_bottom[sig_rank_bottom>0] = 0

    sig_rank_top = np.sign(sig_rank_top - num)
    sig_rank_top[sig_rank_top>0] = 0

    sig_rank = abs(sig_rank_top)+sig_rank_bottom
    
    return sig_rank

def bcak_test(sig_df,price,fee):
    backtest = light_backtest_engine(sig_df.fillna(0),price,0,fee,init_usdt=1000000)

    backtest.run_backtesting()
    BT = backtest.get_backtesting_result()

    result_dict = backtest.cumsum_return_analysis(BT['total_pnl'],print_out=False)
    result_dict.pop('draw_back_series')
    return BT,result_dict

def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
        if t == 0:
            mins, secs = divmod(t, 60)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            print(timeformat, end='\r', flush = True)
    return

def atr(df, symbol, n=14):
    df_symbol = df.loc[df.index.get_level_values('symbol') == symbol]
    high = df_symbol['high']
    low = df_symbol['low']
    close = df_symbol['close']
    df_symbol['tr0'] = abs(high - low)
    df_symbol['tr1'] = abs(high - close.shift(1))
    df_symbol['tr2'] = abs(low - close.shift(1))
    tr = df_symbol[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr

def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    #regress = stats.linregress(x, log_ts)
    mask = ~np.isnan(x) & ~np.isnan(log_ts)
    regress = stats.linregress(x[mask], log_ts[mask])
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def lgb_precision_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'precision_score', precision_score(y_true, y_hat), True


def train_valid_test_split(df, start_date, start_date_valid, end_date_valid, start_date_test, end_date_test):
    X_y_train = df[start_date : start_date_valid]
    X_y_valid = df[start_date_valid +  pd.Timedelta('1 day'): end_date_valid]
    X_y_test = df[start_date_test +  pd.Timedelta('1 day'): end_date_test]
    return X_y_train, X_y_valid, X_y_test

def train_valid_split(df, start_date, start_date_valid, end_date_valid):
    X_y_train = df[start_date : start_date_valid]
    X_y_valid = df[start_date_valid +  pd.Timedelta('1 day'): end_date_valid]
    return X_y_train, X_y_valid

def class_switch_binary(y_valid, y_pred, prob_threshold):
    result = []
    for prob in y_pred:
        if prob > float(prob_threshold):
            result.append(1)
        else: result.append(0)
    result_df = y_valid.copy()
    result_df = result_df.to_frame()
    #result_df.reset_index(level=0, inplace=True)
    result_df['pred'] = result
    return result_df['pred']

def downsample(X_y_train, target_col, test_ratio, random_seed):
    df_positive = X_y_train.loc[X_y_train[target_col]==1]
    df_negative = X_y_train.loc[X_y_train[target_col]==0]
    df_negative_bigger, df_negative_downsampled = train_test_split(df_negative,
                                                               test_size=test_ratio, random_state=random_seed)
    X_y_train_resampled = pd.concat([df_positive, df_negative_downsampled])
    X_y_train_resampled = X_y_train_resampled.sort_index()
    return X_y_train_resampled

def feature_target_split(df, features_cols, target_col):
    X_train = df[features_cols]
    y_train = df[target_col]
    return X_train, y_train

def lgb_train(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, metric = 'auc', prob_threshold = 0.7, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    max_precision = float("-inf")
    max_auc = float("-inf")
    max_precision_total_gain = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'binary',
                'metric': metric,
                'is_unbalance': 'false',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            #print("Using ", metric)
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_f1_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold)
            precision = precision_score(y_valid, y_class_pred_var_threshold)
            auc = roc_auc_score(y_valid, y_class_pred_var_threshold)
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            if precision > max_precision:
                max_precision = precision
                best_pres_model = model
                optimal_precision_depth = max_depth
                optimal_precision_num_leaves = num_leaves
                max_precision_total_gain = total_gain
            if auc > max_auc:
                max_auc = auc
                best_auc_model = model
                optimal_auc_depth = max_depth
                optimal_auc_num_leaves = num_leaves
                max_auc_total_gain = total_gain
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
                optimal_depth = max_depth
                optimal_num_leaves= num_leaves
    #print("max auc = ", max_auc, " at depth = ", optimal_auc_depth, " and num_leaves = ", optimal_auc_num_leaves,
    #      ' with total gain = ', max_auc_total_gain)
    return (best_model, best_pres_model, max_total_gain,
    optimal_depth, optimal_num_leaves, max_precision, optimal_precision_depth, optimal_precision_num_leaves, max_precision_total_gain)

def lgb_train_auc(X_train, y_train, X_valid, y_valid, X_valid_close, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, metric = 'auc', prob_threshold = 0.7, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    max_precision = float("-inf")
    max_auc = float("-inf")
    max_precision_total_gain = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'binary',
                'metric': metric,
                'is_unbalance': 'false',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            #print("Using ", metric)
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_f1_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            #prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold)
            #precision = precision_score(y_valid, y_class_pred_var_threshold)
            auc = roc_auc_score(y_valid, y_class_pred_var_threshold)
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            # if precision > max_precision:
            #     max_precision = precision
            #     best_pres_model = model
            #     optimal_precision_depth = max_depth
            #     optimal_precision_num_leaves = num_leaves
            #     max_precision_total_gain = total_gain
            if auc > max_auc:
                max_auc = auc
                best_auc_model = model
                optimal_auc_depth = max_depth
                optimal_auc_num_leaves = num_leaves
                max_auc_total_gain = total_gain
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
                optimal_depth = max_depth
                optimal_num_leaves= num_leaves
    #print("max auc = ", max_auc, " at depth = ", optimal_auc_depth, " and num_leaves = ", optimal_auc_num_leaves,
    #      ' with total gain = ', max_auc_total_gain)
    return (best_model, best_auc_model, max_total_gain,
    optimal_depth, optimal_num_leaves, max_auc, optimal_auc_depth, optimal_auc_num_leaves, max_auc_total_gain)

def lgb_train_feature_importance(X_train, y_train, X_valid, y_valid, max_depth_range, num_leaves_range, return_col_actual,
              min_data = 11, metric = 'auc', prob_quantile = 0.85, sign = 1):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    max_total_gain = float("-inf")
    for max_depth in max_depth_range:
        for num_leaves in num_leaves_range:
            parameters = {
                'application': 'binary',
                'metric': metric,
                'is_unbalance': 'false',
                #'scale_pos_weight': 9,
                'boosting': 'gbdt',
                'num_leaves': num_leaves,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.2,
                'bagging_freq': 20,
                'learning_rate': 0.1,
                'verbose': -1,
                'min_data_in_leaf': min_data,
                'max_depth': max_depth
            }
            model = lgb.train(parameters,
                                   train_data,
                                   valid_sets=valid_data,
                                   num_boost_round=5000,
                                    verbose_eval=False,
                                   #feval=lgb_precision_score,
                                   early_stopping_rounds=100)
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            prob_threshold = pd.DataFrame(y_pred).quantile(q=prob_quantile)
            y_class_pred_var_threshold = class_switch_binary(y_valid, y_pred, prob_threshold[0])
            X_valid_close_pred = pd.merge(X_valid_close, y_class_pred_var_threshold, left_index=True, right_index=True)
            X_valid_close_pred['gain'] = X_valid_close_pred[return_col_actual] * X_valid_close_pred.pred
            total_gain = X_valid_close_pred.groupby(level='symbol').gain.sum().sum() * sign
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_model = model
    print("----------------------")
    return best_model
def multi_lgb_predict(models, X, y):
    df = X.copy()
    cols = []
    for j, model in enumerate(models, 1):
        if type(model).__name__ == 'KNeighborsClassifier':
            df['y_pred_{}'.format(str(j))] = [item[1] for item in list(model.predict_proba(X))]
            cols.append('y_pred_{}'.format(str(j)))
        else:
            df['y_pred_{}'.format(str(j))] = model.predict(X, num_iteration=model.best_iteration)
            cols.append('y_pred_{}'.format(str(j)))
    df = df[cols]
    df['target'] = y
    return df, cols

def reclass_rank(y_pred, sym_index, quota_dict, class_dict):  #JEFF
    y_pred_ = y_pred.copy()
    zero_indexes = [key for key, val in quota_dict.items() if val == 0]
    for i in zero_indexes:
        y_pred_[0][i] = -1
    max_prob_index = np.argmax(y_pred_[0])
    rank = np.argsort(np.argsort(-y_pred_, axis=0), axis=0) + 1
    if rank[0][max_prob_index] <= quota_dict[max_prob_index]:
        quota_dict[max_prob_index] -= 1
        y_pred2 = y_pred_[1:]
        sym_index2 = sym_index[1:]
        class_dict[max_prob_index].append(sym_index[0])
    else:
        #print(len(y_pred_), y_pred_, rank[0], max_prob_index, quota_dict)
        y_pred2 = np.concatenate((y_pred_[1:], y_pred_[:1]), axis=0)
        sym_index2 = np.concatenate((sym_index[1:], sym_index[:1]), axis=0)
    if len(y_pred2)==0:
        return class_dict
    return reclass_rank(y_pred2, sym_index2, quota_dict, class_dict)

def reclassify_rank(y_pred, sym_index):   #JEFF
    K = y_pred.shape[1]
    init_max = int(len(y_pred)/K) + 1
    quota_dict = {i:init_max for i in range(0, K)}
    class_dict = {i:[] for i in range(0,K)}
    return reclass_rank(y_pred, sym_index, quota_dict, class_dict)

def reclassify_prob(y_pred, sym_index):  #JEFF
    K = y_pred.shape[1]
    sequence_index = np.array(range(len(sym_index)))
    y_pred_ = y_pred.copy()
    rank = np.argsort(np.argsort(-y_pred_, axis=0), axis=0) + 1
    y_pred_is_max = np.argmax(y_pred_, axis=1)
    d = defaultdict(list)
    for k, v in zip(y_pred_is_max, sequence_index):
        d[k].append(v)
    class_dict = dict(d)
    keys = sorted(class_dict.keys())
    for i in range(K):
        if i not in keys:
            class_dict[i] = []
    iter_num = 0
    while True:
        iter_num += 1
        for i in range(int(keys[-1]/2)):
            if len(class_dict[i]) > int(len(y_pred_)/len(keys))+1:
                rank_pick = [rank[j,i] for j in class_dict[i]]
                tuple_arr = list(zip(rank_pick, class_dict[i]))
                sorted_arr = sorted(tuple_arr, key=lambda x: x[0], reverse=True)
                top_categories = [x[1] for x in sorted_arr[:int(len(y_pred_)/len(keys))+1]]
                class_dict[i+1].extend(list(set(class_dict[i])-set(top_categories)))
                class_dict[i] = top_categories
            if len(class_dict[i]) < int(len(y_pred_)/len(keys)):
                rank_pick = [rank[j,i] for j in class_dict[i+1]]
                tuple_arr = list(zip(rank_pick, class_dict[i+1]))
                sorted_arr = sorted(tuple_arr, key=lambda x: x[0], reverse=True)
                top_categories = [x[1] for x in sorted_arr[:min((int(len(y_pred_)/len(keys))-len(class_dict[i])), len(class_dict[i+1]))]]
                class_dict[i].extend(top_categories)
                class_dict[i+1] = list(set(class_dict[i+1])-set(top_categories))
        for i in range(keys[-1], int(keys[-1]/2), -1):
            if len(class_dict[i]) > int(len(y_pred_)/len(keys))+1:
                rank_pick = [rank[j,i] for j in class_dict[i]]
                tuple_arr = list(zip(rank_pick, class_dict[i]))
                sorted_arr = sorted(tuple_arr, key=lambda x: x[0], reverse=True)
                top_categories = [x[1] for x in sorted_arr[:int(len(y_pred_)/len(keys))+1]]
                class_dict[i-1].extend(list(set(class_dict[i])-set(top_categories)))
                class_dict[i] = top_categories
            if len(class_dict[i]) < int(len(y_pred_)/len(keys)):
                rank_pick = [rank[j,i] for j in class_dict[i-1]]
                tuple_arr = list(zip(rank_pick, class_dict[i-1]))
                sorted_arr = sorted(tuple_arr, key=lambda x: x[0], reverse=True)
                top_categories = [x[1] for x in sorted_arr[:min((int(len(y_pred_)/len(keys))-len(class_dict[i])), len(class_dict[i-1]))]]
                class_dict[i].extend(top_categories)
                class_dict[i-1] = list(set(class_dict[i-1])-set(top_categories))
        break_signal = False
        for v in class_dict.values():
            if len(v) not in [int(len(y_pred_)/len(keys))+1, int(len(y_pred_)/len(keys)), int(len(y_pred_)/len(keys))-1]:
                break_signal = True
                break
        if (not break_signal) or (iter_num>=20):
            output_dict = {k:[sym_index[i] for i in v] for k,v in class_dict.items()}
            break
    return output_dict

def reclassify_prdf(y_pred, sym_index, h): #JEFF
    K = y_pred.shape[1]
    arr_result = np.zeros(y_pred.shape[0])
    for i in range(h):
        arr1 = y_pred[:,i]
        arr2 = y_pred[:,K-i-1]
        arr_result += (arr2 - arr1) * (K/2+-i)
    output_dict = {sym_index[i]:arr_result[i] for i in range(len(sym_index))}
#     q = np.quantile(arr_result, np.arange(0.2,1,0.2))
#     groups = np.digitize(arr_result, q)
#     class_dict = {}
#     for i in range(K):
#         class_dict[i] = np.where(groups == i)[0].tolist()
#     output_dict = {k:[sym_index[i] for i in v] for k,v in class_dict.items()}
#     return output_dict
    return output_dict

def reclassify_return(y_pred, sym_index, past_mean_return): #JEFF
    K = y_pred.shape[1]
    arr_result = np.dot(y_pred, past_mean_return)
    output_dict = {sym_index[i]:arr_result[i][0] for i in range(len(sym_index))}
#     q = np.quantile(arr_result, np.arange(0.2,1,0.2))
#     groups = np.digitize(arr_result, q)
#     class_dict = {}
#     for i in range(K):
#         class_dict[i] = np.where(groups == i)[0].tolist()
#     output_dict = {k:[sym_index[i] for i in v] for k,v in class_dict.items()}
#    return output_dict
    return output_dict

def reclassify_sharpe(y_pred, sym_index, past_mean_return, past_sharpe): #JEFF
    K = y_pred.shape[1]
#     mu_i = np.dot(y_pred, past_mean_return)
#     sigma_mu_squared = np.square(past_mean_return) + np.square(past_mean_return/past_sharpe)
#     sigma_pred = np.dot(y_pred, sigma_mu_squared) - np.square(mu_i)
#     arr_result = mu_i/sigma_pred
#     output_dict = {sym_index[i]:arr_result[i][0] for i in range(len(sym_index))}

    arr_result = np.dot(y_pred, past_sharpe)
    output_dict = {sym_index[i]:arr_result[i][0] for i in range(len(sym_index))}
#     q = np.quantile(arr_result, np.arange(0.2,1,0.2))
#     groups = np.digitize(arr_result, q)
#     class_dict = {}
#     for i in range(K):
#         class_dict[i] = np.where(groups == i)[0].tolist()
#     output_dict = {k:[sym_index[i] for i in v] for k,v in class_dict.items()}
#    return output_dict
    return output_dict