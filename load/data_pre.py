import sys 
sys.path.append("..") 
from utils import *

close = pd.read_hdf('../data/close_1min_swap.h5').resample('5min').last().loc['20220101':'20230704',:]

with open('../info.json','r') as r:
    info = json.load(r)

def pre_feature(ft,symbols_all,freq = '5min'):
    feature_tmp = pd.DataFrame()
    for sym in symbols_all:
        
        f = pd.read_hdf('../Factor_Results/'+sym+'/'+ft).shift(1).resample(freq).last().shift(-1)
        feature_tmp = pd.concat([feature_tmp,f],axis=1)
    # feature_tmp = feature_tmp.sub(feature_tmp.mean(axis=1),axis=0).div(feature_tmp.std(axis=1),axis=0)
    return feature_tmp

result_dict = {}
f = info['features_selected']
f_l = []
for i in os.listdir('../Factor_Results/binance.1inch_usdt_swap'):
    if i in f:
        f_l.append(i)
    # else:
    #     print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

print(datetime.now())
pool = Pool(processes=50)
for ft in f_l:
    print(ft)
    result_dict[ft] = pool.apply_async(pre_feature, args=(ft,info['symbol_list']))
pool.close()
pool.join()
print(datetime.now())

featrue = {}
for i in f_l:
    featrue[i] = result_dict[i].get()

for i in featrue.keys():
    status_ = check_factor_inf_null(featrue[i],1,1)
    if status_:
        print(i)
        featrue[i] = featrue[i].replace([np.inf,-np.inf],0)

market_cap = pd.read_hdf('../data/market_cap.h5').shift(1)
featrue['market_cap_ratio'] = ((market_cap/market_cap.rolling(5).mean()).resample('5min').ffill()).loc[close.index,close.columns]
featrue['market_cap'] = (market_cap.resample('5min').ffill()).loc[close.index,close.columns]

df = pd.concat(featrue, axis=1)
df_dataset = df.stack().reset_index()

date_map = dict(zip(close.index, range(len(close.index))))
ts_code_map = dict(zip(list(close.columns), range(len(close.columns))))

df_dataset['ts_code_id'] = df_dataset['level_1'].map(ts_code_map)
df_dataset['trade_date_id'] = df_dataset['level_0'].map(date_map)
df_dataset['ts_date_id'] = (10000+df_dataset['ts_code_id']) * 1000000 + df_dataset['trade_date_id']

Index_feature = pd.read_pickle('../data/Index_feature_144.pkl')
iv = pd.read_pickle('../data/btc_iv.pkl').shift(1)
iv_filter = ((iv/iv.shift(12)).resample('5min').ffill().loc[close.index]).reset_index()
iv_filter['trade_date_id'] = iv_filter.index
Index_feature = Index_feature.merge(iv_filter[['trade_date_id','iv']], how='left', on='trade_date_id')

df_dataset = df_dataset.merge(Index_feature, how='left', on='trade_date_id')

for label_name in ['long_Label_144_p_0_03_l_0_01_end']:
    
    file_name = '../data/train_test_data/'+label_name+'_cap_withiv'
    label = pd.read_pickle('../Label/'+label_name+'.pkl')
    df_dataset_ = df_dataset.merge(label, how='left', on='ts_date_id')

    # df_dataset_ = df_dataset_.dropna().drop(['ts_date_id','trade_date_id','ts_code_id'],axis=1)
    df_dataset_ = df_dataset_.drop(['ts_date_id','trade_date_id','ts_code_id'],axis=1)
    df_dataset_ = df_dataset_.set_index('level_0')
    print(df_dataset_.columns)

    train_start_dt = ['2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12','2023-01','2023-02','2023-03','2023-04']
    train_end_dt = ['2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12','2023-01','2023-02','2023-03','2023-04','2023-05','2023-06']
    test_start_dt = ['2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12','2023-01','2023-02','2023-03','2023-04','2023-05','2023-06']
    test_end_dt = ['2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12','2023-01','2023-02','2023-03','2023-04','2023-05','2023-06']
    # train_start_dt = ['2023-01']
    # train_end_dt = ['2023-06']
    # test_start_dt = ['2023-06']
    # test_end_dt = ['2023-07']
    file_name = file_name+'sonly'
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    # for i in range(len(train_start_dt)):
    #     print(train_start_dt[i],train_end_dt[i])
    #     path = file_name+'/train_dateset_'+train_end_dt[i]+'.pkl'
    #     data = df_dataset_.loc[train_start_dt[i]:train_end_dt[i]]
    #     data.to_pickle(path)
        
    # for i in range(len(test_end_dt)):
        
    #     path = file_name+'/test_dateset_'+test_start_dt[i]+'.pkl'
    #     data = df_dataset_.loc[test_end_dt[i]]
    #     data.to_pickle(path)

    path = file_name+'/'+'pair_trading.pkl'
    df_dataset_.to_pickle(path)



