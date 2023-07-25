import sys 
sys.path.append("..") 
from utils import *

Index_sym = ['binance.btc_usdt_swap','binance.eth_usdt_swap']

future_col = []
close = pd.read_hdf('../data/close_1min_swap.h5').resample('5min').last().loc['20220101':,]
volume = pd.read_hdf('../data/volume_1min_swap.h5').resample('5min').sum().loc['20220101':,]

#  index amount feature
print('计算指数成交量特征')
df_index = close[Index_sym]*volume[Index_sym]
df_index['index_amount'] = df_index.mean(axis=1)
for tmp_col in ['index_amount']:
    for i in range(6):
        new_col_name = tmp_col + '_shift_{}'.format(i+1)
        df_index[new_col_name] = df_index['index_amount'].shift(12*(i+1))
    for i in range(6):
        new_col_name = tmp_col + '_shift_{}'.format(i+1)
        df_index[new_col_name] = (df_index[tmp_col] - df_index[new_col_name]) / df_index[new_col_name]
        future_col.append(new_col_name)

df_index = df_index.reset_index()

date_map = dict(zip(close.index, range(len(close.index))))
df_index['trade_date_id'] = df_index['datetime'].map(date_map)
# future_col.append('datetime')
future_col.append('trade_date_id')
df_index = df_index.iloc[:-2,:]
df_index[future_col].to_pickle('../data/Index_feature_144.pkl')

print('FINISH SAVE IDNEX FEATURE')





