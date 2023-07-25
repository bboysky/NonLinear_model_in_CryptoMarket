import sys 
sys.path.append("..") 
from utils import *

featrue = {}
N = 144
featrue['close'] = pd.read_hdf('data/close_1min_swap.h5').resample('5min').last().loc['20220101':,:]

featrue['high'] = pd.read_hdf('data/high_1min_swap.h5').resample('5min').max().loc['20220101':,:]
featrue['low'] = pd.read_hdf('data/low_1min_swap.h5').resample('5min').min().loc['20220101':,:]
featrue['open'] = pd.read_hdf('data/open_1min_swap.h5').resample('5min').first().loc['20220101':,:]

ts_code_map = dict(zip(list(featrue['close'].columns), range(len(featrue['close'].columns))))
date_map = dict(zip(featrue['close'].index, range(len(featrue['close'].index))))

df = pd.concat(featrue, axis=1)  # 合并DataFrames
df_dataset = df.stack().reset_index()

df_dataset['ts_code_id'] = df_dataset['level_1'].map(ts_code_map)
df_dataset['trade_date_id'] = df_dataset['datetime'].map(date_map)

df_dataset['ts_date_id'] = (10000+df_dataset['ts_code_id']) * 1000000 + df_dataset['trade_date_id']

use_col = []
for i in range(N):
    print('begin shift %d times' % (i+1))
    tmp_df = df_dataset[['ts_date_id', 'high', 'low']]
    tmp_df = tmp_df.rename(columns={'high':'high_shift_{}'.format(i+1), 'low':'low_shift_{}'.format(i+1)})
    use_col.append('high_shift_{}'.format(i+1))
    use_col.append('low_shift_{}'.format(i+1))
    tmp_df['ts_date_id'] = tmp_df['ts_date_id'] - i - 1
    df_dataset = df_dataset.merge(tmp_df, how='left', on='ts_date_id')

df_dataset.to_pickle('../data/max_min.pkl')