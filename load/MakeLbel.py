import sys 
sys.path.append("..") 
from utils import *

df_data = pd.read_pickle('../data/max_min.pkl')

def makelabel_oneside(df_dataset,N,type_,profit,loss):
    for i in range(N):
        # 先判断是否到止损
        if type_ == 'short':
            tmp_col = 'low_shift_' + str(i + 1)
            idx = df_dataset[tmp_col] <= -profit
            df_dataset.loc[idx, 'label_min'] = 1

            # 再判断是否到最小利润点
            tmp_col = 'high_shift_' + str(i + 1)
            idx = df_dataset[tmp_col] > loss
            df_dataset.loc[idx, 'label_max'] = 1

            # 如果不到止损点并且 到了最小利润点， 标签为1
            idx = (df_dataset['label_min'] == 1) & (df_dataset['label_max'] == 0) & (df_dataset['label_final'] == 0)
            df_dataset.loc[idx, 'label_final'] = 1

            # 如果到了止损点并且 到了最小利润点， 标签为1
            idx = (df_dataset['label_max'] == 1) & (df_dataset['label_final'] == 0)
            df_dataset.loc[idx, 'label_final'] = -1

        if type_ == 'long':
            tmp_col = 'low_shift_' + str(i + 1)
            idx = df_dataset[tmp_col] < -loss
            df_dataset.loc[idx, 'label_min'] = 1

            # 再判断是否到最小利润点
            tmp_col = 'high_shift_' + str(i + 1)
            idx = df_dataset[tmp_col] >= profit
            df_dataset.loc[idx, 'label_max'] = 1

            # 如果不到止损点并且 到了最小利润点， 标签为1
            idx = (df_dataset['label_min'] == 0) & (df_dataset['label_max'] == 1) & (df_dataset['label_final'] == 0)
            df_dataset.loc[idx, 'label_final'] = 1

            # 如果到了止损点并且 到了最小利润点， 标签为1
            idx = (df_dataset['label_min'] == 1) & (df_dataset['label_final'] == 0)
            df_dataset.loc[idx, 'label_final'] = -1     

    return df_dataset

for type_ in ['short','long']:
    print('创建:%s 标签'%type_)
    N,profit,loss = 144,0.02,0.01

    df_dataset = df_data.copy()

    for i in range(N):
        df_dataset['high_shift_{}'.format(i+1)] = (df_dataset['high_shift_{}'.format(i+1)] - df_dataset['close']) / df_dataset['close']
        df_dataset['low_shift_{}'.format(i+1)] = (df_dataset['low_shift_{}'.format(i+1)] - df_dataset['close']) / df_dataset['close']

    df_dataset['label_max'] = 0
    df_dataset['label_min'] = 0
    df_dataset['label_final'] = 0
    df_dataset = makelabel_oneside(df_dataset,N,type_,profit,loss)

    df_dataset['label_final'] = df_dataset['label_final'] == 1

    df_dataset['label_final'] = df_dataset['label_final'].apply(lambda x: int(x))

    print('the rate of label 1: %.4f' % (df_dataset['label_final'].sum() / len(df_dataset)))

    name_ = '../Label/'+type_+'_Label_'+str(N)+'_p_'+str(profit).replace('.','_')+'_l_'+str(loss).replace('.','_')+'_end'+'.pkl'
    
    df_dataset[['label_final','ts_date_id']].to_pickle(name_)
    print('保存Label:%s'%name_)


