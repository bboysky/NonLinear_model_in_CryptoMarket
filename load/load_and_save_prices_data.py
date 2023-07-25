import sys 
sys.path.append("..") 
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default = None)
    args = parser.parse_args()
    if args.type=='spot':
        paras = 'spot'
    else:
        paras = 'swap'

    json_path = '../info.json'

    with open(json_path,'r') as r:
        symbol_list = json.load(r)['symbol_list']
        
    if paras == 'spot':
        for i in range(len(symbol_list)):
            symbol_list[i] = symbol_list[i].replace('_swap','')



    def load_data(symbol_list,type_):
        print('loading ',type_)
        key_= {'close':'prices','open':'prices','high':'prices','low':'prices','volume':'volume'}
        data_list = os.listdir('../data')
        file_name = '../data/'+type_+'_1min'+ '_'+paras+'.h5'
        end = datetime.strftime(datetime.now() + timedelta(days=1),'%Y-%m-%d').replace('-','')
        if type_+'_1min_swap.h5' in data_list:
            data = pd.read_hdf(file_name)
            dt_str, dt_end = datetime.strftime(data.index[-1],'%Y-%m-%d').replace('-',''),end
            print('增量补充数据 ，日期:%s--%s '%(dt_str,dt_end))
            close_df = get_close(symbol_list, dt_str, dt_end,type_)
            data = pd.concat([data,close_df],axis=0).drop_duplicates()
            data.to_hdf(file_name, key=key_[type_])
        else:
            dt_str, dt_end = '20220101',end    
            print('全量补充数据 ，日期:%s--%s '%(dt_str,dt_end))
            close_df = get_close(symbol_list, dt_str, dt_end,type_)
            close_df.to_hdf(file_name, key=key_[type_])

    for type_ in ['close','open','high','low','volume']:
        load_data(symbol_list,type_)