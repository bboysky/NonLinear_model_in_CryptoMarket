import sys 
sys.path.append("..") 
from utils import *

def get_factor_param(factor,fac_file):

    path = 'Factors/'+fac_file+'/'+factor+'/'+'factor_param.json'
    with open(path,'r') as r:
        para = json.load(r)

    return para

def get_factorAlgo_class(factor,fac_file):

    module = 'Factors.'+fac_file+'.'+factor+'.calculate_factor'

    r = importlib.import_module(module)
    importlib.reload(r)
    instance = getattr(r, "MyFactor")
    return instance("test")

def cal_f(fact,fac_file,dt_stt,dt_end, symbol_list):
    print('calculate factor value : ',fac_file,fact)
    para = get_factor_param(fact,fac_file)

    factor_class = get_factorAlgo_class(fact,fac_file)
    factor_dict = calculate_factor_value(factor_class,para,dt_stt,dt_end, symbol_list)

    print('Finish calculate factor value',fac_file,fact)
    for p in factor_dict:
        name = fact +'_'+ p
        for sym in factor_dict[p].columns:
        
            path = '../Factor_Results/'+sym

            if not os.path.exists(path):
                os.makedirs(path)
            f = factor_dict[p][sym]
            f.columns = [name]

            f.to_hdf(path+'/'+name+'.h5', key='factor')

if __name__ == '__main__':
    fac_file = 'normal_factor'
    fact_ = os.listdir('Factors/'+fac_file)
    
    with open('../info.json','r') as r:
        symbol_list = json.load(r)['symbol_list']
        
    dt_stt,dt_end = '20220101','20230708'

    print(datetime.now())
    pool = Pool(processes=40)    
    for fact in fact_:
        # cal_f(fact,fac_file,dt_stt,dt_end, symbol_list)
        pool.apply_async(cal_f, args=(fact,fac_file,dt_stt,dt_end, symbol_list))
    pool.close()
    pool.join()
    print(datetime.now())
    print('Finish save ',fact_,' factor ')

