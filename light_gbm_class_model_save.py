from utils import *
from sklearn.metrics import roc_auc_score

# dataset= ['Label_long_0201_Index_144','Label_long_0415_Index_144','Label_short_0301_Index_144','Label_short_0201_Index_144','Label_short_0415_Index_144']#'Dataset_all
dataset= ['long_Label_144_p_0_03_l_0_01_end_cap_withivsonly','short_Label_144_p_0_03_l_0_01_end_cap_withivsonly']

for data_set in dataset:
    # data_set = 'addnew_imp_'+data_set
    for label_type in ['label_final']:
        for _num_boost_round_ in [5000,10000]:
            params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
   
                    'num_leaves': 31,
                    'learning_rate': 0.06,
                    'verbosity':-1,
                    'min_data_in_leaf':20,
                    'feature_fraction':1,
                    'bagging_fraction':1,
                    'lambda_l1':0,
                    'lambda_l2':0,
                    'n_estimators':_num_boost_round_
                }

            end_dt_list = ['2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12','2023-01','2023-02','2023-03','2023-04','2023-05']

            new_factor={}
            model_eval = {}
            feature_imp = {}

            for str_dt in end_dt_list:
                start_ = time.time()
                print(str_dt)
                
                train_data = pd.read_pickle('data/train_test_data/'+data_set+'/'+'train_dateset_'+str_dt+'.pkl').dropna()

                train_data = train_data.drop('level_1',axis=1)
        
                train_X = train_data.drop([label_type],axis=1)
                train_Y = train_data.loc[:,label_type]

                outsample = pd.read_pickle('data/train_test_data/'+data_set+'/'+'test_dateset_'+str_dt+'.pkl').dropna()

                outsample_X = outsample.drop(['level_1','label_final'],axis=1)

                X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)

                # 构建LightGBM模型
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

                gbm = lgb.train(params,
                                lgb_train,
                                num_boost_round=_num_boost_round_,
                                valid_sets = [lgb_train,lgb_eval],
                                early_stopping_rounds=100
                                )

                y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                auc = roc_auc_score(y_test, y_pred)
                precision=precision_score(y_test, (y_pred>0.5).astype(int))
                feature_imp[str_dt] = gbm.feature_importance().tolist()
                model_eval[str_dt] = {
                    'auc':auc,
                    'precision':precision,
                    'best_iteration':gbm.best_iteration,
                    'time':(time.time() - start_)/60
                    }
                outsample['outsample_y_pred'] = gbm.predict(outsample_X, num_iteration=gbm.best_iteration)

                new_factor[str_dt] = outsample.pivot_table(index=outsample.index, columns='level_1', values='outsample_y_pred')

            factor_df = pd.DataFrame()
            for dt in new_factor.keys():
                factor_df = pd.concat([factor_df,new_factor[dt]],axis=0)

            path = 'signal_df_iv/LGBM_result/'+data_set+'_leave_'+str(gbm.params['num_leaves'])+'_'+str(gbm.params['learning_rate'])+'_min_leaf_'+str(gbm.params['min_data_in_leaf'])+'_ff_'+str(gbm.params['feature_fraction'])+'_bf_'+str(gbm.params['bagging_fraction'])+'_l1_'+str(gbm.params['lambda_l1'])+'_l2_'+str(gbm.params['lambda_l2'])+'_r_'+str(_num_boost_round_)+'_'+label_type

            model_json = {
                'eval':model_eval,
                'feature_imp':feature_imp
            }
            with open(path+'.json','w') as f:
                json.dump(model_json,f)
            factor_df.to_hdf(path+'.h5',key = 'signal')
