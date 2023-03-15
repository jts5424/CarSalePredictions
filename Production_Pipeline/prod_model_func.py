import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
import pickle

def get_data(filename,drop_cols):
    df = pd.read_csv(filename).set_index('index').drop(columns=drop_cols)
    return df

def drop_bad_data(df,dict_drop_vals):
    df_filt = df.dropna(axis=0)
    for item in dict_drop_vals.items():
        print('data size: ' + str(len(df_filt)))
        var = item[0]
        val = item[1]
        df_filt = df_filt[~df_filt[var].astype(str).str.contains(str(val))]
    print('percent of data lost from dropping nulls and discrepant values: ' + str(round(100*(len(df) - len(df_filt))/len(df),2)))
    return df_filt

def add_val(df_filt,add_scalar_cols):
    for col in add_scalar_cols:
        val = 1-df_filt[col].min()
        df_filt[col] = df_filt[col].add(val)
    return df_filt


def str_op(df_filt,dict_str_trans):
    for item in dict_str_trans.items():
        var= item[0]
        val_1 = item[1][0]
        val_2 = item[1][1]
        val_3 = item[1][2]
        df_filt[var] = df_filt[var].str.split(val_1).str[val_2].astype(val_3)
    return df_filt

def drop_outliers(df_filt,args):
    df = args[0]
    num_cols = args[1]
    max_zscore= args[2]
    df_num = df_filt[num_cols]
    df_num_filt = df_num[(np.abs(stats.zscore(df_num)) < max_zscore).all(axis=1)]
    df_filt_out_filt = df_filt[df_filt.index.isin(df_num_filt.index)]
    print('Total percent of data lost from dropping nulls, discrepant, and outlier values: '+str(round(100*(len(df) - len(df_filt_out_filt))/len(df),2)))
    return df_filt_out_filt

def calculate_save_res(search,X_test,X_train,y_test,y_train):
    cv_results = pd.DataFrame.from_dict(search.cv_results_)
    cv_res_trunc = cv_results[cv_results.columns[~cv_results.columns.str.contains('split')]]
    cv_res_trunc.to_csv('cv_results.csv')
    best_est = search.best_estimator_
    y_pred = best_est.predict(X_test)
    y_train_pred = best_est.predict(X_train)
    best_est_test_score = pd.DataFrame(index=['explained_variance','mae','mse','r2_score'],columns=['Test Data','Train Data'])
    best_est_test_score.loc['explained_variance','Test Data']=explained_variance_score(y_test,y_pred)
    best_est_test_score.loc['mae','Test Data']=mean_absolute_error(y_test,y_pred)
    best_est_test_score.loc['mse','Test Data']=np.sqrt(mean_squared_error(y_test,y_pred))
    best_est_test_score.loc['r2_score','Test Data']=r2_score(y_test,y_pred)
    best_est_test_score.loc['explained_variance','Train Data']=explained_variance_score(y_train,y_train_pred)
    best_est_test_score.loc['mae','Train Data']=mean_absolute_error(y_train,y_train_pred)
    best_est_test_score.loc['mse','Train Data']=np.sqrt(mean_squared_error(y_train,y_train_pred))
    best_est_test_score.loc['r2_score','Train Data']=r2_score(y_train,y_train_pred)
    best_est_test_score.to_csv('Random Forest Top Performer Tested Results.csv')
    
    with open('model_search.pkl','wb') as f:
        pickle.dump(search,f)

    prod_model = search.best_estimator_
    with open('Sale_Price_Predictor_Prod.pkl','wb') as f:
        pickle.dump(prod_model,f)
    return best_est_test_score,cv_results,cv_res_trunc

