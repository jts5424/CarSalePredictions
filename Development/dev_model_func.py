import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

def get_clean_transform_data(df_origin,dict_drop_vals,dict_map_trans,dict_add_scalar,dict_str_trans):
    df_fixed = df_origin.dropna(axis=0)
    for item in dict_drop_vals.items():
        print('data size: ' + str(len(df_fixed)))
        var = item[0]
        val = item[1]
        df_fixed = df_fixed[~df_fixed[var].astype(str).str.contains(str(val))]
    print('percent of data lost from dropping nulls and discrepant values: ' + str(round(100*(len(df_origin) - len(df_fixed))/len(df_origin),2)))
    for item in dict_map_trans.items():
        var = item[0]
        mapping = item[1]
        df_fixed = df_fixed.replace({var:mapping})
    for item in dict_add_scalar.items():
        var = item[0]
        val = item[1]
        df_fixed[var] = df_fixed[var].add(val)
    for item in dict_str_trans.items():
        var= item[0]
        val_1 = item[1][0]
        val_2 = item[1][1]
        val_3 = item[1][2]
        df_fixed[var] = df_fixed[var].str.split(val_1).str[val_2].astype(val_3)
    return df_fixed
    # total data lost from NaNs and outliers: ~8%

def drop_outliers(df_origin,df_fixed,num_cols,max_zscore):
    df_num = df_fixed[num_cols]
    df_num_filt = df_num[(np.abs(stats.zscore(df_num)) < max_zscore).all(axis=1)]
    df_fixed_filt = df_fixed[df_fixed.index.isin(df_num_filt.index)]
    print('Total percent of data lost from dropping nulls, discrepant, and outlier values: '+str(round(100*(len(df_origin) - len(df_fixed_filt))/len(df_origin),2)))
    return df_fixed_filt

def encode_cats(df_fixed_filt,cat_cols):
    df_model = pd.get_dummies(df_fixed_filt, prefix=cat_cols, columns=cat_cols)
    return df_model


def first_pass_alg_scores(df_model,target,regr,model_name,score_met):
    X = df_model[df_model.columns[df_model.columns!=target]]
    y = df_model[target]
    if model_name not in ['dtr','rfr','ada','gbr']:
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = X.copy()
    scores = cross_validate(regr, X_norm, y, cv=10,scoring=score_met,return_train_score=True)
    df_scores_agg = pd.DataFrame.from_dict(scores)
    return df_scores_agg

def model_scoring_eval(models,df_model,target,score_met):
    model_scores_agg = {}
    for m in models.items():
        model_name = m[0]
        regr = m[1]
        df_scores_agg = first_pass_alg_scores(df_model,target,regr,model_name,score_met)
        model_scores_agg[model_name] = df_scores_agg
        print(model_name)
    return model_scores_agg

def reorg_scores(model_scores_agg):
    scores_reorg = {}
    for c in list(model_scores_agg.items())[0][1].columns:
        df = pd.DataFrame()
        for item in model_scores_agg.items():
            model_name = item[0]
            df_scores = item[1]
            col_scores = df_scores[c]
            df[model_name] = col_scores
        scores_reorg[c] = df
    return scores_reorg

def model_boxplot_compare(boxplot_scores,scores_reorg,title,output_filename):
    n_rows = 2
    n_cols = int(np.ceil(len(boxplot_scores)/n_rows))
    fig,axes= plt.subplots(ncols=n_cols,nrows=n_rows,figsize=(30,20))
    for i,box in enumerate(boxplot_scores):
        df = scores_reorg[box]
        a = df.boxplot(ax=axes.flatten()[i],rot=45)
        a.set_title(box,fontsize=24)
        if 'absolute' in box:
            a.set_ylim([-10,0])
        else:
            a.set_ylim([0,1])
        a.set_xticklabels(a.get_xticklabels(),fontsize=20)
        #a.set_yticklabels(a.get_yticklabels(),fontsize=16)
    fig.suptitle(title,fontsize=28)
    fig.savefig(output_filename+'.png')
