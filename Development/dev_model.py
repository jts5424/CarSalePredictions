#%%
from dev_model_func import *
from dev_input import *
# import data from csv file into pandas dataframe
df_origin = pd.read_csv(filename).set_index('index').drop(columns=drop_cols)
df_fixed = get_clean_transform_data(df_origin,dict_drop_vals,dict_map_trans,dict_add_scalar,dict_str_trans)
#df_fixed_filt= drop_outliers(df_origin,df_fixed,num_cols,max_zscore)
df_fixed_filt = df_fixed
df_model = encode_cats(df_fixed_filt,cat_cols)
model_scores_agg = model_scoring_eval(models,df_model,target,score_met)
scores_reorg = reorg_scores(model_scores_agg)
title = 'Model Score Comparison No Outlier Filter'
output_filename = 'model_res_boxplot_compare_no_outlier_filt'
model_boxplot_compare(boxplot_scores,scores_reorg,title,output_filename)
# %%
