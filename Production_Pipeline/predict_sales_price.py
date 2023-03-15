#%%
import pickle
import pandas as pd
from prod_model_func import *
from prod_input import *
#%%
# production model
model_name = 'Sale_Price_Predictor_Prod.pkl'
# new data path
filename = ''
model_name = 'model_search.pkl'
with open(model_name, 'rb') as f:
    sale_price_predictor = pickle.load(f)
#%%
# get data
data = get_data(filename,drop_cols)
# remove nulls, and discrepant rows
df = drop_bad_data(data,dict_drop_vals)
# 'shift' the year columns s.t. the first year in the data is '1'
df = add_val(df,add_scalar_cols)
# convert certain string columns to appropriate form and data type (removing extraneous info from the values)
df = str_op(df,dict_str_trans)
# makes predictions and save to csv
predictions = pd.Series(sale_price_predictor.predict(df))
predictions.to_csv(filename+'_preds.csv')