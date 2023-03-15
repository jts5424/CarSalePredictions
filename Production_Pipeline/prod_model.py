#%%
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from prod_input import *
from prod_model_func import *

# get data
data = get_data(filename,drop_cols)
# remove nulls, and discrepant rows
df = drop_bad_data(data,dict_drop_vals)
# 'shift' the year columns s.t. the first year in the data is '1'
# if using oop we could output the shift value to the instantiated object for testing data
df = add_val(df,add_scalar_cols)
# convert certain string columns to appropriate form and data type (removing extraneous info from the values)
df = str_op(df,dict_str_trans)

# split data for training/validation and testing
X = df[df.columns[df.columns!=target]]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# we are not removing outliers because they may be actual datapoints but this is an option
#drop_outliers_trans = FunctionTransformer(drop_outliers,kw_args={'args':args})
# convert categorical data appropriately
ohe = OneHotEncoder(handle_unknown='ignore')
# treating the owners column as numeric from the exploratory data analysis that we did
oe = OrdinalEncoder(categories=[own_vals])

# build ML pipeline
cat_enc_trans = make_column_transformer((oe, oe_cols),(ohe, ohe_cols),remainder='passthrough')
step_1 = ('cat_enc',cat_enc_trans)
# this may not be necessary because we are using a random forest model. But for other models it is important for all features to be scaled equally
step_2 = ('scalar',MaxAbsScaler())
# random forest appeared to perform the best in we compared all of the models in the boxplots
step_3 = ('est',RandomForestRegressor())
pipe = Pipeline(steps=[step_1,step_2,step_3])
# use search with cross validation for hyperparameter tuning and validation
rand_pipe = RandomizedSearchCV(estimator=pipe,
                        param_distributions=params,scoring=score_met,
                        n_iter=100,cv=10,verbose=2,random_state=42,
                        n_jobs=-1,return_train_score=True,refit='mse')
# fit the model on the training data
rand_pipe.fit(X_train,y_train)
# calculate, output and store results of the search, test/train, predictions, and production model
best_est_test_score,cv_results,cv_res_trunc = calculate_save_res(rand_pipe,X_test,X_train,y_test,y_train)
# get feature importances from the best model
rand_pipe.best_params_
clf=RandomForestRegressor(bootstrap= True,
 max_depth= 29,
 max_features= 0.1844347898661638,
 min_samples_leaf= 0.07806910002727693,
 min_samples_split= 0.11977318143135092,
 n_estimators= 125)
X = pd.get_dummies(X, prefix=ohe_cols, columns=ohe_cols)
X['owners'] = X['owners'].replace({'First':1,'Second':2,'Third':3,'Fourth & Above':4})
clf.fit(X,y)
feat_imps = pd.Series(index=clf.feature_names_in_,data=clf.feature_importances_)
feat_imps.sort_values(ascending=False,inplace=True)
feat_imps.to_csv('best_model_feature_importances.csv')
