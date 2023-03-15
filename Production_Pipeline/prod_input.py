from sklearn.linear_model import Ridge,Lasso,ElasticNet,ARDRegression,TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
import numpy as np
from scipy.stats import randint, uniform

# data file path
filename = 'Used car sale price estimation.csv'
# columms to be dropped as decided based on the EDA
drop_cols = ['Unnamed: 0','new_price']
# discrepant values found from the EDA
dict_drop_vals = {'power':'null','year':1804}
# adjust time domain
add_scalar_cols = ['year']
dict_str_trans={'mpg':[' ',0,float],'engine_size':[' ',0,int],'power':[' ',0,float],'name':[' ',0,str]}
# numeric columns as decided from the EDA
num_cols = ['year','millage','owners','mpg','engine_size','power','wear_factor']
# categorical columns as decided from the EDA
ohe_cols = ['name','location','fuel_type','gears','seats']
# columns values to map via ordinal encoding. if more than one column, use a list of list for the values 
own_vals =['First','Second','Third','Fourth & Above']
oe_cols = ['owners']
# outlier limit (not used in this pipeline)
max_zscore = 3
# target (column to be predicted)
target = 'sale_price'
# scoring metrics to output from our cross validation
score_met= {'exp_var':'explained_variance','r2':'r2','mae':'neg_mean_absolute_error','mse':'neg_mean_squared_error'}
# Number of trees in random forest randomly sample numbers from 4 to 1000 estimators
n_estimators = randint(4,1000)
# The maximum depth of the tree. randomly sampled
max_depth = randint(1,50)
# Number of features to consider at every split uniformly distributed max_features, bounded between 0.001 and 1
max_features = uniform(0.001, 1)
# Minimum number of samples allowed in a final leaf node # uniform distribution between 0.0001 and 1
min_samples_leaf = uniform(0.0001, .1)
# Minimum number of samples required to split a node # uniform distribution from 0.001 to 0.2
min_samples_split = uniform(0.001, 0.199)
# Method of selecting samples for training each tree
bootstrap = [True, False]



params={
'est__n_estimators':n_estimators,
'est__max_depth':max_depth,
'est__max_features':max_features,
'est__min_samples_split':min_samples_split,
'est__min_samples_leaf':min_samples_leaf,
'est__bootstrap':bootstrap
}
