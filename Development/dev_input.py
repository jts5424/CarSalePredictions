from sklearn.linear_model import Ridge,Lasso,ElasticNet,ARDRegression,TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor

filename = 'Used car sale price estimation.csv'
drop_cols = ['Unnamed: 0','new_price']
dict_drop_vals = {'power':'null','year':1804}
dict_map_trans = {'owners':{'First':1,'Second':2,'Third':3,'Fourth & Above':4}}
dict_add_scalar = {'year':-1997}
dict_str_trans={'mpg':[' ',0,float],'engine_size':[' ',0,int],'power':[' ',0,float],'name':[' ',0,str]}
num_cols = ['year','millage','owners','mpg','engine_size','power','wear_factor']
cat_cols = ['name','location','fuel_type','gears','seats']
max_zscore = 3
target = 'sale_price'
score_met= ['explained_variance','r2','neg_mean_absolute_error']
#models = {'ridge':Ridge(),'lasso':Lasso(),'elastic':ElasticNet(),'ard':ARDRegression(),'theil':TheilSenRegressor(),'svr_lin':SVR(kernel='linear'),'svr_rbf':SVR(kernel='rbf'),'dtr':DecisionTreeRegressor(),'rfr':RandomForestRegressor(),'ada':AdaBoostRegressor(),'gbr':GradientBoostingRegressor()}
models = {'rfr':RandomForestRegressor()}
boxplot_scores = ['test_explained_variance', 'test_r2', 'test_neg_mean_absolute_error','train_explained_variance','train_r2', 'train_neg_mean_absolute_error']

