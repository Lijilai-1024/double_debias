from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from double_debias import DoubleDebias
import numpy as np
y=np.array([i for i in range(0,100)])
D=np.array([i//2 for i in range(0,100)]).reshape(-1,1)
z=np.array([[i**2 for i in range(0,100)], [i**3 for i in range(0,100)]]).transpose()

y_methods = [
    GradientBoostingRegressor(n_estimators=1000), 
    RandomForestRegressor(n_estimators=1000), 
    MLPRegressor(hidden_layer_sizes=(1000, ), max_iter=10000), 
    Lasso(alpha=0.1, max_iter=1000), 
    DecisionTreeRegressor()
]
n_folds = [2,5,20]
for n_fold in n_folds:
    print('n_fold:', n_fold)
    for y_method in y_methods:
        dd = DoubleDebias( y=y,
                    D=D,
                    z=z,
                    y_method= y_method,
                    D_method= LinearRegression(),
                    n_folds=n_fold)
        print(type(y_method))
        print(dd.est_theta())
        print('-------------------')
    