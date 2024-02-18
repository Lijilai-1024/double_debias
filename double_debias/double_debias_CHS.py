import numpy as np
import copy
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y


class DoubleDebias:
    """
    用于实现双重去偏学习算法的估计器类。

    ---

    估计的模型是 y ~ theta D + g(z) + e
    初始化对象后，DoubleDebias.est_theta() 估计 theta。
    """

    def __init__(self, y=None, D=None, z=None, D_method=None, y_method=None, n_folds=2):
        """
        DoubleDebias的基础信息的构造函数

        参数:
        ---
            y : 1-D array
                依赖数据
            D : 1-D or 2-D array
                处理数据
            z : 2-D array
                混淆因素
            D_method : sklearn model
                slearn模型，如sklearn.ensemble.GradientBoostingRegressor。可以是实现了sklearn
                API的任何类
                1) 作为D_method()初始化
                2) D_method.fit()
                3) D_method.predict()
            y_method : sklearn model
                slearn模型，如sklearn.ensemble.GradientBoostingRegressor。可以是实现了sklearn
                API的任何类
                1) 作为y_method()初始化
                2) y_method.fit()
                3) y_method.predict()
            n_folds : Int
                用于估计的折叠数。至少需要2
        """
        check_X_y(z, y)
        check_X_y(D, y)
        self.y = y
        # 如果D是nX1，则将其展平为1-D数组
        if (D.ndim == 2) and (D.shape[1] == 1):
            self.D = np.ravel(D)
        else:
            self.D = D
        self.z = z
        self.nobs = y.shape[0]
        self.KFolds = KFold(n_splits=n_folds)
        self.methods = {'y': y_method, 'D': D_method}
        self.models = {'y': [copy.deepcopy(self.methods['y']) for i in range(self.KFolds.n_splits)],
                       'D': [copy.deepcopy(self.methods['D']) for i in range(self.KFolds.n_splits)]}

    @staticmethod
    def selector_check_(selector):
        """" 验证选择器是否是有效选项，如果不是则引发AttributeError。"""
        if selector not in ['y', 'D']:
            print(f"selector = {selector}. 选择器必须是 'y' 或 'D'")
            raise AttributeError

    @staticmethod
    def theta_formula(ytilde, V, D):
        """"Nyman正交估计器用于theta"""
        a = np.inner(V.transpose(), D.transpose())
        b = np.inner(V.transpose(), ytilde.transpose())
        if isinstance(a, np.float64):
            return b/a
        else:
            return np.linalg.solve(a, b)

    def KFolds_split_(self, selector):
        """" 返回y或D的KFolds对象的索引"""
        self.selector_check_(selector)
        return self.KFolds.split(self.z, getattr(self, selector))

    def est_models_(self, selector):
        """ 估计由选择器指定的y_models或D_models"""
        self.selector_check_(selector)
        for idx, (train, test) in enumerate(self.KFolds_split_(selector)):
            self.models[selector][idx].fit(self.z[train], getattr(self, selector)[train])

    def predict_(self, selector):
        """ 返回每个折叠的y或D数据的预测值的生成器，由选择器指定"""
        self.selector_check_(selector)
        return (self.models[selector][idx].predict(self.z[test]) for idx, (train, test) in enumerate(self.KFolds_split_(selector)))

    def residualize_(self, selector):
        """ 返回每个折叠的y或D数据的残差值的生成器，由选择器指定"""
        return (getattr(self, selector)[test] - self.models[selector][idx].predict(self.z[test]) for idx, (train, test) in enumerate(self.KFolds_split_(selector)))

    def est_thetas(self):
        """ 估计每个数据折叠的theta，存储为self.theta并返回数组 """
        self.thetas = np.array([self.theta_formula(Y, V, self.D[i[1]]) for (Y, V, i) in zip(
            self.residualize_('y'), self.residualize_('D'), self.KFolds_split_('D'))])
        return self.thetas

    def est_theta(self):
        """
        运行theta的完整估计循环

        ---
        估计y和D数据的每个折叠的模型，然后估计每个数据折叠的theta。
        返回thetas的平均值。
        """
        self.est_models_('y')
        self.est_models_('D')
        self.est_thetas()
        return np.mean(self.thetas, axis=0)