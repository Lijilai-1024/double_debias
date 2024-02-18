import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# 定义数据
y = np.array([i for i in range(0, 10)])
D = np.array([i // 2 for i in range(0, 10)])
z = np.array([[i ** 2, i ** 3] for i in range(0, 10)])

# 初始化 KFold 对象
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 存储处理效应的估计值
effects = []

# 在每一折中进行双偏差法估计
for train_index, test_index in kf.split(z):
    z_train, z_test = z[train_index], z[test_index]
    D_train, D_test = D[train_index], D[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 第一阶段回归：预测处理变量 D
    model_D = LinearRegression().fit(z_train, D_train)
    D_train_hat = model_D.predict(z_train)
    D_test_hat = model_D.predict(z_test)

    # 第二阶段回归：估计处理效应 (使用GradientBoostingRegressor)
    model_Y = GradientBoostingRegressor(n_estimators=1000).fit(D_train_hat.reshape(-1, 1), y_train)
    effect = model_Y.feature_importances_[0]  # 注意：这里有可能需要调整，具体取决于你的模型设定
    effects.append(effect)

print("处理效应估计值:", np.mean(effects))