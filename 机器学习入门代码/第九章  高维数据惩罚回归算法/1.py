# 第九章 高维数据惩罚回归算法
# 9.2.1 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

# 9.2.2 数据读取及观察
data = pd.read_csv('正文源代码及数据文件/第九章 高维数据惩罚回归算法/数据9.1.csv')
data.info()
data.isnull().values.any()
data['V1'].value_counts()

# 9.3 变量设置及数据处理
y = data['V1']
X_pre = data.drop(['V1'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X_pre)
np.set_printoptions(suppress=True)
np.mean(X, axis=0)
np.std(X, axis=0)

# 9.4 岭回归算法
# 9.4.1 使用默认惩罚系数构建岭回归模型
model = Ridge()
model.fit(X, y)
model.score(X, y)
model.intercept_
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
y_hat = model.predict(X)
pd.DataFrame({'Actual': y, 'Predicted': y_hat})

# 9.4.2 使用留一交叉验证法寻求最优惩罚系数构建岭回归模型
alphas = np.logspace(-4, 4, 100)
model = RidgeCV(alphas=alphas)
model.fit(X, y)
model.alpha_
model.score(X, y)

# 9.4.3 使用K折交叉验证法寻求最优惩罚系数构建岭回归模型
alphas = np.linspace(10, 100, 1000)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = RidgeCV(alphas=alphas, store_cv_values=True, cv=kfold)
model.fit(X, y)
model.alpha_
model.score(X, y)
model.intercept_
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
y_hat = model.predict(X)
pd.DataFrame({'Actual': y, 'Predicted': y_hat})

# 9.4.4 划分训练样本和测试样本下的最优岭回归模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = RidgeCV(alphas=np.linspace(10, 100, 1000))
model.fit(X_train, y_train)
model.alpha_
model.score(X_test, y_test)

# 9.5 Lasso回归算法
# 9.5.1 使用随机选取惩罚系数构建岭回归模型
model = Lasso(alpha=0.2)
model.fit(X, y)
model.score(X, y)
model.intercept_
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
y_hat = model.predict(X)
pd.DataFrame({'Actual': y, 'Predicted': y_hat})

# 9.5.2 使用留一交叉验证法寻求最优惩罚系数构建Lasso回归模型
alphas = np.linspace(0, 0.3, 100)
model = LassoCV(alphas=alphas)
model.fit(X, y)
model.alpha_
model.score(X, y)

# 9.5.3 使用K折交叉验证法寻求最优惩罚系数构建Lasso回归模型
alphas = np.linspace(0, 0.3, 100)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = LassoCV(alphas=alphas, cv=kfold)
model.fit(X, y)
model.alpha_
model.score(X, y)
model.intercept_
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])

# 9.5.4 划分训练样本和测试样本下的最优Lasso回归模型
model = LassoCV(alphas=np.linspace(0, 0.3, 100))
model.fit(X_train, y_train)
model.alpha_
model.score(X_test, y_test)

# 9.6 弹性网回归算法
# 9.6.1 使用随机选取惩罚系数构建弹性网回归模型
model = ElasticNet(alpha=1, l1_ratio=0.1)
model.fit(X, y)
model.score(X, y)
model.intercept_
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
y_hat = model.predict(X)
pd.DataFrame({'Actual': y, 'Predicted': y_hat})

# 9.6.2 使用K折交叉验证法寻求最优惩罚系数构建弹性网回归模型
alphas = np.logspace(-3, 0, 100)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = ElasticNetCV(cv=kfold, alphas=alphas, l1_ratio=[0.0001, 0.001, 0.01, 0.1, 0.5, 1])
model.fit(X, y)
model.alpha_
model.l1_ratio_
model.score(X, y)
model.intercept_
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])

# 9.6.3 划分训练样本和测试样本下的最优弹性网回归模型
model = ElasticNetCV(cv=kfold, alphas=np.logspace(-3, 0, 100), l1_ratio=[0.0001, 0.001, 0.01, 0.1, 0.5, 1])
model.fit(X_train, y_train)
model.alpha_
model.l1_ratio_
model.score(X_test, y_test)
