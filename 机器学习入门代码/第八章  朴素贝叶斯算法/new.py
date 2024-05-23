# 第八章 朴素贝叶斯算法
# 8.2.2 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import cohen_kappa_score,  confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from mlxtend.plotting import plot_decision_regions

# 8.3 高斯朴素贝叶斯算法示例
# 8.3.1 数据读取及观察
data = pd.read_csv('数据8.1.csv')
data.info()
data.isnull().values.any()
data['V1'].value_counts()
data['V1'].value_counts(normalize=True)

# 8.3.2 将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1'], axis=1)
y = data['V1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

# 8.3.3 高斯朴素贝叶斯算法拟合
model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 8.3.4 绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('高斯朴素贝叶斯ROC曲线')

# 8.3.5 运用两个特征变量绘制高斯朴素贝叶斯决策边界图
X2 = X.iloc[:, 0:2]
model = GaussianNB()
model.fit(X2, y)
model.score(X2, y)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('存款规模')
plt.ylabel('EVA')
plt.title('高斯朴素贝叶斯决策边界')

# 8.4 多项式、补集、二项式朴素贝叶斯算法示例
# 8.4.1 数据读取及观察
data = pd.read_csv('数据8.2.csv')
data['V1'].value_counts()
data['V1'].value_counts(normalize=True)

# 8.4.2 将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1'], axis=1)
y = data['V1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

# 8.4.3 多项式、补集、二项式朴素贝叶斯算法拟合
# 1、多项朴素贝叶斯方法
model = MultinomialNB(alpha=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = MultinomialNB(alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 2、补集朴素贝叶斯方法
model = ComplementNB(alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 3、二项朴素贝叶斯方法
model = BernoulliNB(alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = BernoulliNB(binarize=2, alpha=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 8.4.4 寻求二项式朴素贝叶斯算法拟合的最优参数
# 1、通过将样本分割为训练样本、验证样本、测试样本的方式寻找最优参数
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=100)
y_train.shape, y_val.shape, y_test.shape

best_val_score = 0
for binarize in np.arange(0, 5.5, 0.5):
    for alpha in np.arange(0, 1.1, 0.1):
        model = BernoulliNB(binarize=binarize, alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if score > best_val_score:
            best_val_score = score
            best_val_parameters = {'binarize': binarize, 'alpha': alpha}

best_val_score
best_val_parameters
model = BernoulliNB(**best_val_parameters)
model.fit(X_trainval, y_trainval)
model.score(X_test, y_test)

# 2、采用10折交叉验证方法寻找最优参数
param_grid = {'binarize': np.arange(0, 5.5, 0.5), 'alpha': np.arange(0, 1.1, 0.1)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(BernoulliNB(), param_grid, cv=kfold)
model.fit(X_trainval, y_trainval)
model.score(X_test, y_test)
model.best_params_
model.best_score_

outputs = pd
