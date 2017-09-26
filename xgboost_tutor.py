import pandas as pd
import xgboost as xgb
import sklearn
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
实验数据使用kaggle Titanic的数据
'''

train_df = pd.read_csv('./data/train.csv', header=0)
test_df = pd.read_csv('./data/test.csv', header=0)


class DataFrameImputer(TransformerMixin):
    '''
    该类的作用是处理数据中nan的部分
    分为数据部分和非数据部分，数值部分如果是nan的使用median值进行替代
    非数值部分的使用最多出现的值作为替代值
    '''

    def fit(self, X, y=None):
        '''
        该函数中的X是一个pandas.DataFrame变量
        for c in X: 表示的是对X中的每一个列进行操作
        X[c].value_counts() 将c列的信息按计数降序排序
        .index[0] 取最多的那个值， index[1] 表示最多的值的计数
        X[c].dtype == np.dtype('O') 表示如果该列的数据类型是object也就是非数值型数据
        '''
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        '''
        pandas.DataFrame中的函数fillna是处理数据中nan的一个函数
        该函数中的参数value=self.fill是一个series
        '''
        return X.fillna(self.fill)


feature_columns_to_use = ['Pclass', 'Sex', 'Age', 'Fare', 'Parch']
nonnumeric_columns = ['Sex']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(
    test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# 接下来需要做的就是将非数值型的数据转换成为一个数值型的数据
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# 将pandas.DataFrame转换成为numpy.ndarray
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

'''
如果想在训练数据中再分出一部分作为验证集使用，可以使用train_test_split
train_X, validate_X, train_y, validate_y = train_test_split(train_X, train_y, test_size=0.33, random_state=42)
这里的random_state是一个整数，如果random_state是一个整数，则每次使用这个函数得到的结果是一样的
如果random_state没有设置一个值，则每次得到的结果则不一样
'''
'''
网格搜索与K折交叉验证以自动获取较好的学习率
需要知道如何找到比较重要的特征
'''

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from
# https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300,
                        learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                           'Survived': predictions})
submission.to_csv("submission.csv", index=False)
