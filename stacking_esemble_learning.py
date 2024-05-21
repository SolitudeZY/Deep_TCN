import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


import xgboost as xgb


class StackingClassifier():
    """
    This class is a template of stacking method for classification.
    It only provides fit and predict functions, and works with binary [0, 1] labels.
    predict function returns the probability of label 1.
    To learn how to use, see test/test_stackingclassifier.py

    """

    def __init__(self, base_classifiers, combiner):
        self.base_classifiers = base_classifiers
        self.combiner = combiner

    def fit(self, X, y):
        stacking_train = np.zeros(
            (np.shape(X)[0], len(self.base_classifiers)),
            # np.nan
        )  # 用于存储堆叠训练数据集的二维数组：输入数据集的样本数，基本分类器的数目
        print(stacking_train)

        for model_no in range( len(self.base_classifiers) ):
            if self.base_classifiers[model_no] == 'lr_model':
                X, y = shuffle(X, y)
                self.base_classifiers[model_no].fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
            elif self.base_classifiers[model_no] == 'xgb_model':
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.base_classifiers[model_no].fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
            elif self.base_classifiers[model_no] == 'mul_model':
                scaler = MinMaxScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                print(X)
                self.base_classifiers[model_no].fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)

            else:
                self.base_classifiers[model_no].fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
                # print('开始')
                # print(stacking_train)
                # stacking_train[model_no,model_no] = predicted_y_proba

        self.combiner.fit(stacking_train, y)

    def predict(self, X):
        """
        使用堆叠模型进行预测。

        参数:
        - X: 输入数据矩阵，其中每一行代表一个样本。

        返回值:
        - 预测结果，基于输入数据X通过堆叠模型得到的输出。
        """
        stacking_predict_data = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for model_no in range(len(self.base_classifiers)):
            if self.base_classifiers[model_no] == 'mul_model':
                scaler = MinMaxScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                stacking_predict_data[:, model_no] = self.base_classifiers[model_no].predict(X)
            else:
                stacking_predict_data[:, model_no] = self.base_classifiers[model_no].predict(X)
        return self.combiner.predict(stacking_predict_data)

    def partial_fit(self, X, y):
        stacking_train = np.zeros(
            (np.shape(X)[0], len(self.base_classifiers)),
            # np.nan
        )  # 用于存储堆叠训练数据集的二维数组：输入数据集的样本数，基本分类器的数目
        # print(stacking_train)

        # for model_no in range(len(self.base_classifiers) - 3):
        for model_no in range(len(self.base_classifiers)):

            if self.base_classifiers[model_no] == 'lr_model':
                X, y = shuffle(X, y)
                self.base_classifiers[model_no].partial_fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
            elif self.base_classifiers[model_no] == 'xgb_model':
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.base_classifiers[model_no].partial_fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
            elif self.base_classifiers[model_no] == 'mul_model':
                scaler = MinMaxScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                self.base_classifiers[model_no].partial_fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
            elif self.base_classifiers[model_no] == xgb.XGBClassifier():
                print("ture")
                continue

            else:
                self.base_classifiers[model_no].partial_fit(X, y)
                stacking_train[:, model_no] = self.base_classifiers[model_no].predict(X)
                # print('开始')
                # print(stacking_train)
                # stacking_train[model_no,model_no] = predicted_y_proba

        self.combiner.partial_fit(stacking_train, y)
