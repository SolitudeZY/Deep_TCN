# from sklearn.linear_model import LogisticRegression  # 例如，用Logistic Regression作为基础分类器
# from sklearn.ensemble import RandomForestClassifier  # 例如，用Random Forest作为基础分类器
# from sklearn.svm import SVC  # 例如，用SVM作为基础分类器
from rnn import RNN
from model import TCN
from cnn import CNN
import lstm
from stacking_esemble_learning import StackingClassifier
from utils import data_generator, save

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 实例化模型： 首先，根据你的数据特性，定义合适的参数，如输入特征的数量、输出类别数、TCN的通道数、卷积核大小以及丢弃率，然后创建TCN对象：
input_size = 10  # 例如，假设输入数据有10个特征
output_size = 5  # 假设有5个不同的类别
num_channels = [32, 64, 128]  # 可以根据需要调整
kernel_size = 2
dropout = 0.5

base_classifiers = [
    ('rnn_model', RNN()),
    ('tcn_model', TCN(input_size, output_size, num_channels, kernel_size, dropout)),
    ('cnn_model', CNN())
    ("lstm", lstm.RNN())
]
model = lstm.RNN()
# 你可以选择不同的组合器，例如使用RNN作为组合器
combiner = RNN()

# 创建分类器
stacking_clf = StackingClassifier(base_classifiers, combiner)
# 数据集生成
dataset = data_generator('./dataset', 128)

X_train, y_train = dataset  # 这里加载你的训练数据和标签
X_test, y_test = dataset  # 加载测试数据和标签

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)

from sklearn.metrics import accuracy_score

y_pred_proba = stacking_clf.predict(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
