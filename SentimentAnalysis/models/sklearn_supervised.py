from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import os

DIR = os.path.dirname(__file__)
def sklearn_supervised(data=None,
                       label=None,
                       model_savepath=DIR + '/sentence_transform/classify.model',
                       model_name='SVM',
                       **sklearn_param):
    '''
    :param data: 训练文本
    :param label: 训练文本的标签
    :param model_savepath: 模型保存路径
    :param model_name: 机器学习分类模型,SVM,KNN,Logistic
    :param return: 训练好的模型
    '''

    if model_name == 'KNN':
        # 调用KNN,近邻=5
        model = KNeighborsClassifier(**sklearn_param)
    elif model_name == 'SVM':
        # 核函数为linear,惩罚系数为1.0
        model = SVC(**sklearn_param)
        model.fit(data, label)
    elif model_name == 'Logistic':
        model = LogisticRegression(**sklearn_param)  # 核函数为线性,惩罚系数为1
        model.fit(data, label)

    if model_savepath != None:
        joblib.dump(model, model_savepath)  # 保存模型


    return model
