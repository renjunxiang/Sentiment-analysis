import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

from SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from SentimentAnalysis.models.keras_log_plot import keras_log_plot

model = SentimentAnalysis()

dataset = pd.read_excel(os.getcwd() + '/data/traindata.xlsx', sheet_name=0)
data = dataset['evaluation']
label = dataset['label']
train_data, test_data, train_label, test_label = train_test_split(data,
                                                                  label,
                                                                  test_size=0.1,
                                                                  random_state=42)
test_data = test_data.reset_index(drop=True)
test_label = test_label.reset_index(drop=True)
# 建模获取词向量词包
model.creat_vocab(texts=train_data,
                  sg=0,
                  size=5,
                  window=5,
                  min_count=1,
                  vocab_savepath=None)

# 导入词向量词包
# model.load_vocab_word2vec(vocab_loadpath=os.getcwd() + '/vocab_word2vec.model')

###################################################################################
# 进行机器学习
model.train(texts=train_data,
            label=train_label,
            model_name='SVM',
            model_savepath=os.getcwd() + '/models/classify.model')

# 导入机器学习模型
model.load_model(model_loadpath=os.getcwd() + '/models/classify.model',
                 model_name='SVM',
                 data_info_path=os.getcwd() + '/data_info.json')

# 进行预测:概率
result_prob = model.predict_prob(texts=test_data)
result_prob = pd.DataFrame(result_prob, columns=model.label)
result_prob['predict'] = result_prob.idxmax(axis=1)
result_prob['data'] = test_data
result_prob = result_prob[['data'] + list(model.label) + ['predict']]
print('prob:\n', result_prob)
print('score:', np.sum(result_prob['predict'] == np.array(test_label)) / len(result_prob['predict']))

###################################################################################
# 进行深度学习
model.train(texts=train_data,
            label=train_label,
            model_name='Conv1D',
            batch_size=200,
            epochs=20,
            verbose=2,
            maxlen=None,
            model_savepath=os.getcwd() + '/models/classify.h5')

# 导入深度学习模型
model.load_model(model_loadpath=os.getcwd() + '/models/classify.h5',
                 model_name='Conv1D',
                 data_info_path=os.getcwd() + '/data_info.json')

# 进行预测:概率
result_prob = model.predict_prob(texts=test_data)
result_prob = pd.DataFrame(result_prob, columns=model.label)
result_prob['predict'] = result_prob.idxmax(axis=1)
result_prob['data'] = test_data
result_prob = result_prob[['data'] + list(model.label) + ['predict']]
print('prob:\n', result_prob)
print('score:', np.sum(result_prob['predict'] == np.array(test_label)) / len(result_prob['predict']))

keras_log_plot(model.train_log)
