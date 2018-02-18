# Sentiment-analysis:情感分析

[![](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/)<br>
[![](https://img.shields.io/badge/baidu--aip-2.1.0.0-brightgreen.svg)](https://pypi.python.org/pypi/baidu-aip/2.1.0.0)
[![](https://img.shields.io/badge/pandas-0.21.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.21.0)
[![](https://img.shields.io/badge/numpy-1.13.1-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.13.1)
[![](https://img.shields.io/badge/matplotlib-2.1.0-brightgreen.svg)](https://pypi.python.org/pypi/matplotlib/2.1.0)
[![](https://img.shields.io/badge/jieba-0.39-brightgreen.svg)](https://pypi.python.org/pypi/jieba/0.39)
[![](https://img.shields.io/badge/gensim-3.2.0-brightgreen.svg)](https://pypi.python.org/pypi/gensim/3.2.0)
[![](https://img.shields.io/badge/scikit--learn-0.19.1-brightgreen.svg)](https://pypi.python.org/pypi/scikit-learn/0.19.1)
[![](https://img.shields.io/badge/requests-2.18.4-brightgreen.svg)](https://pypi.python.org/pypi/requests/2.18.4)

## 语言
Python3.5<br>
## 依赖库
baidu-aip=2.1.0.0<br>
pandas=0.21.0<br>
numpy=1.13.1<br>
jieba=0.39<br>
gensim=3.2.0<br>
scikit-learn=0.19.1<br>
keras=2.1.1<br>
requests=2.18.4<br>



## 项目介绍
通过对已有标签的帖子进行训练，实现新帖子的情感分类，用法类似scikit-learn。<br>
已完成机器学习算法中KNN、SVM和Logistic的封装，神经网络算法中的一维卷积核LSTM封装。训练集为一万条记录，SVM效果最好，准确率在87%左右.<br>
SentimentAnalysis文件夹可以直接作为模块使用<br>
***PS：该项目在上一个项目Text-Classification基础上封装而成~目前公司情感分析借鉴这个项目，有很多不足，欢迎萌新、大佬多多指导！***

## 用法简介
### 1.导入模块，创建模型
``` python
from SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
model = SentimentAnalysis()
```

### 2.借助第三方平台，打情感分析标签。用于在缺乏标签的时候利用BAT三家的接口创建训练集，5000条文档共耗时约45分钟
``` python
texts=['国王喜欢吃苹果',
       '国王非常喜欢吃苹果',
       '国王讨厌吃苹果',
       '国王非常讨厌吃苹果']
texts_withlabel=model.creat_label(texts)
```

### 3.通过gensim模块创建词向量词包
``` python
model.creat_vocab(texts=texts,
                  sg=0,
                  size=5,
                  window=5,
                  min_count=1,
                  vocab_savepath=os.getcwd() + '/vocab_word2vec.model')
# 也可以导入词向量
model.load_vocab_word2vec(os.getcwd() + '/models/vocab_word2vec.model')
# 词向量模型
model.vocab_word2vec
```

### 4.通过scikit-learn进行机器学习
``` python
model.train(texts=train_data,
            label=train_label,
            model_name='SVM',
            model_savepath=os.getcwd() + '/classify.model')
# 也可以导入机器学习模型
model.load_model(model_loadpath=os.getcwd() + '/classify.model')
# 训练的模型
model.model
# 训练集标签
model.label
```

### 5.通过keras进行深度学习(模型的后缀不同)
``` python
model.train(texts=train_data,
            label=train_label,
            model_name='Conv1D',
            batch_size=100,
            epochs=2,
            verbose=1,
            maxlen=None,
            model_savepath=os.getcwd() + '/classify.h5')

# 导入深度学习模型
model.load_model(model_loadpath=os.getcwd() + '/classify.h5')
# 训练的模型
model.model
# 训练的日志
model.train_log
# 可视化训练过程
from SentimentAnalysis.models.keras_log_plot import keras_log_plot
keras_log_plot(model.train_log)
# 训练集标签
model.label
```

### 6.预测
``` python
# 概率
result_prob = model.predict_prob(texts=test_data)
result_prob = pd.DataFrame(result_prob, columns=model.label)
result_prob['predict'] = result_prob.idxmax(axis=1)
result_prob['data'] = test_data
result_prob = result_prob[['data'] + list(model.label) + ['predict']]
print('prob:\n', result_prob)

# 分类
result = model.predict(texts=test_data)
print('score:', np.sum(result == np.array(test_label)) / len(result))
result = pd.DataFrame({'data': test_data,
                       'label': test_label,
                       'predict': result},
                      columns=['data', 'label', 'predict'])
print('test\n', result)
```

### 7.开启API
``` python
model.open_api()
#http://0.0.0.0:5000/SentimentAnalyse/?model_name=模型名称&prob=是否需要返回概率&text=分类文本
```

### 其他说明
在训练集很小的情况下，sklearn的概率输出predict_prob会不准。目前发现，SVM会出现所有标签概率一样，暂时没看源码，猜测是离超平面过近不计算概率，predict不会出现这个情况。

## 一个简单的demo
``` python
from SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from SentimentAnalysis.models.keras_log_plot import keras_log_plot
import numpy as np

train_data = ['国王喜欢吃苹果',
              '国王非常喜欢吃苹果',
              '国王讨厌吃苹果',
              '国王非常讨厌吃苹果']
train_label = ['正面', '正面', '负面', '负面']
# print('train data\n',
#       pd.DataFrame({'data': train_data,
#                     'label': train_label},
#                    columns=['data', 'label']))
test_data = ['涛哥喜欢吃苹果',
             '涛哥讨厌吃苹果',
             '涛哥非常喜欢吃苹果',
             '涛哥非常讨厌吃苹果']
test_label = ['正面', '负面', '正面', '负面']

# 创建模型
model = SentimentAnalysis()

# 查看bat打的标签
print(model.creat_label(test_data))

# 建模获取词向量词包
model.creat_vocab(texts=train_data,
                  sg=0,
                  size=5,
                  window=5,
                  min_count=1,
                  vocab_savepath=os.getcwd() + '/vocab_word2vec.model')

# 导入词向量词包
# model.load_vocab_word2vec(vocab_loadpath=os.getcwd() + '/vocab_word2vec.model')

###################################################################################
# 进行机器学习
model.train(texts=train_data,
            label=train_label,
            model_name='SVM',
            model_savepath=os.getcwd() + '/classify.model')

# 导入机器学习模型
# model.load_model(model_loadpath=os.getcwd() + '/classify.model')

# 进行预测:概率
result_prob = model.predict_prob(texts=test_data)
result_prob = pd.DataFrame(result_prob, columns=model.label)
result_prob['predict'] = result_prob.idxmax(axis=1)
result_prob['data'] = test_data
result_prob = result_prob[['data'] + list(model.label) + ['predict']]
print('prob:\n', result_prob)

# 进行预测:分类
result = model.predict(texts=test_data)
print('score:', np.sum(result == np.array(test_label)) / len(result))
result = pd.DataFrame({'data': test_data,
                       'label': test_label,
                       'predict': result},
                      columns=['data', 'label', 'predict'])
print('test\n', result)
###################################################################################
# 进行深度学习
model.train(texts=train_data,
            label=train_label,
            model_name='Conv1D',
            batch_size=100,
            epochs=2,
            verbose=1,
            maxlen=None,
            model_savepath=os.getcwd() + '/classify.h5')

# 导入深度学习模型
# model.load_model(model_loadpath=os.getcwd() + '/classify.h5')

# 进行预测:概率
result_prob = model.predict_prob(texts=test_data)
result_prob = pd.DataFrame(result_prob, columns=model.label)
result_prob['predict'] = result_prob.idxmax(axis=1)
print(result_prob)

# 进行预测:分类
result = model.predict(texts=test_data)
print(result)
print('score:', np.sum(result == np.array(test_label)) / len(result))
result = pd.DataFrame({'data': test_data,
                       'label': test_label,
                       'predict': result},
                      columns=['data', 'label', 'predict'])
print('test\n', result)

keras_log_plot(model.train_log)

```
bat打标签<br>
![bat](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/label.png)<br>
SVM<br>
![SVM](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/SVM.png)<br>
Conv1D<br>
![Conv1D](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/Conv1D.png)<br>

## 简单API
做了一个api：http://192.168.3.59:5000/SentimentAnalyse/?model_name=模型名称&prob=是否需要返回概率&text=分类文本<br>
192.168.3.59：ip地址，由服务器决定<br>
模型名称：目前支持：SVM,Conv1D<br>
prob：0返回分类，1返回概率<br>
``` python
from SentimentAnalysis import SentimentAnalysis
model = SentimentAnalysis()
model.open_api()
```
<br>
例子<br>

SVM模型，返回概率<br>
url:http://192.168.3.59:5000/SentimentAnalyse/?model_name=SVM&prob=1&text=东西很不错<br>
![api1](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/api1.png)<br>

Conv1D模型，返回分类<br>
url:http://192.168.3.59:5000/SentimentAnalyse/?model_name=Conv1D&prob=0&text=东西很不错<br>
![api2](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/api2.png)<br>

文本中的词语均不在词向量词库中<br>
url:http://192.168.3.59:5000/SentimentAnalyse/?model_name=SVM&prob=0&text=呜呜呜<br>
![api3](https://github.com/renjunxiang/Sentiment-analysis/blob/master/picture/api3.png)<br>













