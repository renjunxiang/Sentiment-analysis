import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import History
from keras.models import load_model
import jieba
import json

from SentimentAnalysis.creat_data import bat
from SentimentAnalysis.sentence_transform.creat_vocab_word2vec import creat_vocab_word2vec
from SentimentAnalysis.models.sklearn_supervised import sklearn_supervised
from SentimentAnalysis.models import sklearn_config
from SentimentAnalysis.models.neural_bulit import neural_bulit

jieba.setLogLevel('WARN')
DIR = os.path.dirname(__file__)

class SentimentAnalysis():
    # def __init__(self):
    #     pass

    # open api
    def open_api(self):
        os.system("python %s" % (DIR + '/flask_api.py'))
        os.close()

    # get labels from baidu,ali,tencent
    def creat_label(self, texts):
        results_dataframe = bat.creat_label(texts)
        return results_dataframe

    def creat_vocab(self,
                    texts=None,
                    sg=0,
                    size=5,
                    window=5,
                    min_count=1,
                    vocab_savepath=DIR + '/models/vocab_word2vec.model'):
        '''
        get dictionary by word2vec
        :param texts: list of text
        :param sg: 0 CBOW,1 skip-gram
        :param size: the dimensionality of the feature vectors
        :param window: the maximum distance between the current and predicted word within a sentence
        :param min_count: ignore all words with total frequency lower than this
        :param vocab_savepath: path to save word2vec dictionary
        :return: None
        '''
        # 构建词向量词库
        self.vocab_word2vec = creat_vocab_word2vec(texts=texts,
                                                   sg=sg,
                                                   vocab_savepath=vocab_savepath,
                                                   size=size,
                                                   window=window,
                                                   min_count=min_count)

    def load_vocab_word2vec(self,
                            vocab_loadpath=DIR + '/models/vocab_word2vec.model'):
        '''
        load dictionary
        :param vocab_loadpath: path to load word2vec dictionary
        :return: 
        '''
        self.vocab_word2vec = word2vec.Word2Vec.load(vocab_loadpath)

    def train(self,
              texts=None,
              label=None,
              model_name='SVM',
              model_savepath=DIR + '/models/classify.model',
              net_shape=None,
              batch_size=100,  # 神经网络参数
              epochs=2,  # 神经网络参数
              verbose=2,  # 神经网络参数
              maxlen=None,  # 神经网络参数
              **sklearn_param):
        '''
        use sklearn/keras to train model
        :param texts: x
        :param label: y
        :param model_name: name want to train
        :param model_savepath: model save path
        :param batch_size: for keras fit
        :param epochs: for keras fit
        :param verbose: for keras fit
        :param maxlen: for keras pad_sequences
        :param sklearn_param: param for sklearn
        :return: None
        '''
        self.model_name = model_name
        self.label = np.unique(np.array(label))
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
        data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
        if maxlen is None:
            maxlen = max([len(i) for i in texts_cut])
        self.maxlen = maxlen
        # sklearn模型，词向量计算均值
        if model_name in ['SVM', 'KNN', 'Logistic']:
            data = [sum(i) / len(i) for i in data]
            # 配置sklearn模型参数
            if model_name == 'SVM':
                if sklearn_param == {}:
                    sklearn_param = sklearn_config.SVC
            elif model_name == 'KNN':
                if sklearn_param == {}:
                    sklearn_param = sklearn_config.KNN
            elif model_name == 'Logistic':
                if sklearn_param == {}:
                    sklearn_param = sklearn_config.Logistic
            # 返回训练模型
            self.model = sklearn_supervised(data=data,
                                            label=label,
                                            model_savepath=model_savepath,
                                            model_name=model_name,
                                            **sklearn_param)

        # keras神经网络模型，
        elif model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            data = pad_sequences(data, maxlen=maxlen, padding='post', value=0, dtype='float32')
            label_transform = np.array(pd.get_dummies(label))
            if net_shape is None:
                if model_name == 'Conv1D_LSTM':
                    net_shape = [
                        {'name': 'InputLayer', 'input_shape': data.shape[1:]},
                        {'name': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same',
                         'activation': 'relu'},
                        {'name': 'MaxPooling1D', 'pool_size': 5, 'padding': 'same', 'strides': 2},
                        {'name': 'LSTM', 'units': 16, 'activation': 'tanh', 'recurrent_activation': 'hard_sigmoid',
                         'dropout': 0., 'recurrent_dropout': 0.},
                        {'name': 'Flatten'},
                        {'name': 'Dense', 'activation': 'relu', 'units': 64},
                        {'name': 'Dropout', 'rate': 0.2, },
                        {'name': 'softmax', 'activation': 'softmax', 'units': len(np.unique(label))}
                    ]

                elif model_name == 'LSTM':
                    net_shape = [
                        {'name': 'InputLayer', 'input_shape': data.shape[1:]},
                        {'name': 'Masking'},
                        {'name': 'LSTM', 'units': 16, 'activation': 'tanh', 'recurrent_activation': 'hard_sigmoid',
                         'dropout': 0., 'recurrent_dropout': 0.},
                        {'name': 'Dense', 'activation': 'relu', 'units': 64},
                        {'name': 'Dropout', 'rate': 0.2, },
                        {'name': 'softmax', 'activation': 'softmax', 'units': len(np.unique(label))}
                    ]
                elif model_name == 'Conv1D':
                    net_shape = [
                        {'name': 'InputLayer', 'input_shape': data.shape[1:]},
                        {'name': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same',
                         'activation': 'relu'},
                        {'name': 'MaxPooling1D', 'pool_size': 5, 'padding': 'same', 'strides': 2},
                        {'name': 'Flatten'},
                        {'name': 'Dense', 'activation': 'relu', 'units': 64},
                        {'name': 'Dropout', 'rate': 0.2, },
                        {'name': 'softmax', 'activation': 'softmax', 'units': len(np.unique(label))}
                    ]

            model = neural_bulit(net_shape=net_shape,
                                 optimizer_name='Adagrad',
                                 lr=0.001,
                                 loss='categorical_crossentropy')
            history = History()
            model.fit(data, label_transform,
                      batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[history])
            train_log = pd.DataFrame(history.history)
            self.model = model
            self.train_log = train_log
            if model_savepath != None:
                model.save(model_savepath)
        with open(DIR + '/data_info.json', mode='w', encoding='utf-8') as f:
            json.dump({'maxlen': maxlen, 'label': list(self.label)}, f)

    def load_model(self,
                   model_loadpath=DIR + '/models/classify.model',
                   model_name=None,
                   data_info_path=DIR + '/data_info.json'):
        '''
        load sklearn/keras model
        :param model_loadpath: path to load sklearn/keras model
        :param model_name: load model name 
        :param data_info_path: date information path
        :return: None
        '''

        with open(data_info_path, encoding='utf-8') as f:
            data_info = json.load(f)
        self.maxlen = data_info['maxlen']
        self.label = data_info['label']
        self.model_name = model_name

        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            self.model = joblib.load(model_loadpath)
        elif self.model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            self.model = load_model(model_loadpath)

    def predict_prob(self,
                     texts=None):
        '''
        predict probability
        :param texts:  list of text
        :return: list of probability
        '''
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = [sum(i) / len(i) for i in data]
            self.testdata = data
            results = self.model.predict_proba(data)
        elif self.model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = pad_sequences(data, maxlen=self.maxlen, padding='post', value=0, dtype='float32')
            self.testdata = data
            results = self.model.predict(data)
        return results

    def predict(self,
                texts=None):
        '''
        predict class
        :param texts:  list of text
        :return: list of classify
        '''
        # 文本转词向量
        vocab_word2vec = self.vocab_word2vec
        if self.model_name in ['SVM', 'KNN', 'Logistic']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = [sum(i) / len(i) for i in data]
            self.testdata = data
            results = self.model.predict(data)
        elif self.model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
            data = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in texts_cut]
            data = pad_sequences(data, maxlen=self.maxlen, padding='post', value=0, dtype='float32')
            self.testdata = data
            results = self.model.predict(data)
            results = pd.DataFrame(results, columns=self.label)
            results = results.idxmax(axis=1)
        return results


if __name__ == '__main__':
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

    model.creat_vocab(texts=train_data,
                      sg=0,
                      size=5,
                      window=5,
                      min_count=1,
                      vocab_savepath=None)

    # 导入词向量词包
    # model.load_vocab_word2vec(vocab_loadpath=DIR + '/vocab_word2vec.model')

    ###################################################################################
    # 进行机器学习
    model.train(texts=train_data,
                label=train_label,
                model_name='SVM',
                model_savepath=DIR + '/models/classify.model')

    # 导入机器学习模型
    model.load_model(model_loadpath=DIR + '/models/classify.model',
                     model_name='SVM',
                     data_info_path=DIR + '/data_info.json')

    # 进行预测:概率
    result_prob = model.predict_prob(texts=test_data)
    result_prob = pd.DataFrame(result_prob, columns=model.label)
    result_prob['predict'] = result_prob.idxmax(axis=1)
    result_prob['data'] = test_data
    result_prob = result_prob[['data'] + list(model.label) + ['predict']]
    print('prob:\n', result_prob)
    print('score:', np.sum(result_prob['predict'] == np.array(test_label)) / len(result_prob['predict']))
