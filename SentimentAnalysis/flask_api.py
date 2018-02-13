# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
import os
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
api = Api(app)

DIR = os.path.dirname(__file__)
class sentiment_analyse(Resource):
    def get(self):
        model_name = request.args.get('model_name')
        text = request.args.get('text')
        prob=request.args.get('prob')
        '''
        model_name='SVM'
        text='刚买就降价了桑心'
        '''

        model = SentimentAnalysis()
        # 导入词向量词包
        model.load_vocab_word2vec(vocab_loadpath=DIR + '/models/vocab_word2vec.model')

        if model_name in ['SVM', 'KNN', 'Logistic']:
            # 导入机器学习模型
            model.load_model(model_loadpath=DIR + '/models/classify.model',
                             model_name=model_name,
                             data_info_path=DIR + '/data_info.json')
        elif model_name in ['Conv1D_LSTM', 'Conv1D', 'LSTM']:
            # 导入深度学习模型
            model.load_model(model_loadpath=DIR + '/models/classify.h5',
                             model_name=model_name,
                             data_info_path=DIR + '/data_info.json')

        try:
            if prob == '1':
                # 进行预测:概率
                result_prob = model.predict_prob(texts=[text])
                result_prob = result_prob.astype(np.float64)
                result_prob = pd.DataFrame(result_prob, columns=model.label)
                result_prob['predict'] = result_prob.idxmax(axis=1)
                result_prob['text'] = [text]
                result_prob = result_prob[['text'] + list(model.label) + ['predict']]
                result = [{i: result_prob.loc[0, i]} for i in ['text'] + list(model.label) + ['predict']]
            else:
                # 进行预测:类别
                result_classify = model.predict(texts=[text])
                result = [{'text': text},{'predict':result_classify[0]}]

            return result
        except Exception as e:
            return '该文本的词语均不在词库中，无法识别'+' (error: %s)'%e

#http://192.168.3.59:5000/SentimentAnalyse/?model_name=Conv1D&prob=1&text=东西为什么这么烂
api.add_resource(sentiment_analyse, '/SentimentAnalyse/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')



