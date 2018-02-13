from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
import pandas as pd
import jieba
from gensim.models import word2vec, doc2vec
import numpy as np
import os

jieba.setLogLevel('WARN')
DIR = os.path.dirname(__file__)

def creat_vocab_word2vec(texts=None,
                         sg=0,
                         vocab_savepath=DIR + '/vocab_word2vec.model',
                         size=5,
                         window=5,
                         min_count=1):
    '''
    
    :param texts: list of text
    :param sg: 0 CBOW,1 skip-gram
    :param size: the dimensionality of the feature vectors
    :param window: the maximum distance between the current and predicted word within a sentence
    :param min_count: ignore all words with total frequency lower than this
    :param vocab_savepath: path to save word2vec dictionary
    :return: None
    
    '''
    texts_cut = [[word for word in jieba.lcut(one_text) if word != ' '] for one_text in texts]  # 分词
    # 训练
    model = word2vec.Word2Vec(texts_cut, sg=sg, size=size, window=window, min_count=min_count)
    if vocab_savepath != None:
        model.save(vocab_savepath)

    return model


if __name__ == '__main__':
    texts = ['全面从严治党',
             '国际公约和国际法',
             '中国航天科技集团有限公司']
    vocab_word2vec = creat_vocab_word2vec(texts=texts,
                                          sg=0,
                                          vocab_savepath=DIR + '/vocab_word2vec.model',
                                          size=5,
                                          window=5,
                                          min_count=1)
