from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D
from SentimentAnalysis.models.parameter.optimizers import optimizers

def neural_bulit(net_shape,
                 optimizer_name='Adagrad',
                 lr=0.001,
                 loss='categorical_crossentropy'):
    '''
    :param net_shape: 神经网络格式
    net_shape = [
             {'name': 'InputLayer','input_shape': [10, 5]},
             {'name': 'Dropout','rate': 0.2,},
             {'name': 'Masking'},
             {'name': 'LSTM','units': 16,'activation': 'tanh','recurrent_activation': 'hard_sigmoid','dropout': 0.,'recurrent_dropout': 0.},
             {'name': 'Conv1D','filters': 64,'kernel_size': 3,'strides': 1,'padding': 'same','activation': 'relu'},
             {'name': 'MaxPooling1D','pool_size': 5,'padding': 'same','strides': 2},
             {'name': 'Flatten'},
             {'name': 'Dense','activation': 'relu','units': 64},
             {'name': 'softmax','activation': 'softmax','units': 2}
             ]
    :param optimizer_name: 优化器
    :param lr: 学习率
    :param loss: 损失函数
    :param return: 返回神经网络模型
    '''
    model = Sequential()

    def add_InputLayer(input_shape,
                       **param):
        model.add(InputLayer(input_shape=input_shape,
                             **param))

    def add_Dropout(rate=0.2,
                    **param):
        model.add(Dropout(rate=rate,
                          **param))

    def add_Masking(mask_value=0,
                    **param):
        model.add(Masking(mask_value=mask_value,
                          **param))

    def add_LSTM(units=16,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 implementation=1,
                 dropout=0,
                 recurrent_dropout=0,
                 **param):
        model.add(LSTM(units=units,
                       activation=activation,
                       recurrent_activation=recurrent_activation,
                       implementation=implementation,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       **param))

    def add_Conv1D(filters=16,  # 卷积核数量
                   kernel_size=3,  # 卷积核尺寸，或者[3]
                   strides=1,
                   padding='same',
                   activation='relu',
                   kernel_initializer=initializers.normal(stddev=0.1),
                   bias_initializer=initializers.normal(stddev=0.1),
                   **param):
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         **param))

    def add_MaxPooling1D(pool_size=3,  # 卷积核尺寸，或者[3]
                         strides=1,
                         padding='same',
                         **param):
        model.add(MaxPooling1D(pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               **param))

    def add_Flatten(**param):
        model.add(Flatten(**param))

    def add_Dense(units=16,
                  activation='relu',
                  kernel_initializer=initializers.normal(stddev=0.1),
                  **param):
        model.add(Dense(units=units,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        **param))

    for n in range(len(net_shape)):
        if net_shape[n]['name'] == 'InputLayer':
            del net_shape[n]['name']
            add_InputLayer(name='num_' + str(n) + '_InputLayer',
                           **net_shape[n])
        elif net_shape[n]['name'] == 'Dropout':
            del net_shape[n]['name']
            add_Dropout(name='num_' + str(n) + '_Dropout',
                        **net_shape[n])
        elif net_shape[n]['name'] == 'Masking':
            del net_shape[n]['name']
            add_Masking(name='num_' + str(n) + '_Masking',
                        **net_shape[n])
        elif net_shape[n]['name'] == 'LSTM':
            del net_shape[n]['name']
            add_LSTM(name='num_' + str(n) + '_LSTM',
                     **net_shape[n])
        elif net_shape[n]['name'] == 'Conv1D':
            del net_shape[n]['name']
            add_Conv1D(name='num_' + str(n) + '_Conv1D',
                       **net_shape[n])
        elif net_shape[n]['name'] == 'MaxPooling1D':
            del net_shape[n]['name']
            add_MaxPooling1D(name='num_' + str(n) + '_MaxPooling1D',
                             **net_shape[n])
        elif net_shape[n]['name'] == 'Flatten':
            del net_shape[n]['name']
            add_Flatten(name='num_' + str(n) + '_Flatten',
                        **net_shape[n])
        elif net_shape[n]['name'] == 'Dense':
            del net_shape[n]['name']
            add_Dense(name='num_' + str(n) + '_Dense',
                      **net_shape[n])
        elif net_shape[n]['name'] == 'softmax':
            del net_shape[n]['name']
            add_Dense(name='num_' + str(n) + '_softmax',
                      **net_shape[n])

    optimizer = optimizers(name=optimizer_name, lr=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    net_shape = [{'name': 'InputLayer',
                  'input_shape': [10, 5],
                  },
                 {'name': 'Conv1D'
                  },
                 {'name': 'MaxPooling1D'
                  },
                 {'name': 'Flatten'
                  },
                 {'name': 'Dense'
                  },
                 {'name': 'Dropout'
                  },
                 {'name': 'softmax',
                  'activation': 'softmax',
                  'units': 3
                  }
                 ]
    model = neural_bulit(net_shape=net_shape,
                         optimizer_name='Adagrad',
                         lr=0.001,
                         loss='categorical_crossentropy')
    model.summary()
