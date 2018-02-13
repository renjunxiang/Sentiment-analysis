import matplotlib.pyplot as plt


def keras_log_plot(train_log=None):
    plt.plot(train_log['acc'], label='acc', color='red')
    plt.plot(train_log['loss'], label='loss', color='yellow')
    if 'val_acc' in train_log.columns:
        plt.plot(train_log['val_acc'], label='val_acc', color='green')
    if 'val_loss' in train_log.columns:
        plt.plot(train_log['val_loss'], label='val_loss', color='blue')
    plt.legend()
    plt.show()
