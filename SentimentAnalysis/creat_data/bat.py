from SentimentAnalysis.creat_data import baidu, ali, tencent
import pandas as pd
# from collections import OrderedDict
import numpy as np


def creat_label(texts):
    results = []
    count_i = 0
    for one_text in texts:
        result_baidu = baidu.creat_label([one_text], interface='API')
        result_ali = ali.creat_label([one_text])
        result_tencent = tencent.creat_label([one_text])

        result_all = [one_text,
                      result_baidu[0][1], result_baidu[0][6],
                      result_ali[0][1], result_ali[0][3],
                      result_tencent[0][1], result_tencent[0][4]]
        results.append(result_all)

        # result = OrderedDict()
        # result['evaluation'] = result_all[0]
        # result['label_baidu'] = result_all[1]
        # result['msg_baidu'] = result_all[2]
        # result['label_ali'] = result_all[3]
        # result['msg_ali'] = result_all[4]
        # result['label_tencent'] = result_all[5]
        # result['msg_tencent'] = result_all[6]

        count_i += 1
        if count_i % 50 == 0:
            print('baidu finish:%d' % (count_i))

    results_dataframe = pd.DataFrame(results,
                                     columns=['evaluation',
                                              'label_baidu', 'msg_baidu',
                                              'label_ali', 'msg_ali',
                                              'label_tencent', 'msg_tencent'])
    results_dataframe['label_baidu'] = np.where(results_dataframe['label_baidu'] == 2,
                                                '正面',
                                                np.where(results_dataframe['label_baidu'] == 1, '中性', '负面'))
    results_dataframe['label_ali'] = np.where(results_dataframe['label_ali'] == '1', '正面',
                                              np.where(results_dataframe['label_ali'] == '0', '中性',
                                                       np.where(results_dataframe['label_ali'] == '-1', '负面', '非法')))
    results_dataframe['label_tencent'] = np.where(results_dataframe['label_tencent'] == 1, '正面',
                                                  np.where(results_dataframe['label_tencent'] == 0, '中性', '负面'))
    return results_dataframe


if __name__ == '__main__':
    print(creat_label(['价格便宜啦，比原来优惠多了',
                       '壁挂效果差，果然一分价钱一分货',
                       '东西一般般，诶呀',
                       '讨厌你',
                       '一般'
                       ]))
