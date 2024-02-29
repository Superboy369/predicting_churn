import pandas as pd
import re
import jieba


def data_process(file='D:\\@我的记录文件夹\\临时文件夹\\message80W1\\message80W1.csv'):
    data = pd.read_csv(file, header=None, index_col=0)
    data.columns = ['label', 'message']
    n = 5000

    a = data[data['label'] == 0].sample(n)
    b = data[data['label'] == 1].sample(n)
    data_new = pd.concat([a, b], axis=0)

    data_dup = data_new['message'].drop_duplicates()
    data_qumin = data_dup.apply(lambda x: re.sub('x', '', x))

    jieba.load_userdict('D:\\@我的记录文件夹\\临时文件夹\\newdic1.txt')
    data_cut = data_qumin.apply(lambda x: jieba.lcut(x))

    stopWords = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\stopword.txt', encoding='GB18030', sep='hahaha', header=None)
    stopWords = ['≮', '≯', '≠', '≮', ' ', '会', '月', '日', '–'] + list(stopWords.iloc[:, 0])
    data_after_stop = data_cut.apply(lambda x: [i for i in x if i not in stopWords])
    labels = data_new.loc[data_after_stop.index, 'label']
    adata = data_after_stop.apply(lambda x: ' '.join(x))

    return adata, data_after_stop, labels

data_process()