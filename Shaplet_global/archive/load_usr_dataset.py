# -*- coding: utf-8 -*-
import pandas
from config import *
#from os import path

def load_usr_dataset_by_name(fname, length,quantitative,seg):
    """
    load UCR dataset given dataset name.
    :param fname:
        dataset name, e.g., Earthquakes.
    :param length:
        time series length that want to load in.
    :return:
    """
    dir_path = '/data/usr/lhr/Time_shapelet/Time_series'
    #print(dir_path)
    assert path.isfile('{}/{}/{}/{}_TEST.tsv'.format(dir_path,quantitative,seg, fname)), '{} NOT EXIST in UCR!'.format(fname)
    train_data = pandas.read_csv('{}/{}/{}/{}_TRAIN.tsv'.format(dir_path,quantitative,seg, fname), sep='\t', header=None)
    test_data = pandas.read_csv('{}/{}/{}/{}_TEST.tsv'.format(dir_path,quantitative,seg, fname), sep='\t', header=None)
    init = train_data.shape[1] - length
    #513-504（EQS地震局数据列数是513，504是args.seg_length * args.num_segment，即24*21,21是通过512/24计算出的，我们的数据目前seg_length是1）
    x_train, y_train = train_data.values[:, init:].astype(np.float).reshape(-1, length, 1), \
                       train_data[0].values.astype(np.int)#x_train (322,504,1)  y_train (322,1)
    x_test, y_test = test_data.values[:, init:].astype(np.float).reshape(-1, length, 1), \
                     test_data[0].values.astype(np.int)
    lbs = np.unique(y_train)
    y_train_return, y_test_return = np.copy(y_train), np.copy(y_test)
    for idx, val in enumerate(lbs):
        y_train_return[y_train == val] = idx
        y_test_return[y_test == val] = idx
    #标签重新映射：将原始标签重新映射为从0开始的连续整数。
    return x_train, y_train_return, x_test, y_test_return


