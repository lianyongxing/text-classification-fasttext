# -*- coding: utf-8 -*-
# @Time    : 2/10/23 7:17 PM
# @Author  : LIANYONGXING
# @FileName: train.py

import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import load_data, make_fasttext_data_format
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

def valid(model, sentences, labels):
    valid_sentences = [' '.join([i for i in sentence]) for sentence in sentences]
    res_lab, res_score = model.predict(valid_sentences)
    res = [[int(res_lab[i][0].strip('__label__')), res_score[i][0]] for i in range(len(res_lab))]
    res_lab = [int(res_lab[i][0].strip('__label__')) for i in range(len(res_lab))]
    res_score =  [res_score[i][0] for i in range(len(res_score))]
    logging.info('\n' + classification_report(labels, [i[0] for i in res], target_names=class_names))
    return res_lab, res_score

if __name__ == '__main__':

    # 0. params
    class_names = ['正常', '违规']

    epoch           = 2             # epoch                         [50]
    lr              = 0.5           # learning rate                 [0.01]
    dim             = 100           # embedding dim                 [128]
    ws              = 5             # size of the context window
    minn            = 1             # min length of char ngram
    maxn            = 2             # max length of char ngram
    neg             = 4             # number of negative sampled
    wordNgram       = 2             # max length of word ngram      [3]
    loss            = 'softmax'     # loss function
    lrUpdateRate    = 50            # lr update rate

    test_ratio      = 0.2
    random_state    = 20

    train_raw_filepath = './datas/train_dats_cnews.csv'             # input raw files
    model_save_path = './outputs/model_test.pt'                     # path to model save
    valid_result_save_path = './datas/valid_result.csv'             # path to save valid results

    train_set_temppath = './datas/train.txt'

    # 1. Train model
    datas = load_data(train_raw_filepath)

    train_set, valid_set = train_test_split(datas, test_size=test_ratio, random_state=random_state)
    train_datas = make_fasttext_data_format(train_set['content_filter'].tolist(), train_set['lab'].tolist(), train_set_temppath)

    model = fasttext.train_supervised(train_set_temppath,
                                      label='__label__',
                                      dim=dim,
                                      lr=lr,
                                      ws=ws,
                                      minn=minn,
                                      maxn=maxn,
                                      neg=neg,
                                      word_ngrams=wordNgram,
                                      epoch=epoch,
                                      loss=loss,
                                      lrUpdateRate=lrUpdateRate)

    model.save_model(model_save_path)
    logging.info('模型训练并且存储完成： %s' % model_save_path)

    # 2. Eval model
    res_lab, res_score = valid(model, valid_set['content_filter'].tolist(), valid_set['lab'].tolist())
    valid_set['pre_lab'] = res_lab
    valid_set['pre_score'] = res_score
    valid_set.to_csv(valid_result_save_path, index=False, encoding='utf_8_sig')
    logging.info('验证集结果输出完成： %s' % valid_result_save_path)