# -*- coding: utf-8 -*-
# @Time    : 2/10/23 8:53 PM
# @Author  : LIANYONGXING
# @FileName: test.py

import fasttext
from text_process import sentence_filter
from utils import load_data


def test_single(model, sententce):
    sent_filter = sentence_filter(sententce)
    sent_filter_eval_format = ' '.join([i for i in sent_filter])
    result = model.predict(sent_filter_eval_format)
    print('Predict sententce : "%s"' % sententce)
    print('label: %s' % (int(result[0][0].strip('__label__'))))
    print('score: %s' % (result[1][0])  )


def test(model, sentences):
    eval_sententces = [' '.join([i for i in cf]) for cf in sentences]
    res_lab, res_score = model.predict(eval_sententces)
    res_lab = [int(res_lab[i][0].strip('__label__')) for i in range(len(res_lab))]
    res_score =  [res_score[i][0] for i in range(len(res_score))]
    return res_lab, res_score


if __name__ == '__main__':

    # 0. Load model
    model_path = './outputs/model_test.pt'
    model = fasttext.load_model(model_path)

    # 1. Test Single Sentence
    test_sentence = "今晚你吃的撒撒爱上撒上海"
    test_single(model, test_sentence)


    # 2. Test Datasets
    test_data_filepath = './datas/yq_comments.csv'      # test data path
    test_result_save_path = './outputs/result.csv'      # path to save result

    test_datas = load_data(test_data_filepath)
    labs, scores = test(model, test_datas['content_filter'].tolist())

    test_datas['pre_lab'] = labs
    test_datas['pre_scores'] = scores

    test_datas.to_csv(test_result_save_path, index=False, encoding='utf_8_sig')