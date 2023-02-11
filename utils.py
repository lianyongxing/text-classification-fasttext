# -*- coding: utf-8 -*-
# @Time    : 2/11/23 10:52 AM
# @Author  : LIANYONGXING
# @FileName: utils.py
from text_process import sentence_filter
import pandas as pd
from tqdm import tqdm

def load_data(filepath):
    datas = pd.read_csv(filepath)
    datas = datas[~datas['content'].isna()]
    datas['content_filter'] = datas['content'].apply(lambda x: sentence_filter(x))
    return datas


def make_fasttext_data_format(sentences, labels, filename):
    train_datas = []
    for i in tqdm(range(len(sentences))):
        try:
            train_data = "__label__" + str(labels[i]) + "\t" + " ".join([i for i in sentences[i]])
        except Exception as e:
            print(sentences[i])
            continue
        train_datas.append(train_data)

    with open(filename, 'w') as f:
        f.writelines([i+'\n' for i in train_datas])
    return train_datas