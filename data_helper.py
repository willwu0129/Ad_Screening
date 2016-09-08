# coding=utf-8
import jieba
import os
import re
import numpy as np

#
# This part does the pre-processing of the existing articles. The coding for obtaining articles from web is not included

def art_seg_new(category):
    articles = []
    art_size = 0

    with open(os.path.join(os.getcwd(), 'training', category, 'trim_art.txt')) as f:
        count = 1
        article = []
        art_size_temp = 0

        for line in f:
            if count == 1:
                count += 1
                continue

            line = string_pre_process(line)
            if line != ''.join(['------------------------', str(count), '------------------------']) \
                    and line != ''.join(['-end-']):

                #print(line)
                if line == '':
                    continue
                seg_res = list(jieba.cut(line))
                art_size_temp += len(seg_res)
                article.append(" ".join(seg_res))

            else:
                print("art %d" % count)
                article = " ".join(article)
                articles.append(article)
                if art_size_temp > art_size:
                    art_size = art_size_temp


                art_size_temp = 0
                article = []
                count += 1

    with open(os.path.join(os.getcwd(), 'Stop_words.txt')) as f:
        stoplist = set([line.strip().decode('utf8') for line in f])
    #remove common words and tokenize
    texts = [[word for word in art.split() if word not in stoplist] for art in articles]
    articles = [' '.join(art) for art in texts]

    with open(os.path.join(os.getcwd(), 'training', category, 'trim_anno.txt')) as f:
        labels = []
        line = f.readline()
        line = string_pre_process(line)
        while line != '-end-':
            if line == '是':
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            line = f.readline()
            line = string_pre_process(line)
        labels = np.array(labels)
    return articles, labels, art_size



def art_seg(category):

    articles = []
    art_size = 0
    line_size = 0
    with open(os.path.join(os.getcwd(), 'training', category, 'exp_art.txt')) as f:
        count = 1
        article = []

        for line in f:
            if count == 1:
                count += 1
                continue

            line = string_pre_process(line)
            if line != ''.join(['------------------------', str(count), '------------------------']) \
                    and line != ''.join(['-end-']):

                print(line)
                if line == '':
                    continue
                seg_res = list(jieba.cut(line))
                sentences = sent_sep(seg_res)
                for sent in sentences:
                    article.append(" ".join(sent))
                    if len(sent) > line_size:
                        line_size = len(sent)
            else:
                print("art %d" % count)
                articles.append(article)
                if len(article) > art_size:
                    art_size = len(article)
                article = []
                count += 1

    with open(os.path.join(os.getcwd(), 'training', category, 'exp_anno.txt')) as f:
        labels = []
        line = f.readline()
        line = string_pre_process(line)
        while line != '-end-':
            if line == '是':
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            line = f.readline()
            line = string_pre_process(line)
        labels = np.array(labels)
    return articles, labels, art_size, line_size


def string_pre_process(string):
    string = re.sub(r"\xE3\x80\x80", '', string)
    string = string.strip('\r\n')
    string = string.strip('\n')
    return string


def sent_sep(afterseg):
    start = 0
    i = 0
    sents = []
    token = ''
    punt_list = '''.!?;~。！？’；”～'''.decode('utf8')
    for word in afterseg:
        if word in punt_list and token not in punt_list:  # 检查标点符号下一个字符是否还是标点

            sents.append(afterseg[start:i + 1])
            start = i + 1
            i += 1
        else:
            i += 1
            print(afterseg[start:i + 2])
            token = list(afterseg[start:i + 2]).pop()  # 取下一个字符
    if start < len(afterseg):
        sents.append(afterseg[start:])
    return sents


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    art_seg('K')
    print(os.getcwd())

