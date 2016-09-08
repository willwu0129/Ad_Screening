# coding=utf-8
from gensim import corpora
import os
import data_helper as dh
import logging
from collections import defaultdict
from pprint import pprint  # pretty-printer
from gensim import corpora, models, similarities
import numpy as np

#
# This function is used to vectorize the articles.
# It first does pre-processing: (1) deleting words appearing in the stop word list; (2) deleting words only appearing once
# Then the work flow of the process is: forming the dictionary -> forming the corpus from dictionary -> forming tfidf corpus from corpus -> forming lsi corpus from tfidf


def vectorizing(category):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    articles, y, art_size, line_size = dh.art_seg(category)
    documents = []
    for article in articles:
        document = ' '.join((l for l in article))
        documents.append(document)

    with open(os.path.join(os.getcwd(), 'Stop_words.txt')) as f:
        stoplist = set([line.strip().decode('utf8') for line in f])
    #remove common words and tokenize
    texts = [[word for word in document.split() if word not in stoplist] for document in documents]
    #remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]


    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
    print(dictionary)
    print(dictionary.token2id)

    corpus = [dictionary.doc2bow(text) for text in texts]
    #corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
    #print(corpus)


    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
    corpus_lsi = lsi[corpus_tfidf]
    lsi.print_topics(50)
    x = []
    for item in corpus_lsi:
        x.append(np.array([sub_item[1] for sub_item in item]))
    x = np.array(x)
    return x, y


# index = similarities.MatrixSimilarity(lsi[corpus])
# sims = index[vec_lsi]
# print(list(enumerate(sims)))
#
#
# sims = sorted(enumerate(sims), key=lambda item: -item[1])
# print(sims)

#lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1, chunksize=10000, passes=1)
#lda.print_topics(20)


