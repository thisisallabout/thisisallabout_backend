from __future__ import division
import matplotlib
matplotlib.use('Agg')

import nltk
from nltk.corpus import stopwords
import re
import os
from time import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans
from nltk.stem.porter import PorterStemmer
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def vectorize_cluster(dataset, rangeMin=2, rangeMax=21, tfidfpath='./dataset/', type=None, info=None):
    stemmer = PorterStemmer()

    def stem_words(words_list, stemmer):
        return [stemmer.stem(word) for word in words_list]

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_words(tokens, stemmer)
        return stems

    cachedStopWords = set(stopwords.words("english"))
    cachedStopWords.update(['great', 'MAGA', 'America', 'make', 'American', '...', 'Trump'])

    print('TFIDF_KMEANS: Extracting features from the dataset')
    t0 = time()
    n_features = 50000
    hasher = HashingVectorizer(n_features=n_features, stop_words=cachedStopWords, non_negative=True, norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())

    if type == 'titleonly':
        X = vectorizer.fit_transform(dataset)
    else:
        X = vectorizer.fit_transform(dataset.values())

    print("TFIDF_KMEANS: Extraction done in %fs" % (time() - t0))
    print(" n_samples: %d, n_features: %d" % X.shape)
    print()

    print("TFIDF_KMEANS: LSA Dimension reduction")
    n_components = 100
    t0 = time()
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)
    print("TFIDF_KMEANS: Reduction done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("TFIDF_KMEANS: Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    print()

    verbose = False
    print("TFIDF_KMEANS: Finding the best n_clusters value by running MinibatchKmeans...")
    range_n_clusters = range(rangeMin, rangeMax)
    range_n_clusters_km = []
    for range_k in range_n_clusters:
        km = MiniBatchKMeans(n_clusters=range_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=verbose)

        print(" Finding the best n_clusters - Clustering sparse data with %s" % km)
        t0 = time()
        km.fit(X)
        print(" Finding the best n_clusters - sparse data done in %0.3fs" % (time() - t0))
        print(' Finding the best n_clusters: {0}'.format(range_k))
        print(" Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))
        range_n_clusters_km.append(metrics.silhouette_score(X, km.labels_, sample_size=1000))
        print()

    range_n_clusters_km_index_max = max(range(len(range_n_clusters_km)), key=range_n_clusters_km.__getitem__)
    km_optimal = range_n_clusters[range_n_clusters_km_index_max]
    print('TFIDF_KMEANS: Found optimal n_clusters: {0}'.format(km_optimal))

    km_final = MiniBatchKMeans(n_clusters=km_optimal, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=verbose)

    print("TFIDF_KMEANS: Clustering sparse data with %s" % km_final)
    t0 = time()
    km_final.fit(X)
    print(" done in %0.3fs" % (time() - t0))
    print(" Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km_final.labels_, sample_size=1000))
    print()

    cluster_dict_store = {}

    for i in set(km_final.labels_):
        if type == 'titleonly':
            proc_cluster_it = [list(dataset)[x] for x in np.where(km_final.labels_ == i)[0]]
        else:
            proc_cluster_it = [list(dataset.keys())[x] for x in np.where(km_final.labels_ == i)[0]]
        cluster_dict_store[i] = proc_cluster_it

    def clean_text(raw_text):
        letters_only = re.sub('[^a-zA-Z]', ' ', str(raw_text))
        words = letters_only.lower().split()
        useful_words = [x for x in words if x not in cachedStopWords]

        useful_words_string = ' '.join(useful_words)
        return useful_words_string

    cluster_themes_dict = {}
    for key in cluster_dict_store.keys():
        current_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=cachedStopWords)
        current_tfs = current_tfidf.fit_transform(map(clean_text, cluster_dict_store[key]))

        current_tf_idfs = dict(zip(current_tfidf.get_feature_names(), current_tfidf.idf_))
        tf_idfs_tuples = current_tf_idfs.items()
        cluster_themes_dict[key] = sorted(tf_idfs_tuples, key=lambda x: x[1])[:7]
    # print('Random Cluster {0} key words: {1}'.format(cluster_pick, [x[0] for x in cluster_themes_dict[cluster_pick]]))

    return cluster_themes_dict, cluster_dict_store