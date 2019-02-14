from pymongo import MongoClient
from processes.daily.tfidf_kmeans import vectorize_cluster
from datetime import datetime, timedelta
import numpy as np
import nltk
import re
from nltk.stem.porter import PorterStemmer
import itertools
from operator import itemgetter
import pytz
import json
import os

theme_blacklists = ['periscope', 'pbs', 'newshour', 'npr', 'watch']

def cluster_ner(text):
    sentences = nltk.sent_tokenize(''.join(text))
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

    def extract_entity_names(t):
        entity_names = []

        if hasattr(t, 'label') and t.label:
            if t.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in t]))
            else:
                for child in t:
                    entity_names.extend(extract_entity_names(child))

        return entity_names

    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))

    for name in entity_names:
        if len(re.compile('\W').split(name)) <= 1:
            check = set(list(filter(lambda x: name in x, entity_names)))
            check.remove(name)
            if len(check) == 1:
                entity_names.remove(name)
            elif len(check) >= 2:
                entity_names.remove(name)
                entity_names.extend(check)

    return set(entity_names)


def stem_words(words_list):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words_list]


def cluster_articles(item, type=None, mode=None):
    print('* PROCESS Initiated /type=daily /init=cluster')
    print('HEAD_CLUSTER: Setting up MongoClient kevin@main \n')

    client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
    db = client['main']
    collection = db['aggregator_' + item]

    datastore_path = './processes/daily/store/'

    print('HEAD_CLUSTER: Start loading articles from database: {0} mode'.format(mode))
    last_hour_date_time = datetime.now() - timedelta(hours=24)
    unix_time = last_hour_date_time.strftime("%s")
    print('HEAD_CLUSTER: Timestamp of the last item: {0}'.format(unix_time))

    cursor = list(collection.find({"ts": {"$gt": int(unix_time)}}).sort([('_id', 1)]))
    print('HEAD_CLUSTER: Total length of the items fetched: {0}'.format(len(cursor)))

    if len(cursor):
        print('\nHEAD_CLUSTER: Processing loaded articles...')
        parsed_article_title = []
        parsed_article_text = []

        for article in cursor:
            if isinstance(article['title'], str):
                if isinstance(article['text'], str):
                    parsed_article_title.append(article['title'])
                    parsed_article_text.append(article['text'])

        parsed_article_dict = dict(zip(parsed_article_title, parsed_article_text))
        print('HEAD_CLUSTER: Finished processing loaded articles, passing it to TFIDF_KMEANS \n')

        os.makedirs(datastore_path, exist_ok=True)
        tfidfpath = datastore_path
        parsed_data = vectorize_cluster(parsed_article_dict, 2, 15, tfidfpath, 'today')
        origin_data_raw = cursor

        # print(parsed_data[0])
        parsed_data_themes = []
        for index, item in enumerate(parsed_data[0]):
            converted_np = np.array(parsed_data[0][index]).tolist()
            tokens = []
            for token in converted_np:
                tokens.append(token[0])

            parsed_data_themes.append(tokens)

        parsed_articlecluster = []

        for index, item in enumerate(parsed_data_themes):
            zip_data = {}
            theme_data = []

            for token in item:
                title_tokens = nltk.word_tokenize(' '.join(parsed_data[1][index]))
                title_stems = stem_words(title_tokens)
                title_token_zip = dict(zip(title_stems, title_tokens))

                matched_token = title_token_zip.get(token)
                # print('Target token: {0} / Matched: {1} / Tokens Dict: {2}'.format(token, matched_token, title_token_zip))
                if isinstance(matched_token, str):
                    if len(matched_token) > 1:
                        theme_data.append(matched_token)
                elif isinstance(matched_token, list):
                    if len(matched_token[0]) > 1:
                        theme_data.append(matched_token[0])
                else:
                    if len(token) > 1:
                        theme_data.append(token)

            # theme_data_l = [x.lower() for x in theme_data]
            #if any(x in theme_data_l for x in theme_blacklists):
                #print('Clustring unit: there is one or more blacklisted words in themes')
            #else:
            print("HEAD_CLUSTER: Extracting named-entities for cluster {0}".format(index))
            ner_result = list(cluster_ner(' '.join(parsed_data[1][index])))

            zip_data['theme'] = theme_data
            zip_data['namedentity'] = ner_result
            zip_data['articles'] = parsed_data[1][index]

            parsed_articlecluster.append(zip_data)

        parsed_articlecluster_packed = []
        us_eastern_time = pytz.timezone('US/Eastern')
        for cluster in parsed_articlecluster:
            cluster_data = {}
            cluster_data['theme'] = cluster['theme']
            cluster_data['namedentity'] = cluster['namedentity']
            article_list = []

            for title in cluster['articles']:
                filter = next((item for item in origin_data_raw if item["title"] == title), False)
                if filter:
                    # if 'text' in filter:
                        # del filter['text']
                    if 'authors' in filter:
                        del filter['authors']
                    if '_id' in filter:
                        filter['_id'] = str(filter['_id'])

                    timeformat = '%Y-%m' # mode <= 48

                    filter['time_filter'] = datetime.fromtimestamp(filter['ts'], us_eastern_time).strftime(timeformat)
                    article_list.append(filter)

            cluster_data['groups'] = []
            sorted_articles = sorted(article_list, key=itemgetter('time_filter'))
            for key, group in itertools.groupby(sorted_articles, key=lambda x: x['time_filter']):
                group_articles = {}
                group_articles['time_filterby'] = key
                group_articles['articles'] = list(group)
                cluster_data['groups'].append(group_articles)

            parsed_articlecluster_packed.append(cluster_data)

        print('HEAD_CLUSTER: Finished initial processing')
        with open(tfidfpath + '/result.json', 'w') as outfile:
            json.dump(parsed_articlecluster_packed, outfile, indent=4, sort_keys=True)

        print('HEAD_CLUSTER: Saved into json file in {0}'.format(datastore_path))

        from processes.daily.cluster_postproc import postprocess
        postprocess(type)
    else:
        print('HEAD_CLUSTER: The collection is empty, unable to process.')

