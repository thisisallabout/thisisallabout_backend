from pymongo import MongoClient
from cluster.tfidf_kmeans import vectorize_cluster
from datetime import datetime, timedelta
import numpy as np
import nltk
import re
from nltk.stem.porter import PorterStemmer
import itertools
from operator import itemgetter
import pytz
import json

print('Initiating cluster unit...')
print('MASTER CLUSTER UNIT: Setting up MongoClient kevin@main')
client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
db = client['main']

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
    collection = db['aggregator_' + item]

    print('MASTER CLUSTER UNIT: Start loading articles from database: {0} mode'.format(mode))
    if mode is None:
        last_hour_date_time = datetime.now() - timedelta(hours=24)
        unix_time = last_hour_date_time.strftime("%s")
        print(unix_time)

        cursor = list(collection.find({"ts": {"$gt": int(unix_time) }}).sort([('_id', 1)]))
    elif mode == 90000009:
        # cursor = list(collection.find({}).sort([('_id', 1)]).limit(3000))
        cursor = list(collection.find({}).sort([('_id', 1)]))
        print(len(cursor))
    elif mode == 90000001:
        # cursor = list(collection.find({}).sort([('_id', 1)]).limit(3000))
        cursor = list(collection.find({"ts": {"$gt": int(1489554000), "$lt": int(1521089999)}}).sort([('_id', 1)]))
        print(len(cursor))
    else:
        last_hour_date_time = datetime.now() - timedelta(hours=mode)
        unix_time = last_hour_date_time.strftime("%s")
        print(unix_time)

        cursor = list(collection.find({"ts": {"$gt": int(unix_time)}}).sort([('_id', 1)]))
        print(len(cursor))

    if len(cursor):
        print('MASTER CLUSTER UNIT: Processing loaded articles...')
        parsed_article_title = []
        parsed_article_text = []

        def contains(small, big):
            for i in range(len(big) - len(small) + 1):
                for j in range(len(small)):
                    if big[i + j] != small[j]:
                        break
                else:
                    return i, i + len(small)
            return False

        if type == 'trumpsaid':
            for article in cursor:
                if isinstance(article['title'], str):
                    parsed_article_title.append(article['title'])
        else:
            for article in cursor:
                if isinstance(article['title'], str):
                    if isinstance(article['text'], str):
                        parsed_article_title.append(article['title'])
                        parsed_article_text.append(article['text'])

        if type == 'trumpsaid':
            parsed_article_dict = parsed_article_title
        else:
            parsed_article_dict = dict(zip(parsed_article_title, parsed_article_text))
        print('MASTER CLUSTER UNIT: Finished processing loaded articles')

        tfidfpath = './dataset/' + type + '/'
        if type == 'today':
            parsed_data = vectorize_cluster(parsed_article_dict, 2, 15, tfidfpath, 'today')
        elif type == 'trumpsaid':
            parsed_data = vectorize_cluster(parsed_article_dict, 2, 25, tfidfpath, 'trumpsaid')
        else:
            parsed_data = vectorize_cluster(parsed_article_dict, 2, 10, tfidfpath)
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
            print("MASTER CLUSTER UNIT: Extracting named-entities")
            ner_result = list(cluster_ner(' '.join(parsed_data[1][index])))

            zip_data['theme'] = theme_data
            zip_data['namedentity'] = ner_result
            zip_data['articles'] = parsed_data[1][index]

            parsed_articlecluster.append(zip_data)

        # print(parsed_articlecluster)

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

                    if type == 'today':
                        timeformat = '%Y-%m' # mode <= 48

                        filter['time_filter'] = datetime.fromtimestamp(filter['ts'], us_eastern_time).strftime(timeformat)
                        article_list.append(filter)
                    else:
                        timeformat = ''
                        if mode <= 72:
                            timeformat = '%Y-%m'
                        elif mode <= 168:
                            timeformat = '%Y-%m-%d'
                        elif mode == 90000009:
                            timeformat = '%Y-%m'
                        elif mode == 90000001:
                            timeformat = '%Y-%m'

                        filter['time_filter'] = datetime.fromtimestamp(filter['ts'], us_eastern_time).strftime(timeformat)
                        article_list.append(filter)

            cluster_data['groups'] = []
            sorted_articles = sorted(article_list, key=itemgetter('time_filter'))
            for key, group in itertools.groupby(sorted_articles, key=lambda x: x['time_filter']):
                group_articles = {}
                group_articles['time_filterby'] = key
                group_articles['articles'] = list(group)
                cluster_data['groups'].append(group_articles)
                print('MASTER CLUSTER UNIT: total {0} articles in month {1}...'.format(key, len(list(group))))

            parsed_articlecluster_packed.append(cluster_data)

        # print('Raw article data reference: '), parsed_articlecluster_packed

        print('MASTER CLUSTER UNIT: finished processing')
        with open(tfidfpath + '/result.json', 'w') as outfile:
            json.dump(parsed_articlecluster_packed, outfile, indent=4, sort_keys=True)

        print('MASTER CLUSTER UNIT: saved into json file.')

        if type == 'today':
            from cluster.cluster_postproc import postprocess
            postprocess(type)
    else:
        print('MASTER CLUSTER UNIT: The collection is empty, unable to process.')

