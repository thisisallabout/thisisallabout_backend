from pymongo import MongoClient
from tfidf_kmeans import vectorize_cluster
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
import pandas as pd

print('Initiating cluster unit...')
print('Clustering unit: Setting up MongoClient kevin@main')
client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
db = client['main']

theme_blacklists = ['periscope', 'pbs', 'newshour', 'npr', 'watch']

def cluster_ner(text):
    # entity recognition
    print("Clustering unit: Extracting named-entities")

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


def sort_articles_month(item, type=None, mode=None):
    collection = db['aggregator_' + item]

    print('Clustering unit: Start loading articles from database: {0} mode'.format(mode))
    if mode is None:
        last_hour_date_time = datetime.now() - timedelta(hours=24)
        unix_time = last_hour_date_time.strftime("%s")
        print(unix_time)

        cursor = list(collection.find({"ts": {"$gt": int(unix_time) }}).sort([('_id', 1)]))
    elif mode == 90000009:
        # cursor = list(collection.find({}).sort([('_id', 1)]).limit(3000))
        cursor = list(collection.find({}).sort([('_id', 1)]))
        print(len(cursor))
    else:
        last_hour_date_time = datetime.now() - timedelta(hours=mode)
        unix_time = last_hour_date_time.strftime("%s")
        print(unix_time)

        cursor = list(collection.find({"ts": {"$gt": int(unix_time)}}).sort([('_id', 1)]))
        print(len(cursor))

    if len(cursor):
        print('Clustering unit: Processing loaded articles...')
        parsed_articlecluster_packed = []


        us_eastern_time = pytz.timezone('US/Eastern')
        df = pd.DataFrame(cursor)
        df['date'] = pd.to_datetime(df['ts'],unit='s')
        df['date_local'] = df.apply(lambda x: x['date'].tz_localize('UTC').tz_convert('America/New_York'), axis=1)
        grouped_df = df.groupby(pd.Grouper(key='date_local', freq='60D'))
        # print(grouped_df.apply(lambda x: x['date_local'].set_index()))

        for name, group in grouped_df:
            article_list = []
            print('ID: ' + str(name))
            tsobj = group.ts.tolist()
            for index, tsitem in enumerate(tsobj):
                article_item = {}
                article_item['twitterid'] = group.twitterid.tolist()[index]
                article_item['ts'] = tsitem
                article_item['title'] = group.title.tolist()[index]
                article_item['origin'] = group.origin.tolist()[index]
                article_list.append(article_item)

            group_articles = {}
            group_articles['month'] = str(name)
            group_articles['articles'] = article_list
            parsed_articlecluster_packed.append(group_articles)
            print('Clustering unit: total {0} articles in month {1}...'.format(str(name), len(article_list)))


        # print('Raw article data reference: ')
        # print(parsed_articlecluster_packed)

        print('Clustering unit: finished processing')

        tfidfpath = './monthly/dataset/' + type + '/'
        os.makedirs(tfidfpath, exist_ok=True)
        with open(tfidfpath + '/result.json', 'w') as outfile:
            json.dump(parsed_articlecluster_packed, outfile, indent=4, sort_keys=True)

        print('Clustering unit: saved into json file.')

        ########### Clustering
        parsed_articlecluster_final = []
        for monthlyitem in parsed_articlecluster_packed:
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


            for article in monthlyitem['articles']:
                if isinstance(article['title'], str):
                    parsed_article_title.append(article['title'])

            parsed_article_dict = parsed_article_title
            print('Clustering unit: Finished processing loaded articles')

            datapath = './monthly/cluster/' + type + '/'
            os.makedirs(datapath, exist_ok=True)

            if type == 'trumpsaid':
                parsed_data = vectorize_cluster(parsed_article_dict, 1, 25, datapath, 'titleonly', monthlyitem['month'])

            # print(parsed_data[0])
            parsed_data_themes = []
            for index, item in enumerate(parsed_data[0]):
                converted_np = np.array(parsed_data[0][index]).tolist()
                tokens = []
                for token in converted_np:
                    tokens.append(token[0])

                parsed_data_themes.append(tokens)

            parsed_articlecluster_flattened = [val for sublist in parsed_data[1].values() for val in sublist]
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

                theme_data_l = [x.lower() for x in theme_data]
                # if any(x in theme_data_l for x in theme_blacklists):
                # print('Clustring unit: there is one or more blacklisted words in themes')
                # else:
                ner_result = list(cluster_ner(' '.join(parsed_data[1][index])))

                article_list = []
                for title in parsed_data[1][index]:
                    filter = next((item for item in cursor if item["title"] == title), False)
                    if filter:
                        if 'text' in filter:
                            del filter['text']
                        if 'authors' in filter:
                            del filter['authors']
                        if '_id' in filter:
                            filter['_id'] = str(filter['_id'])

                        article_list.append(filter)

                zip_data['theme'] = theme_data
                zip_data['namedentity'] = ner_result
                zip_data['articles'] = article_list
                zip_data['month'] = monthlyitem['month']

                parsed_articlecluster.append(zip_data)

                print('Clustering unit: finished processing')
                with open(datapath + '/' + monthlyitem['month'] + '.json', 'w') as outfile:
                    json.dump(parsed_articlecluster, outfile, indent=4, sort_keys=True)

                print('Clustering unit: saved into json file.')

            parsed_articlecluster_final.append(parsed_articlecluster)

        print(parsed_articlecluster_final)

        # print('Raw article data reference: ')
        # print(parsed_articlecluster)


    else:
        print('Clustering unit: The collection is empty, unable to process.')

