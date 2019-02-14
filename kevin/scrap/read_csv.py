import csv
import re
import pickle
from aggregator_twitter import aggregator
from unit_cluster import cluster_articles_offline
import json
import jsonpickle
import time
import datetime
import dateutil.parser
from pymongo import MongoClient

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def read_parse_csv(source_origin, type):
    print('Reading fetched csv data...')
    filename = '../scrap/output_' + source_origin + '.csv'
    reader = csv.DictReader(open(filename), delimiter=';')
    csv_tweets = []

    for line in reader:
        url_string = re.findall(r'(https?://[^\s]+)', line['text'])
        if type != '':
            line_converted = dict(line)
            data = {}

            data['twitterid'] = line_converted['id']
            data['origin'] = source_origin
            if url_string:
                url_string_filtered = re.sub('\\xa0$', '', url_string[0])
                data['url'] = url_string_filtered

            removeurl = re.sub(r"http\S+", "", line_converted['text'])
            removetwturl = re.sub(r"pic.twitter\S+", "", removeurl)

            data['title'] = removetwturl
            dt = dateutil.parser.parse(line_converted['date'])
            unixts = int(time.mktime(dt.timetuple()))
            data['ts'] = unixts

            csv_tweets.append(data)
        else:
            if url_string:
                url_string_filtered = re.sub('\\xa0$', '', url_string[0])
                if 'twitter.com/' in url_string_filtered:
                    print('Skips item: social channel')
                else:
                    line_converted = dict(line)
                    data = {}

                    data['twitterid'] = line_converted['id']
                    data['origin'] = source_origin
                    data['url'] = url_string_filtered
                    data['ts'] = line_converted['date']

                    url_duplicate_check = []
                    for item in csv_tweets:
                        if item['url'] == url_string_filtered:
                            url_duplicate_check.append(item)

                    if len(url_duplicate_check) > 0:
                        print('Fetch process: there seems one or more duplicated url in the dataset: ')
                        print(url_duplicate_check)
                    else:
                        csv_tweets.append(data)


    final_csv_tweets = csv_tweets[::-1]
    print('Successfully fetched csv data, total length: {0}'.format(len(final_csv_tweets)))
    # print(csv_tweets[::-1])


    if type == 'offline_text':
        print('Initializing offline clustering module...')
        cluster_articles_offline(final_csv_tweets, 'trumpsaid', 90000009)
    elif type == 'offline':
        origin_data = aggregator(None, None, 'direct', final_csv_tweets)

        print('Initializing offline clustering module...')
        cluster_articles_offline(origin_data[1], None, 90000009)
    elif type == '':
        origin_data = aggregator(None, None, 'direct', final_csv_tweets)

        try:
            pkl_filename = './scrap/dataset_' + source_origin + '_db.pkl'
            pkl_c_filename = './scrap/dataset_' + source_origin + '_cluster.pkl'
            with open(pkl_filename, 'wb') as f:
                pickle.dump(origin_data[1], f, pickle.HIGHEST_PROTOCOL)
            print('Fetched origin data stored in {0} for mongodb'.format(pkl_filename))

            with open(pkl_c_filename, 'wb') as f:
                pickle.dump(origin_data[0], f, pickle.HIGHEST_PROTOCOL)
                print('Fetched origin data stored in {0} for clustering'.format(pkl_c_filename))
            print(len(origin_data))

            print('Initiating dataset reader...')
            with open(pkl_filename, 'rb') as handle:
                unserialized_data = pickle.load(handle)

            print('Fetched pickled dataset: total length: {0}'.format(len(unserialized_data)))
            # print(unserialized_data)

            print('Initiating aggregator unit...')
            print('Setting up MongoClient kevin@main')
            client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
            db = client['main']

            collection = db['aggregator_trumpsaid']

            json_converted = jsonpickle.encode(unserialized_data)

            print('Saving data into mongodb...')
            result = collection.insert_many(json.loads(json_converted))
            print('DB Insert result: {0}'.format(result))
            print('DB Insert result ids: {0}'.format(result.inserted_ids))
        except Exception as e:
            print('There was a problem saving data into a dataset file: ')
            print(origin_data)
    else:
        try:
            print('Initiating aggregator unit...')
            print('Setting up MongoClient kevin@main')
            client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
            db = client['main']

            collection = db['aggregator_trumpsaid']

            json_converted = jsonpickle.encode(final_csv_tweets)

            print('Saving data into mongodb...')
            result = collection.insert_many(json.loads(json_converted))
            print('DB Insert result: {0}'.format(result))
            print('DB Insert result ids: {0}'.format(result.inserted_ids))
        except Exception as e:
            print('There was a problem saving data into a dataset file: ')
            print(final_csv_tweets)


read_parse_csv('realdonaldtrump', 'save')