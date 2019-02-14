from aggregate.process_twitter import aggregator
from pymongo import MongoClient
import datetime
from datetime import timezone
import sys
import re

def aggregate_db(item):
    class Transcript(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.logfile = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.logfile.write(message)

        def flush(self):
            pass

    sys.stdout = Transcript('./logs/aggregate_' + item + '.txt')

    print('MASTER AGGREGATE UNIT: Initiating aggregator unit...')
    print('MASTER AGGREGATE UNIT: Setting up MongoClient kevin@main')
    client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
    db = client['main']

    twitter_list_target = 'usasociety'
    if item == 'usasociety':
        twitter_list_target = 'usa-society1'
    elif item == 'theyear':
        twitter_list_target = 'usa-publicfunded'
    elif item == 'trumpsaid':
        twitter_list_target = 'usa-trump'

    collection = db['aggregator_' + item]
    if len(list(collection.find().sort('ts', -1).limit(1))):
        cursor = list(collection.find().sort('ts', -1).limit(1))
        print(cursor)
    else:
        cursor = []

    if len(cursor):
        last_id = cursor[0]['twitterid']
        print('MASTER AGGREGATE UNIT: Found latest twitter id from the collection: {0} {1}'.format(last_id, twitter_list_target))
        origin_data = aggregator(twitter_list_target, last_id)
    else:
        print('MASTER AGGREGATE UNIT: The collection is empty, fetching the last 50 tweets available...')
        origin_data = aggregator(twitter_list_target)

    if len(origin_data[1]):
        print(origin_data[1])
        result = collection.insert_many(origin_data[1])
        print('MASTER AGGREGATE UNIT: DB Insert result: {0}'.format(result))
        print('MASTER AGGREGATE UNIT: DB Insert result ids: {0}'.format(result.inserted_ids))
    else:
        print('MASTER AGGREGATE UNIT: Empty result returned from aggregator; finished.')

    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
