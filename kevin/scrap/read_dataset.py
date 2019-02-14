import pickle
from tfidf_kmeans import parse_aggregated
import numpy as np
import json
import jsonpickle
from pymongo import MongoClient

print('Initiating dataset reader...')
with open('./dataset/dataset_npr_db.pkl', 'rb') as handle:
    unserialized_data = pickle.load(handle)

print('Fetched pickled dataset: total length: {0}'.format(len(unserialized_data)))
# print(unserialized_data)

print('Initiating aggregator unit...')
print('Setting up MongoClient kevin@main')
client = MongoClient('mongodb://kevin:eHAdpMJze8XubCUWGXo@23.239.14.16:27017/main')
db = client['main']

def aggregator_storeobject():
    # ccresult = db.create_collection('aggregator_trumptimemachine')
    # print('Collection creation result: {0}'.format(ccresult))
    collection = db['aggregator_trumptimemachine']

    json_converted = jsonpickle.encode(unserialized_data)

    print('Saving data into mongodb...')
    result = collection.insert_many(json.loads(json_converted))
    print('DB Insert result: {0}'.format(result))
    print('DB Insert result ids: {0}'.format(result.inserted_ids))


aggregator_storeobject()



def cluster_articles():
    if len(unserialized_data):
        print('Processing loaded articles for clustering...')
        parsed_article_title = []
        parsed_article_text = []

        for item in unserialized_data:
            parsed_article_title.append(item['title'])
            parsed_article_text.append(item['text'])

        parsed_article_dict = dict(zip(parsed_article_title, parsed_article_text))
        print('Finished processing loaded articles for clustering')

        parsed_data = parse_aggregated(parsed_article_dict, 5)
        origin_data_raw = unserialized_data

        print(parsed_data[0])
        # print(parsed_data[1])

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
            zip_data['theme'] = item
            zip_data['articles'] = parsed_data[1][index]

            parsed_articlecluster.append(zip_data)


        print(parsed_articlecluster)

        parsed_articlecluster_flattened = [val for sublist in parsed_data[1].values() for val in sublist]
        parsed_articlecluster_raw = []
        for title in parsed_articlecluster_flattened:
            filter = next((item for item in origin_data_raw if item["title"] == title), False)
            if filter:
                parsed_articlecluster_raw.append(filter)
    else:
        print('The collection is empty, unable to cluster.')


#cluster_articles()