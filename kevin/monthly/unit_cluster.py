import json
import numpy as np
import pprint
from interpret import detect_peaks
from interpret.parser_articles import parse_aggregated
import nltk
import os
from nltk.stem.porter import PorterStemmer
from whoosh.index import create_in
from whoosh.fields import *
import whoosh.qparser
import pytz

def interpret(type):
    datapath = './monthly/dataset/' + type + '/'
    interpret_datapath = './monthly/interpret/' + type + '/'
    result_file = datapath + 'result.json'
    with open(result_file) as data_file:
        result_data = json.load(data_file)

    all_results = []

    for group in result_data:
        pick_list = []
        pick_months = []
        pick_toparticles = []
        peak_indexes_months = []
        pick_theme = ' '.join(group['theme'])
        print('Interpreter: theme: ', pick_theme)

        for item in group['groups']:
            all_documents_title = []
            all_documents_text = []
            schema = Schema(title=TEXT(stored=True), path=ID(stored=True))
            os.makedirs(interpret_datapath + "indexes", exist_ok=True)
            ix = create_in(interpret_datapath + "indexes", schema)
            writer = ix.writer()

            for article in item['articles']:
                writer.add_document(title=article['title'], path=article['_id'])

            writer.commit()

            with ix.searcher() as searcher:
                query = whoosh.qparser.QueryParser("title", ix.schema, group=whoosh.qparser.OrGroup).parse(pick_theme)
                results = searcher.search(query, limit=50)

                if len(results):
                    article_pick = results
                    print('Clustering: index length:')
                    print(len(article_pick))

                    for a in article_pick:
                        all_documents_title.append(a['title'])
                        if type != 'trumpsaid':
                            all_documents_text.append(a['content'])

            pick_list.append(len(item['articles']))
            pick_months.append(item['month'])
            print(pick_list, pick_months, len(all_documents_title))
            if type == 'trumpsaid':
                all_documents_dict = all_documents_title
            else:
                all_documents_dict = dict(zip(all_documents_title, all_documents_text))

            if len(all_documents_dict) <= 12:
                pick_toparticles.append([])
                print('@@@@@@@@ ERRRR')
                continue
            else:
                if type == 'today' or type == 'trumpsaid':
                    parsed_data = parse_aggregated(all_documents_dict, 2, 13, interpret_datapath, type)
                else:
                    parsed_data = parse_aggregated(all_documents_dict, 2, 13, interpret_datapath)

            print('+++++++ PARSING')
            parsed_data_themes = []
            for index, parseditem in enumerate(parsed_data[0]):
                converted_np = np.array(parsed_data[0][index]).tolist()
                tokens = []
                for token in converted_np:
                    tokens.append(token[0])

                parsed_data_themes.append(tokens)

            #print('Interpreter: all key themes:')
            #print(parsed_data_themes)

            stemmer = PorterStemmer()

            def stem_words(words_list, stemmer):
                return [stemmer.stem(word) for word in words_list]

            for index, themeitem in enumerate(parsed_data_themes):
                zip_data = {}
                theme_data = []

                for token in themeitem:
                    title_tokens = nltk.word_tokenize(' '.join(parsed_data[1][index]))
                    title_stems = stem_words(title_tokens, stemmer)
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

            print(theme_data)

            noduplicates = []
            for theme in theme_data:
                if theme not in group['theme']:
                    noduplicates.append(theme)

            if len(noduplicates) <= 0:
                querytext = ' '.join(theme_data)
            else:
                querytext = ' '.join(noduplicates)

            article_pick = []
            toparticles_matched = []
            with ix.searcher() as searcher:
                query = whoosh.qparser.QueryParser("title", ix.schema, group=whoosh.qparser.OrGroup).parse(querytext)
                results = searcher.search(query)

                if len(results):
                    if type == 'today':
                        article_pick = results[0:4]
                    else:
                        article_pick = results[0:8]
                    print(article_pick)

                    for a in article_pick:
                        for article in item['articles']:
                            if a['path'] == article['_id']:
                                if 'text' in article:
                                    del article['text']
                                toparticles_matched.append(article)


            pick_toparticles.append(toparticles_matched)
            print(toparticles_matched)
            print('########## DONE: {0}'.format(item['month']))

        print('Interpreter: Detect peaks with minimum height and distance filters.')
        peak_indexes = detect_peaks.detect_peaks(pick_list, mph=7, mpd=2).tolist()
        print(dict(zip(pick_months, pick_list)))

        print('Interpreter: Peaks are: %s' % peak_indexes)
        for item in peak_indexes:
            print('Interpreter: Peaks are: %s' % pick_months[int(item)])
            peak_indexes_months.append(pick_months[int(item)])

        result_pick_data = {}
        result_pick_data['theme'] = group['theme']
        result_pick_data['list'] = pick_list
        result_pick_data['months'] = pick_months
        result_pick_data['toparticles'] = pick_toparticles
        result_pick_data['peaks'] = peak_indexes_months

        then = datetime.datetime.now(pytz.utc)
        timeest = str(then.astimezone(pytz.timezone('US/Eastern')))
        result_pick_data['timestamp'] = timeest

        print(result_pick_data)
        all_results.append(result_pick_data)
        print('--------- CLUSTER DONE: {0}'.format(group['theme']))

    if type == 'today':
        interpret_datapath_today = '../dataset/' + type + '_data.json'
        with open(interpret_datapath_today, 'w') as f:
            json.dump(all_results, f, indent=4, sort_keys=True)
    else:
        with open(interpret_datapath + 'result.json', 'w') as f:
            json.dump(all_results, f, indent=4, sort_keys=True)

