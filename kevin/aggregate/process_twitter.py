from __future__ import division
import twitter
from newspaper import Article, Config
from urllib.parse import urlparse
import json
import re
import requests
from nltk.corpus import stopwords
import time
import datetime
import math
import metadata_parser
import dateutil.parser

def aggregator(list=None, tweetid=0, mode=None, data=None):
    config = Config()
    config.follow_meta_refresh = True

    def get_article_info(article, origin=None, id=None, ts=None):
        try:
            article.download()
            article.parse()
            data = {}
            
            blacklist = ['USA TODAY: Latest World and US News']
            if any(word in article.title for word in blacklist):
                print('AGGREGATE_TWITTER: Skiping aggregation: wrong pages')
                return None
            else:
                if id:
                    data['twitterid'] = id
                data['title'] = article.title
                data['url'] = article.url
                if origin:
                    data['origin'] = origin
                if ts:
                    dt = dateutil.parser.parse(ts)
                    unixts = int(time.mktime(dt.timetuple()))
                    data['ts'] = unixts

                if article.text:
                    data['text'] = clean_text(article.text)
                else:
                    data['text'] = ''

                if article.top_image:
                    data['image'] = article.top_image
                if article.authors:
                    data['authors'] = article.authors

            return data
        except Exception as e:
            print("AGGREGATE_TWITTER: Article Parse Error: ", e)

    def get_article_meta(url, origin=None, id=None, ts=None):
        try:
            page = metadata_parser.MetadataParser(url=url)
            data = {}

            blacklist = ['트위터', 'Twitter', 'Instagram', 'Youtube', 'Snapchat']
            if any(word in page.get_metadata('title') for word in blacklist):
                print('AGGREGATE_TWITTER: Skiping aggregation: social channels')
                return None
            else:
                if id:
                    data['twitterid'] = id
                data['title'] = page.get_metadata('title')
                data['url'] = page.get_metadata('url')
                if origin:
                    data['origin'] = origin
                if ts:
                    dt = dateutil.parser.parse(ts)
                    unixts = int(time.mktime(dt.timetuple()))
                    data['ts'] = unixts

                if page.get_metadata('description'):
                    data['text'] = clean_text(page.get_metadata('description'))
                else:
                    data['text'] = ''

                if page.get_metadata('image'):
                    data['image'] = page.get_metadata('image')

            return data
        except Exception as e:
            print("AGGREGATE_TWITTER: Article Parse Error: ", e)

    def get_tweet_data(article, origin=None, id=None, ts=None):
        try:
            data = {}

            if id:
                data['twitterid'] = id
            removeurl = re.sub(r"http\S+", "", article.text)
            removetwturl = re.sub(r"pic.twitter\S+", "", removeurl)

            data['title'] = removetwturl
            if len(article.urls):
                data['url'] = article.urls[0].expanded_url

            if origin:
                data['origin'] = origin
            if ts:
                dt = dateutil.parser.parse(ts)
                unixts = int(time.mktime(dt.timetuple()))
                data['ts'] = unixts

            data['text'] = ''

            return data
        except Exception as e:
            print("AGGREGATE_TWITTER: Tweet Parse Error: ", e)

    def clean_text(raw_text):
        letters_only = re.sub('[^a-zA-Z]', ' ', raw_text)
        words = letters_only.lower().split()
        stopwords_eng = set(stopwords.words("english"))
        useful_words = [x for x in words if x not in stopwords_eng]

        # Combine words into a paragraph again
        useful_words_string = ' '.join(useful_words)
        return useful_words_string

    print('AGGREGATE_TWITTER: Start aggregator in {0} mode'.format(mode))
    print('AGGREGATE_TWITTER: Start parsing articles... This might take a while: ')
    parsed_article_title = []
    parsed_article_text = []
    parsed_article_data = []

    session = requests.Session()  # so connections are recycled
    if mode is None:
        print('AGGREGATE_TWITTER: Authenticating with twitter...')
        api = twitter.Api(consumer_key='Bx3Nj8nBrBRSdJM3bFVJ6HBd8',
                                     consumer_secret='qGmZ5w9Stx9APwoILPVTRP7DnCnAiIjRnNkNxgnUdFyFZU7QUU',
                                     access_token_key='826316996564226050-KA5pVjPBKqgwBxs1S121hZwFaWqPZWZ',
                                     access_token_secret='2l1WWaZQ3XZp6DrRflY10B6cGXDBbRMVaNltXsfKFWMZ0')

        if tweetid != 0:
            print('AGGREGATE_TWITTER: Start retrieving twitter timeline: since_id {0}'.format(tweetid))
            twitter_timeline = api.GetListTimeline(slug=list, owner_screen_name="tiaa_obv000",
                                               include_rts=False, since_id=tweetid, count="1000")
        else:
            print('AGGREGATE_TWITTER: Start retrieving twitter timeline: no since_id')
            twitter_timeline = api.GetListTimeline(slug=list, owner_screen_name="tiaa_obv000",
                                                   include_rts=False, count="500")
        # print(twitter_timeline)
        print(len(twitter_timeline))

        if len(twitter_timeline) <= 0:
            print('AGGREGATE_TWITTER: Twitter data is up to date')
        elif twitter_timeline[0].id == int(tweetid):
            print('AGGREGATE_TWITTER: Twitter data is up to date: {0}'.format(twitter_timeline[0].id))
        else:
            for index, status in enumerate(twitter_timeline):
                if list == 'usa-trump':
                    parse_data = get_tweet_data(status, status.user.name, status.id, status.created_at)

                    if parse_data:
                        print('AGGREGATE_TWITTER: Index {0}/{1}, parsed title: {2}'.format(index, len(twitter_timeline),
                                                                        parse_data['title']))
                        parsed_article_title.append(parse_data['title'])
                        parsed_article_data.append(parse_data)
                else:
                    if len(status.urls):
                        if status.urls[0].url:
                            try:
                                resp = session.head(status.urls[0].url, allow_redirects=True)
                                if 'twitter.com/' not in resp.url:
                                    url_path = urlparse(resp.url).path
                                    if (url_path != '') and (url_path != '/'):
                                        parse_target = Article(resp.url)
                                        parse_data = get_article_info(parse_target, status.user.name, status.id, status.created_at)

                                        if parse_data:
                                            print('AGGREGATE_TWITTER: Index {0}/{1}, parsed title: {2}'.format(index, len(twitter_timeline), parse_data['title']))
                                            parsed_article_title.append(parse_data['title'])
                                            parsed_article_text.append(parse_data['text'])
                                            parsed_article_data.append(parse_data)
                                    else:
                                        print('AGGREGATE_TWITTER: Skipped parsing: the url doesn\'t have any paths')
                                else:
                                    print('AGGREGATE_TWITTER: Skipped parsing: the url contains twitter.com')
                                    print('AGGREGATE_TWITTER: Index {0}/{1}, Skipped parsing: the url contains twitter.com {2}'.format(index, len(twitter_timeline), resp.url))
                            except requests.TooManyRedirects:
                                print('AGGREGATE_TWITTER: Article Parse Error: too many redirects')
                            except requests.RequestException as e:
                                print('AGGREGATE_TWITTER: Article Parse Error: {0}'.format(e))
    elif mode == 'direct':
        for index, item in enumerate(data):
            try:
                print(item['url'])
                resp = session.head(item['url'], allow_redirects=True)
                if 'twitteri.com/' not in resp.url:
                    parse_target = Article(resp.url)
                    parse_data = get_article_info(parse_target, item['origin'], item['twitterid'], item['ts'])

                    if parse_data:
                        print('AGGREGATE_TWITTER: Index {0}/{1}, parsed title: {2}, {3}'.format(index, len(data),
                                                                        parse_data['title'], parse_data['text']))
                        parsed_article_title.append(parse_data['title'])
                        parsed_article_text.append(parse_data['text'])
                        parsed_article_data.append(parse_data)
                else:
                    print('AGGREGATE_TWITTER: Skipped parsing: the url contains twitter.com')
                    print('AGGREGATE_TWITTER: Index {0}/{1}, Skipped parsing: the url contains twitter.com {2}'.format(index, len(
                        data), resp.url))
            except requests.TooManyRedirects:
                print('AGGREGATE_TWITTER: Article Parse Error: too many redirects')
            except requests.RequestException as e:
                print('AGGREGATE_TWITTER: Article Parse Error: {0}'.format(e))
                continue

    if list == 'usa-trump':
        parsed_article_dict = parsed_article_title
    else:
        parsed_article_dict = dict(zip(parsed_article_title, parsed_article_text))
    print('AGGREGATE_TWITTER: Finished parsing all articles in timeline: ')

    return parsed_article_dict, parsed_article_data