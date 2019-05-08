# kevin
Full back-end set for [thisisallabout](https://thisisallabout.com), a non-profit data journalism project.
  
## Modules

#### It consists of two modules: news aggregation and clustering module.

1. action_aggregation — It automatically aggregates news story from major news publishers every hour using Twitter API. Once it fetches news data available on the web page linked with each tweet, it filters out stop words to remove unnecessary information. It saves the final data to database and that's all for this part.
2. action_clustering — For clustering, this module has to be initiated either by manually or through cron schedule. It passes an appropriate clustering parameter depending on target source you specified on command line.

To initiate each module on shell:

    path/action_aggregation.py --target=PARAMETER  

    path/action_clustering.py --target=PARAMETER 

    path/action_monthly_clustering.py --target=PARAMETER  ## monthly clustering module

    path/action_monthly_clustering_postproc.py --target=PARAMETER  ## monthly clustering post processing module

You can edit the param and functions triggered by it on each module .py file.

---

### Folder structure
- It creates a log file on ./logs/. Make sure you have the folder set up along with modules as well.
- It creates some files including tfidf data during the processing in ./dataset/.

### clustering: vectorize_cluster configurations

vectorize_cluster is the core and main function of tfidf_kmeans.py module which passes kmeans result back to main clustering module.

- rangeMin, rangeMax: K-Means clustering iteration for elbow detection. The module iterates multiple K-Means calls to automatically figure out an ideal cluster size.
 - cachedStopWords: the stopwords list. You may edit this or make it load the list from a separate file.
 - n_features: the features of your data for optimal K-Means result
 - X = vectorizer.fit_transform(...): provide appropriate data to the parameter for proper transformation
 - n_components: the components of your data for optimal K-Means result

### clustering: "titleonly" condition
The param/option "titleonly" is used for single-line data (i.e. Trump's tweets). Otherwise, you should call functions without the option and provide full title-text structure data.

### clustering: cluster_postproc module
./processes/daily/cluster.py module (daily news clustering) uses a post-processing module after K-Means. It picks top articles through whoosh search (cluster tfidf themes are used for query) and generates new LDA topic keywords describing the cluster better.

### clustering: monthly clustering module
./processes/monthly/ directory contains a dedicated clustering module for monthly-type data. It was originally designed for President Trump's tweet analysis, grouping tweet data into monthly chunk for better clustering of topics shifting throughout the year.

To start clustering, first run path/action_monthly_clustering.py --target=PARAMETER. Once initial clustering work is done, it will save the result file at ./processes/monthly/cluster/TARGETPARAMETER/. 

To process the result file for better readability or front-end rendering, you could run a separate cluster_postproc for monthly clustering result. Run path/action_monthly_clustering_postproc.py --target=PARAMETER. The final data will be located at ./processes/monthly/postproc/.

* It is recommended to change "trumpsaid" target parameter on this module and ./processes/monthly/cluster.py, ./processes/monthly/cluster_bimonthly.py to your own one.
* On ./processes/monthly/cluster.py and ./processes/monthly/cluster_bimonthly.py, it calls ./processes/monthly/tfidf_kmeans.py with a different target parameter: "titleonly". This doesn't necessarily has to do anything with monthly clustering. Please check the explanation of this condition parameter above.

### module database address
By default, it's configured to 

    mongodb://test:test@0.0.0.0:9999/main

You can replace all at once in all files with database connection by searching the default address.

### Legacy files

./cluster/. All codes in this directory has been moved to ./processes/.

## Data source

It originally aggregated and processed stories from CNN, Fox News, The New York Times, The Hill, Washington Post, The Wall Street Journal, NPR, Chicago Tribune, USA Today, Politico, L.A. Times, NBC News, PBS NewsHour, The Washington Times, The New Yorker, CBS News, C-SPAN, ABC News, The Atlantic, AP, The New Republic, The Boston Globe, Business Insider, CNBC, Bloomberg, and Financial Times using Twitter API. 

You may set up your own twitter list and connect it to aggregator module via Twitter API.

  
## Libraries

* NLTK, gensim (LDAModel/doc2bow) — LDA Topics implemented on Feb 08, 2019.

* scikit-learn (HashingVectorizer/TfidfVectorizer/MiniBatchKMeans/TruncatedSVD)
* KneeLocator to detect elbow

* pandas, scipy, numpy

* whoosh for searching top matching articles

* MongoDB/pymongo

  

## License

* MIT License

## Maintainer

Todd Oh (fieldofgreentea@gmail.com) [twitter](https://twitter.com/_toddoh)