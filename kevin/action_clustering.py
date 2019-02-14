from cluster.cluster import cluster_articles
from processes.daily.cluster import cluster_articles as cluster_daily
import sys
import argparse

def init_task(type):
    if type == 'today':
        collection_list = ['usasociety']
        for item in collection_list:
            cluster_daily(item, 'today', 24)
    elif type == 'theyear':
        collection_list = ['theyear']
        for item in collection_list:
            cluster_articles(item, 'theyear', 90000001)
    elif type == 'trumpsaid':
        collection_list = ['trumpsaid']
        for item in collection_list:
            cluster_articles(item, 'trumpsaid', 90000009)


# 168hr is 1 week
## run task
parser = argparse.ArgumentParser()
parser.add_argument("--target", help="run clustering for provided target")
args = parser.parse_args()

class Transcript(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass

sys.stdout = Transcript('./logs/action_clustering_' + vars(args)['target'] + '.txt')

print(vars(args), vars(args)['target'])

if vars(args)['target'] != '':
    init_task(vars(args)['target'])
else:
    print('ACTION_CLUSTERING: No argument provided, finished with none.')


sys.stdout.logfile.close()
sys.stdout = sys.stdout.terminal