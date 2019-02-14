from monthly.monthly import sort_articles_month
import sys
import argparse

def init_task(type):
    if type == 'trumpsaid':
        collection_list = ['trumpsaid']
        for item in collection_list:
            sort_articles_month(item, 'trumpsaid', 90000009)


# 168hr is 1 week
## run task
parser = argparse.ArgumentParser()
parser.add_argument("--target", help="run interpreter for provided target")
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

sys.stdout = Transcript('./logs/action_mon_clustering_' + vars(args)['target'] + '.txt')

print(vars(args), vars(args)['target'])

if vars(args)['target'] != '':
    init_task(vars(args)['target'])
else:
    print('ACTION_MONTHLYCLUSTERING: No argument provided, finished with none.')


sys.stdout.logfile.close()
sys.stdout = sys.stdout.terminal