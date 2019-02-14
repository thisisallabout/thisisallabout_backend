from aggregate.aggregator import aggregate_db
import sys
import argparse

def init_task(type):
    if type == 'today':
        collection_list = ['usasociety']
        for item in collection_list:
            aggregate_db(item)
    elif type == 'theyear':
        collection_list = ['theyear']
        for item in collection_list:
            aggregate_db(item)
    elif type == 'trumpsaid':
        collection_list = ['trumpsaid']
        for item in collection_list:
            aggregate_db(item)


## run task
parser = argparse.ArgumentParser()
parser.add_argument("--target", help="run aggregator for provided target")
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

sys.stdout = Transcript('./logs/action_aggregation_' + vars(args)['target'] + '.txt')

print(vars(args), vars(args)['target'])

if vars(args)['target'] != '':
    init_task(vars(args)['target'])
else:
    print('ACTION_AGGGREGATION: No argument provided, finished with none.')


sys.stdout.logfile.close()
sys.stdout = sys.stdout.terminal