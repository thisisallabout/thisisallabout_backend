from interpret.unit_interpreter import interpret
from interpret.unit_interpreter_today import interpret_today
import sys
import argparse

## run task
parser = argparse.ArgumentParser()
parser.add_argument("--target", help="run interpreter for provided target")
args = parser.parse_args()

orig_stdout = sys.stdout
f = open('./logs/action_interpreter_' + vars(args)['target'] + '.txt', 'w')
sys.stdout = f

print(vars(args), vars(args)['target'])

def interpreter_run(type):
    if type == '':
        print('Interpreter: No type provided, finished with none.')
    elif type == 'today':
        interpret_today(type)
    else:
        interpret(type)



if vars(args)['target'] != '':
    interpret(vars(args)['target'])
else:
    print('Interpreter: No argument provided, finished with none.')

sys.stdout = orig_stdout
f.close()