import pandas as pd
from aalpy.learning_algs import run_RPNI
from aalpy.utils import save_automaton_to_file
import time
import sys
import random

random.seed(1)


# defining transformers
def only_player(word):
    return tuple([x[0] for x in word])


def only_cell(i):
    def f(word):
        return tuple([x[1:] for x in word if x[1:] == str(i)])

    return f


transformers = [(only_player, True)] + [(only_cell(i), True) for i in range(9)]


def transform(data, f):
    return [(f(x), l) for x, l in data]


# loading data
data = []
with open("data/" + "accepted_words.csv", 'r') as f:
    for line in f:
        data.append((tuple(line.strip().split(",")), True))
with open("data/" + "rejected_words.csv", 'r') as f:
    for line in f:
        data.append((tuple(line.strip().split(",")), False))
random.shuffle(data)

# train test split
train = data[:int(len(data) * 0.5)]
test = data[int(len(data) * 0.5):]

models = []
start = time.time()
for i in range(len(transformers)):
    transformed_data = transform(train, transformers[i][0])
    positive = set([x[0] for x in transformed_data if x[1]])
    negative = set([x[0] for x in transformed_data if not x[1]])
    new_positive = set()
    new_negative = set()
    conflict = set()
    for word in positive:
        if word not in negative:
            new_positive.add(word)
        else:
            conflict.add(word)
    for word in negative:
        if word not in positive:
            new_negative.add(word)
        else:
            conflict.add(word)
    new_data = [(x, True) for x in new_positive] + [(x, False) for x in new_negative] \
               + [(x, transformers[i][1]) for x in conflict if transformers[i][1] is not None]
    models.append(run_RPNI(new_data, automaton_type='dfa'))
print("elapsed time:", time.time() - start)

acc_all = 0
acc_any = 0
for seq, l in test[:1000]:
    predicted = []
    for i in range(len(models)):
        new_word = transformers[i][0](seq)
        if len(new_word) > 0:
            predicted.append(models[i].execute_sequence(models[i].initial_state, new_word)[-1])
        else:
            predicted.append(models[i].initial_state.is_accepting)
    if all(predicted) == l:
        acc_all += 1
        print(seq)
    if any(predicted) == l:
        acc_any += 1
print("accuracy test all:", acc_all/len(test))
print("accuracy test any:", acc_any/len(test))

for i in range(len(models)):
    models[i].visualize(path="model_" + str(i))
