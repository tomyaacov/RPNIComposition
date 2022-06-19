import pandas as pd
from aalpy.learning_algs import run_RPNI
import time
from sklearn.model_selection import train_test_split
import random
random.seed(1)

data = []
with open("data/" + "accepted_words.csv", 'r') as f:
    for line in f:
        data.append((tuple(line.strip().split(",")), True))
with open("data/" + "rejected_words.csv", 'r') as f:
    for line in f:
        data.append((tuple(line.strip().split(",")), False))
random.shuffle(data)

train = data[:10000]
test = data[10000:20000]


start = time.time()
model = run_RPNI(train, automaton_type='dfa')
print("elapsed time:", time.time() - start)
acc = 0
for seq, l in train:
    if model.execute_sequence(model.initial_state, seq)[-1] == l:
        acc += 1
print("accuracy train:", acc/len(train))
acc = 0
for seq, l in test:
    if model.execute_sequence(model.initial_state, seq)[-1] == l:
        acc += 1
print("accuracy test:", acc/len(test))
model.visualize()