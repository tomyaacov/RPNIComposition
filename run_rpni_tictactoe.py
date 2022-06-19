import pandas as pd
from aalpy.learning_algs import run_RPNI
from aalpy.utils import save_automaton_to_file
import time
import sys
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

if sys.argv[1] == "test":
    train = data[:200]
    test = data[100:200]
else:
    train = data[:int(len(data) * 0.5)]
    test = data[int(len(data) * 0.5):]


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
save_automaton_to_file(model, path="tic_tac_toe_model")