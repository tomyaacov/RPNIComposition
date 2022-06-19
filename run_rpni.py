import pickle
from aalpy.learning_algs import run_RPNI
import time

EXPERIMENT_NAME = "ABC_4"
with open("data/" + EXPERIMENT_NAME + "_data.pkl", 'rb') as file:
    data = pickle.load(file)
start = time.time()
model = run_RPNI(data, automaton_type='dfa')
print("elapsed time:", time.time() - start)
acc = 0
for seq, l in data:
    if model.execute_sequence(model.initial_state, seq)[-1] == l:
        acc += 1
print("accuracy train:", acc/len(data))
with open("data/" + "ABC_10" + "_test.pkl", 'rb') as file:
    data = pickle.load(file)
acc = 0
for seq, l in data:
    if model.execute_sequence(model.initial_state, seq)[-1] == l:
        acc += 1
print("accuracy test:", acc/len(data))
model.visualize()

