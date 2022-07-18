import pickle
EXPERIMENT_NAME = "ABC_4"
from aalpy.learning_algs import run_RPNI
import time

with open("data/" + EXPERIMENT_NAME + "_data_formatted.pkl", 'rb') as file:
    data = pickle.load(file)

models = []
start = time.time()
for i in range(len(data)):
    models.append(run_RPNI(data[i], automaton_type='dfa'))
print("elapsed time:", time.time() - start)
models[-1].make_input_complete()
def f_only_A(word):
    return tuple([x for x in word if x == "A"])

def f_only_B(word):
    return tuple([x for x in word if x == "B"])

def f_couples(word):
    return tuple([word[i]+word[i+1] for i in range(len(word) - 1)])

transformations = [f_only_B, f_couples, f_only_A]


with open("data/" + "ABC_10" + "_train.pkl", 'rb') as file:
    data = pickle.load(file)
acc_all = 0
acc_any = 0
for seq, l in data:
    predicted = []
    for i in range(len(models)):
        new_word = transformations[i](seq)
        if len(new_word) > 0:
            predicted.append(models[i].execute_sequence(models[i].initial_state, new_word)[-1])
        else:
            predicted.append(models[i].initial_state.is_accepting)
    if all(predicted) == l:
        acc_all += 1
    if any(predicted) == l:
        acc_any += 1
print("accuracy train all:", acc_all/len(data))
print("accuracy train any:", acc_any/len(data))

with open("data/" + "ABC_10" + "_test.pkl", 'rb') as file:
    data = pickle.load(file)
acc_all = 0
acc_any = 0
for seq, l in data:
    predicted = []
    for i in range(len(models)):
        new_word = transformations[i](seq)
        if len(new_word) > 0:
            predicted.append(models[i].execute_sequence(models[i].initial_state, new_word)[-1])
        else:
            predicted.append(models[i].initial_state.is_accepting)
    if all(predicted) == l:
        acc_all += 1
    if any(predicted) == l:
        acc_any += 1
print("accuracy test all:", acc_all/len(data))
print("accuracy test any:", acc_any/len(data))


for i in range(len(models)):
    models[i].visualize(path="model_" + str(i))

