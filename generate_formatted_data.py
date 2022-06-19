import itertools
import random
import pickle


EXPERIMENT_NAME = "ABC_4"

with open("data/" + EXPERIMENT_NAME + "_data.pkl", 'rb') as file:
    data = pickle.load(file)

positive = [x[0] for x in data if x[1]]
negative = [x[0] for x in data if not x[1]]

def f_only_A(word):
    return tuple([x for x in word if x == "A"])

def f_only_B(word):
    return tuple([x for x in word if x == "B"])

def f_couples(word):
    return tuple([word[i]+word[i+1] for i in range(len(word) - 1)])

transformations = [f_only_B, f_couples, f_only_A]

new_data = {}
negatives = set()
for i in range(len(transformations)):
    new_positive = set()
    new_negative = set()
    for word in positive:
        new_word = transformations[i](word)
        if len(new_word) > 0 and new_word not in negatives:
            new_positive.add(new_word)
    for word in negative:
        new_word = transformations[i](word)
        if new_word not in new_positive and len(new_word) > 0 and new_word not in negatives:
            new_negative.add(new_word)
            negatives.add(new_word)
    new_data[i] = [(x, True) for x in new_positive] + [(x, False) for x in new_negative]

with open("data/" + EXPERIMENT_NAME + "_data_formatted.pkl", 'wb') as f:
    pickle.dump(new_data, f)




