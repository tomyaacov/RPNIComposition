import itertools
import random
import pickle

DICTIONARY = ["A", "B", "C"]
NUMBER_OF_STATES = 10
EXPERIMENT_NAME = "ABC_10"
SEQUENCE_LENGTH = 10

random.seed(10)

def l(w):
    return w.count("A")%2==0 and w.count("B")%2==0 and "BB" in w
words = []
for i in range(1, SEQUENCE_LENGTH+1):
    words.extend(list(itertools.product(DICTIONARY, repeat=i)))
random.shuffle(words)
data_to_file = ""
data_to_pkl = []

for current_word in words:
    label = l("".join(current_word))
    data_to_pkl.append((current_word, label))

# with open("data/" + EXPERIMENT_NAME + "_data.pkl", 'wb') as f:
#     pickle.dump(data_to_pkl, f)


split_point = int(len(data_to_pkl)*0.7)
train = data_to_pkl[:split_point]
test = data_to_pkl[split_point:]


with open("data/" + EXPERIMENT_NAME + "_train.pkl", 'wb') as f:
    pickle.dump(train, f)

with open("data/" + EXPERIMENT_NAME + "_test.pkl", 'wb') as f:
    pickle.dump(test, f)
