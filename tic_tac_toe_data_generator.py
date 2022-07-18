from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import itertools
import csv
from iteration_utilities import random_product
import random
random.seed(1)

df = pd.read_csv("data/tic-tac-toe-endgame.csv")
MAX_SEQUENCE_LENGTH = 9
MIN_SEQUENCE_LENGTH = 9
DICTIONARY = ["X"+str(x) for x in range(9)] + ["O"+str(x) for x in range(9)]
ROWS = [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"], ["0", "3", "6"], ["1", "4", "7"], ["2", "5", "8"], ["0", "4", "8"], ["2", "4", "6"]]
WIN_OPTIONS = [["X"+str(c) for c in x] for x in ROWS] + [["O"+str(c) for c in x] for x in ROWS]

def move_after_win(word):
    for option in WIN_OPTIONS:
        for opt in itertools.permutations(option):
            if move_after_win_rec(word, opt):
                return True
    return False


def move_after_win_rec(word, option):
    if len(option) == 0:
        return len(word) > 0
    if option[0] in word:
        return move_after_win_rec(word[word.index(option[0])+1:], option[1:])
    else:
        return False

accepted_words = []
move_after_win_list = []
for index, row in df.iterrows():
    l_x = [str(y) + str(x) for x, y in zip(range(9), row.tolist()) if y == "X"]
    l_o = [str(y) + str(x) for x, y in zip(range(9), row.tolist()) if y == "O"]
    for w1 in itertools.permutations(l_x):
        for w2 in itertools.permutations(l_o):
            result = [None] * (len(w1) + len(w2))
            result[::2] = w1
            result[1::2] = w2
            if move_after_win(result):
                move_after_win_list.append(result)
            else:
                accepted_words.append(result)

print("size of accepted words:", len(accepted_words))
print("size of move_after_win_list words:", len(move_after_win_list))

with open("data/accepted_words.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(accepted_words)






def is_legal(word):
    if len(word) != len(set(word)):  # same move
        return False
    cells = [x[1:] for x in word]
    if len(cells) != len(set(cells)):  # same cell
        return False
    now_x = True
    for c in word:
        if (c[0] == "X") != now_x:
            return False
        now_x = not now_x
    return True



#rejected_words_left = 1000 # len(accepted_words)  # maybe increase
#min_word_length = min([len(w) for w in accepted_words])
#max_word_length = 9  # max([len(w) for w in accepted_words])


def generate_negative_words_by_length(DICTIONARY, length, size, accepted_words):
    print("started creating negative examples of length", length)
    rejected_words = []
    while size > 0:
        word = random_product(DICTIONARY, repeat=length)
        if (not is_legal(word)) or move_after_win(word):
            rejected_words.append(word)
            size -= 1
        if size % 10000 == 0:
            print("left generating:", size, "negative examples of length", length)
    return rejected_words

with ThreadPoolExecutor(5) as executor:
    processes = []
    processes.append(executor.submit(generate_negative_words_by_length, DICTIONARY, 5, 100000, accepted_words))
    processes.append(executor.submit(generate_negative_words_by_length, DICTIONARY, 6, 200000, accepted_words))
    processes.append(executor.submit(generate_negative_words_by_length, DICTIONARY, 7, 300000, accepted_words))
    processes.append(executor.submit(generate_negative_words_by_length, DICTIONARY, 8, 400000, accepted_words))
    processes.append(executor.submit(generate_negative_words_by_length, DICTIONARY, 9, 500000, accepted_words))
    rejected_words = [p.result() for p in processes]



with open("data/rejected_words.csv", "w") as f:
    writer = csv.writer(f)
    for r in rejected_words:
        writer.writerows(r)
