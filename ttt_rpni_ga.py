import random
import numpy as np
from deap import algorithms, base, creator, tools
from aalpy.learning_algs import run_RPNI
import multiprocessing

random.seed(10)
INPUT_DIM = 18
INPUT_TOKENS = ["X"+str(i) for i in range(9)] + ["O"+str(i) for i in range(9)]
OUTPUT_DIM = 3  # maybe it should be more
OUTPUT_TOKENS = [str(x) for x in range(OUTPUT_DIM+1)]
POP_SIZE_1 = 300
POP_SIZE_2 = 300
N_GEN_1 = 500
N_GEN_2 = 500
FIRST_POP = None


# loading data
DATA = []
with open("data/" + "accepted_words.csv", 'r') as f:
    for line in f:
        DATA.append((tuple(line.strip().split(",")), True))
with open("data/" + "rejected_words.csv", 'r') as f:
    for line in f:
        DATA.append((tuple(line.strip().split(",")), False))
random.shuffle(DATA)

# train test split
TRAIN = DATA[:int(len(DATA) * 0.7)]
TEST = DATA[int(len(DATA) * 0.3):]

TRAIN = DATA[:1000]
TEST = DATA[:1000]


def _transform(x, code):
    result = [OUTPUT_TOKENS[code[INPUT_TOKENS.index(i)]] for i in x]
    return tuple([x for x in result if x != "0"])


def transform(data, code):
    return [(_transform(x, code), l) for x, l in data]


def sample_individual(ind, size, dim):
    return ind([random.randint(0, dim) for _ in range(size)] + [random.randint(0, 2)])

def learn_model(individual):
    new_l = [False, True, None][list(individual)[-1]]
    transformed_data = transform(TRAIN, list(individual)[:-1])
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
    if len(new_positive) == 0 and new_l != True:
        return  # invalid transformation - all samples negative
    if len(new_negative) == 0 and new_l != False:
        return  # invalid transformation - all samples positive
    new_data = [(x, True) for x in new_positive] + [(x, False) for x in new_negative] \
               + [(x, new_l) for x in conflict if new_l is not None]
    random.shuffle(new_data)
    model = run_RPNI(new_data, automaton_type='dfa', print_info=True)
    return model


def eval_ind(individual):
    model = learn_model(individual)
    if model is None:
        return 0,
    else:
        return 2*(1/model.size),


def run_first_ga():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", sample_individual, creator.Individual, INPUT_DIM, OUTPUT_DIM)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=OUTPUT_DIM, indpb=1 / INPUT_DIM)
    toolbox.register("select", tools.selTournament, tournsize=3)  # maybe different selection?

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=POP_SIZE_1)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.1, ngen=N_GEN_1,
                                   stats=stats, halloffame=hof, verbose=False)
    print(log)
    print(pop)
    return pop


def eval_comb(individual):
    ts = [FIRST_POP[i] for i, x in enumerate(list(individual)) if x == 1]
    models = []
    for t in ts:
        model = learn_model(t)
        model.make_input_complete('self_loop')
        if model is None:
            return 0, 1
        else:
            models.append(model)
    acc_all = 0
    for seq, l in TEST:
        predicted = []
        for i in range(len(models)):
            new_word = _transform(seq, ts[i][:-1])
            if len(new_word) > 0:
                predicted.append(models[i].execute_sequence(models[i].initial_state, new_word)[-1])
            else:
                predicted.append(models[i].initial_state.is_accepting)
        if all(predicted) == l:
            acc_all += 1
    return acc_all/len(TEST), sum(individual)/len(individual)


def run_second_ga():
    creator.create("FitnessMulti", base.Fitness, weights=(100.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(FIRST_POP))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_comb)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=POP_SIZE_2)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.1, ngen=N_GEN_2,
                                   stats=stats, halloffame=hof, verbose=False)
    print(log)
    print(pop)
    return pop, hof


if __name__ == "__main__":
    FIRST_POP = run_first_ga()
    second_pop, hof = run_second_ga()
    print(hof)