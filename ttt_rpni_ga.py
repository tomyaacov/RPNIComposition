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
N_GEN_1 = 400
N_GEN_2 = 400
FIRST_POP = None
FIRST_POP_MODELS = None
FIRST_POP_RESULTS = None


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
TRAIN_FIRST = DATA[:int(len(DATA) * 0.1)]
TRAIN = DATA[:int(len(DATA) * 0.7)]
TEST = DATA[int(len(DATA) * 0.7):]
UNTRAINABLE = set()


def _transform(x, code):
    result = [OUTPUT_TOKENS[code[INPUT_TOKENS.index(i)]] for i in x]
    return tuple([x for x in result if x != "0"])


def transform(data, code):
    return [(_transform(x, code), l) for x, l in data]


def sample_individual(ind, size, dim):
    return ind([random.randint(0, dim) for _ in range(size)])
    #return ind([random.randint(0, dim) for _ in range(size)] + [random.randint(0, 2)])


def get_transformed_data(individual, data):
    # try:
    #     new_l = [False, True, None][list(individual)[-1]]
    # except IndexError:
    #     print(list(individual))
    if tuple(individual) in UNTRAINABLE:
        return
    new_l = True
    transformed_data = transform(data, list(individual))
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
        UNTRAINABLE.add(tuple(individual))
        return
    if len(new_negative) == 0 and new_l != False:
        UNTRAINABLE.add(tuple(individual))
        return
    new_data = [(x, True) for x in new_positive] + [(x, False) for x in new_negative] \
               + [(x, new_l) for x in conflict if new_l is not None]
    return new_data

def learn_model(individual):
    if tuple(individual) in UNTRAINABLE:
        return
    new_data = get_transformed_data(individual, TRAIN)
    if new_data is None:
        UNTRAINABLE.add(tuple(individual))
        return
    else:
        random.shuffle(new_data)
        model = run_RPNI(new_data, automaton_type='dfa', print_info=True)
        return model


def eval_ind(individual):
    new_data = get_transformed_data(individual, TRAIN_FIRST)
    if new_data is None:
        return 0,
    else:
        return 2*(1/len(new_data)),


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
    models = [FIRST_POP_MODELS[i] for i, x in enumerate(list(individual)) if x == 1]
    results = [FIRST_POP_RESULTS[i] for i, x in enumerate(list(individual)) if x == 1]
    acc_all = 0
    for id in range(len(TEST)):
        predicted = []
        for i in range(len(models)):
            if results[i] is None:
                return 0, 1
            predicted.append(results[i][id])
        if all(predicted) == TEST[id][1]:
            acc_all += 1
    return acc_all/len(TEST), sum(individual)/len(individual)


def transformation_to_model(t, i):
    global FIRST_POP_MODELS, FIRST_POP_RESULTS
    model = learn_model(t)
    if model is None:
        FIRST_POP_MODELS[i] = model
        FIRST_POP_RESULTS[i] = None
    else:
        model.make_input_complete('self_loop')
        FIRST_POP_MODELS[i] = model
        predicted = []
        for seq, l in TEST:
            new_word = _transform(seq, t)
            if len(new_word) > 0:
                predicted.append(model.execute_sequence(model.initial_state, new_word)[-1])
            else:
                predicted.append(model.initial_state.is_accepting)
        FIRST_POP_RESULTS[i] = predicted


def run_second_ga():

    with multiprocessing.Pool() as p:
        p.starmap(transformation_to_model, [(t, i) for i, t in enumerate(FIRST_POP)])

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
    FIRST_POP_MODELS = [None] * len(FIRST_POP)
    FIRST_POP_RESULTS = [None] * len(FIRST_POP)
    second_pop, hof = run_second_ga()
    print(hof)