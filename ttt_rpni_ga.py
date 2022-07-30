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
train = data[:int(len(data) * 0.7)]
test = data[int(len(data) * 0.3):]


def _transform(x, code):
    result = [OUTPUT_TOKENS[code[INPUT_TOKENS.index(i)]] for i in x]
    return tuple([x for x in result if x != "0"])


def transform(data, code):
    return [(_transform(x, code), l) for x, l in data]


def sample_individual(ind, size, dim):
    return ind([random.randint(0, dim) for _ in range(size)] + [random.randint(0, 2)])


def eval_ind(individual):
    new_l = [False, True, None][list(individual)[-1]]
    transformed_data = transform(train, list(individual)[:-1])
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
        return (0, ) # invalid transformation - all samples negative
    if len(new_negative) == 0 and new_l != False:
        return (0, ) # invalid transformation - all samples positive
    new_data = [(x, True) for x in new_positive] + [(x, False) for x in new_negative] \
               + [(x, new_l) for x in conflict if new_l is not None]
    random.shuffle(new_data)
    model = run_RPNI(new_data, automaton_type='dfa')
    return (2*(1/model.size),)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# toolbox.register("attr_int", sample_individual, INPUT_DIM, OUTPUT_DIM)
toolbox.register("individual", sample_individual, creator.Individual, INPUT_DIM, OUTPUT_DIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_ind)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=OUTPUT_DIM, indpb=1 / INPUT_DIM)
# toolbox.register("select", tools.selBest, k=100)
toolbox.register("select", tools.selTournament, tournsize=3)


pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.1, ngen=500,
                                   stats=stats, halloffame=hof, verbose=False)
    print(log)
    print(pop)


if __name__ == "__main__":
    main()