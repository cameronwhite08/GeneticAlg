import random
import time

import neural_net as net;

from deap import base
from deap import creator
from deap import tools

import numpy as np

# global control vars
population_size = 50
Mutation_Probability = 0.8
current_max_fitness = 1
generation_counter = 0
gens_before_print = 100
pop_keep_percent = .2

num_inputs = 4
num_outputs = 2
# only 1 hidden layer for now
num_hidden_nodes = 5


# this is going to change considerably
def mutate_individual(individual, indpb=.5):
    for idx in range(len(individual)):
        if random.random() < indpb:
            individual[idx] = rand_weight()
    return individual


def ind_to_np_array(x):
    return np.array([arr for arr in x])


def create_individual(individual):
    weights = []
    # 4 inputs, 2 outputs, 5 nodes in hidden layer
    layer1_weights, layer2_weights = net.get_rand_weights(num_inputs, num_outputs, num_hidden_nodes)

    # flatten weight arrays so it can be stored in the individual
    flat1 = layer1_weights.flatten()
    flat2 = layer2_weights.flatten()
    for x in flat1:
        weights.append(x)
    for x in flat2:
        weights.append(x)

    return individual(x for x in weights)


# the goal ('fitness') function to be maximized
# this will be a loop for a individual's try at the game
def evaluate_individual(individual):

    arr = ind_to_np_array(individual)

    num_first_layer_weights = num_inputs * num_hidden_nodes

    layer1_w = np.reshape(arr[:num_first_layer_weights], (num_inputs, num_hidden_nodes))
    layer2_w = np.reshape(arr[num_first_layer_weights:], (num_hidden_nodes, num_outputs))

    l1 = forward_prop(individual)

    # how much did we miss?
    l1_error = y - l1
    # print('eval: ' + str(l1) + '\t' + str(abs(1-np.sum(l1_error))))
    # print('eval2: ' + str(l1) + '\t' + str(abs(np.sum(l1_error))))
    # print()
    return float(np.sum(np.power(l1_error, 2))),


def print_gen_info(print_divider=True):
    print("Generation {0} outputs: {1} -- fitness: {2}".format(generation_counter,
                                                               forward_prop(best_individual),
                                                               current_max_fitness))
    print("Evolved weights {0}:{1}".format(best_individual[0], best_individual[1]))
    if print_divider:
        print('-' * 20)


# seed the random
random.seed(int(time.time()))
np.random.seed(int(time.time()))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# define 'individual' to have len(goalString) 'rand_weight' elements ('genes')
toolbox.register("individual", create_individual, creator.Individual)
# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register the fitness function
toolbox.register("evaluate", evaluate_individual)
# register the crossover function
toolbox.register("crossover", tools.cxUniform, indpb=.5)
# register a mutation operator with a probability to flip each gene of 0.05
toolbox.register("mutate", mutate_individual)
# set the selection method to grab the top performers
toolbox.register("select", tools.selBest)


# create an initial population of individuals
pop = toolbox.population(n=population_size)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Evolution loop
while abs(current_max_fitness) > .01:

    # Select the top % of the population
    top_performers = toolbox.select(pop, int(len(pop) * pop_keep_percent))  # take % of population

    # Randomly shuffle the population before crossover
    random.shuffle(top_performers)

    next_gen = []

    # Produce enough offspring until the population is refilled
    # loop until we refill population
    while len(next_gen) < len(pop):
        # to get random top performers
        i = np.random.randint(0, len(top_performers))
        j = np.random.randint(0, len(top_performers))

        # don't want to crossover with oneself
        if i == j:
            continue

        # clone new children to crossover
        child1 = toolbox.clone(top_performers[i])
        child2 = toolbox.clone(top_performers[j])

        # crossover
        new_children = toolbox.crossover(child1, child2)

        # save into next generation
        next_gen.extend(new_children)

    for mutant in next_gen:
        # mutate an individual with probability Mutation_Probability
        if random.random() < Mutation_Probability:
            toolbox.mutate(mutant)

    # Evaluate the individuals
    population = [ind_to_np_array(x) for x in next_gen]
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(next_gen, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = next_gen

    best_individual = tools.selBest(pop, 1)
    best_individual = best_individual[0]
    current_max_fitness = best_individual.fitness.values[0]

    if generation_counter % gens_before_print is 0:
        print_gen_info()
    generation_counter = generation_counter + 1

print("-- End of evolution --")
print_gen_info(print_divider=False)
