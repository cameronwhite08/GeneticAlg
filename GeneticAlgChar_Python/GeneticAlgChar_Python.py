import random
import time

from deap import base
from deap import creator
from deap import tools

import numpy as np

# global control vars
Mutation_Probability = 0.7
population_size = 50
current_max_fitness = 1
generation = 0

# sigmoid function
def nonlin(x):
    return 1 / (1 + np.exp(-x))

def getRandWeight(z=None):
    return float(2 * np.random.random(1) - 1)

def individualTonpArray(x):
    return np.array([arr for arr in x])

def forwardProp(individual):
    syn0 = individualTonpArray(individual)

    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    return l1

# the goal ('fitness') function to be maximized
def evaluate_individual(individual):
    l1 = forwardProp(individual)

    # how much did we miss?
    l1_error = y - l1

    return float(1-np.abs(np.sum(l1_error))),

# currently trying to evolve the AND gate
# input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# output dataset
y = np.array([[0,
               0,
               0,
               1]]).T

# seed the random
random.seed(int(time.time()))
np.random.seed(int(time.time()))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator which samples uniformly from the range [0,25]
toolbox.register("rand_weight", getRandWeight)
# define 'individual' to have len(goalString) 'rand_letter' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_weight, 2)
# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register the fitness function
toolbox.register("evaluate", evaluate_individual)
# register the crossover function
toolbox.register("crossover", tools.cxUniform, indpb=.5)
# register a mutation operator with a probability to flip each gene of 0.05
toolbox.register("mutate", getRandWeight)
# set the selection method to grab the top performers
toolbox.register("select", tools.selBest)


# create an initial population of individuals
pop = toolbox.population(n=population_size)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Evolution loop
while current_max_fitness > .01:

    # Select the top 20% of the population
    offspring = toolbox.select(pop, int(len(pop) * .2))  # take top 20% of population
    # Clone the selected individuals (*5 to copy top performers to being the entire new population)
    offspring = list(map(toolbox.clone, offspring * 5))

    # Randomly shuffle the population before crossover
    random.shuffle(offspring)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        r = toolbox.crossover(child1, child2)

    for mutant in offspring:
        # mutate an individual with probability Mutation_Probability
        if random.random() < Mutation_Probability:
            toolbox.mutate(mutant)

    # Evaluate the individuals
    population = [individualTonpArray(x) for x in offspring]
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    best_individual = tools.selBest(pop, 1)
    best_individual = best_individual[0]
    current_max_fitness = best_individual.fitness.values[0]
    if generation % 100 is 0:
        print("{0}: {1}".format(generation, forwardProp(best_individual)))
    generation = generation + 1

print("-- End of evolution --")


