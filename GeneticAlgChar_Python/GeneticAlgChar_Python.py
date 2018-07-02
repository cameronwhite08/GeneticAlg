import random
import time

from deap import base
from deap import creator
from deap import tools


# the goal ('fitness') function to be maximized
def evaluate_individual(individual):
    numCorrect = 0
    for guess, goal in zip(individual, goalString):
        if guess == ord(goal) - ord('a'):
            numCorrect += 1

    return pow(2, numCorrect),


def char_codes_to_char(ints):
    strin = ''
    for i in ints:
        strin += chr(ord('a') + i)
    return strin

# global vars
goalString = 'cameronwhite'
Mutation_Probability = 0.2
population_size = 12
max_fitness = 0
generation = 0

# seed the random
random.seed(int(time.time()))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator which samples uniformly from the range [0,25]
toolbox.register("rand_letter", random.randint, 0, 25)
# define 'individual' to have len(goalString) 'rand_letter' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_letter, len(goalString))
# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register the fitness function
toolbox.register("evaluate", evaluate_individual)
# register the crossover function
toolbox.register("crossover", tools.cxOnePoint)
# register a mutation operator with a probability to flip each gene of 0.05
toolbox.register("mutate", tools.mutUniformInt, low=0, up=25, indpb=0.1)
# set the selection method to grab the top performers
toolbox.register("select", tools.selBest)


# create an initial population of individuals
pop = toolbox.population(n=population_size)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Evolution loop
while max_fitness < pow(2, len(goalString)):

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
    fitnesses = map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    best_individual = tools.selBest(pop, 1)[0]
    max_fitness = best_individual.fitness.values[0]
    print("Best of generation {0}: {1}".format(generation, char_codes_to_char(best_individual)))
    generation = generation + 1

print("-- End of evolution --")


