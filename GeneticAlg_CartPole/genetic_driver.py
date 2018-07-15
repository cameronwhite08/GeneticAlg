import random
import time

import neural_net as net

from deap import base, creator, tools, algorithms
import multiprocessing

import numpy as np

import gym

# global control vars
population_size = 50
num_generations = 5
mutation_probability = 0.2
mate_probability = .5
current_env = 'CartPole-v0'

net_inputs = 4
net_outputs = 2
# only 1 hidden layer for now
num_hidden_nodes = 5


def mutate_individual(individual, indpb=.1):
    for idx in range(len(individual)):
        # randomly mutate individual genes
        if np.random.random(1) < indpb:
            # randomly decide the mutation
            decision = float(np.random.random(1))

            rand = float(np.random.random(1))

            # whole new weight
            if decision < .25:
                individual[idx] = 2 * rand - 1
                continue
            # change weight by random percentage
            if decision < .5:
                multiplier = rand / 1.5  # get the multiplier on the range of [.5,1.5)
                individual[idx] = individual[idx] * multiplier
                continue
            if decision < .75:
                sign = 1
                # to make this randomly addition or subtraction
                if np.random.random(1) < .5:
                    sign = -sign
                individual[idx] += (rand * sign)
                continue
            # do something as if 1>rand>.75
            individual[idx] = -individual[idx]

    return individual,


def ind_to_np_array(x):
    return np.array([arr for arr in x])


def create_individual(individual):
    weights = []
    # 4 inputs, 2 outputs, 5 nodes in hidden layer
    layer1_weights, layer2_weights = net.get_rand_weights(net_inputs, net_outputs, num_hidden_nodes)

    # flatten weight arrays so it can be stored in the individual
    flat1 = layer1_weights.flatten()
    flat2 = layer2_weights.flatten()
    for x in flat1:
        weights.append(x)
    for x in flat2:
        weights.append(x)

    return individual(x for x in weights)


# this will be a loop for a individual's try at the game
def evaluate_individual(individual, display=False):
    flat_weights = ind_to_np_array(individual)
    # find split point of flat_weights
    num_first_layer_weights = net_inputs * num_hidden_nodes

    layer1_w = np.reshape(flat_weights[:num_first_layer_weights], (net_inputs, num_hidden_nodes))
    layer2_w = np.reshape(flat_weights[num_first_layer_weights:], (num_hidden_nodes, net_outputs))

    # do this to avoid 200 step limit
    env = gym.make(current_env).env
    observation = env.reset()
    fitness = 0

    for t in range(1500):
        if display:
            env.render()
        # print(observation)

        # get observation into np array
        obs = ind_to_np_array(observation)

        # forward propagate based on the observation and store result as a list
        result = list(net.forward_prop(obs, layer1_w, layer2_w))

        # action agent will take is based on output with max result
        max_index = result.index(max(result))

        # take a step
        observation, reward, done, info = env.step(max_index)

        # add reward to agent's fitness
        fitness += reward

        if done:
            if display:
                print("Episode finished after {} time steps".format(t + 1))
            break
    return fitness,


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
toolbox.register("mate", tools.cxTwoPoint)
# register a mutation operator with a probability to flip each gene of 0.05
toolbox.register("mutate", mutate_individual)
# set the selection method to grab the top performers
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == '__main__':
    # seed the random
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    # Process Pool of 8 workers
    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)

    try:
        # create an initial population of individuals
        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        current_max_fitness = 0

        # Evolution loop
        while current_max_fitness < 1000:
            pop, book = algorithms.eaSimple(pop, toolbox,
                                            cxpb=mate_probability,  # probability of mating 2 individuals
                                            mutpb=mutation_probability,  # probability of mutating an individual
                                            ngen=num_generations,  # number of generations
                                            stats=stats,  # statistics object
                                            halloffame=hof)  # contains the best individuals

            bestInd = hof

            current_fitness = float(bestInd[0].fitness.values[0])
            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                print("Current max fitness: {0}".format(current_max_fitness))
                evaluate_individual(bestInd[0], True)

            print("Current max fitness: {0}".format(current_max_fitness))
    # always close the thread pool
    finally:
        pool.close()

    print("-- End of evolution --")
