import numpy as np


# sigmoid function
def nonlin(x):
    return 1 / (1 + np.exp(-x))


def get_rand_weights(num_inputs, num_outputs, hidden_layer_nodes):
    # Generate weights with value between -1 to 1 so that mean is overall 0
    weights_1 = 2 * np.random.random((num_inputs, hidden_layer_nodes)) - 1
    weights_2 = 2 * np.random.random((hidden_layer_nodes, num_outputs)) - 1
    return weights_1, weights_2


def forward_prop(inputs, layer1_weights, layer2_weights):
    layer1 = nonlin(np.dot(inputs, layer1_weights))
    layer2 = nonlin(np.dot(layer1, layer2_weights))
    return layer2


def flatten(layer1_weights, layer2_weights):
    weights = []
    # flatten weight arrays so it can be stored in the individual
    flat1 = layer1_weights.flatten()
    flat2 = layer2_weights.flatten()
    for x in flat1:
        weights.append(x)
    for x in flat2:
        weights.append(x)
    return weights


def un_flatten(num_inputs, num_hidden, num_outputs, weights):
    # find split point of flat_weights
    num_first_layer_weights = num_inputs * num_hidden

    layer1_w = np.reshape(weights[:num_first_layer_weights], (num_inputs, num_hidden))
    layer2_w = np.reshape(weights[num_first_layer_weights:], (num_hidden, num_outputs))

    return layer1_w, layer2_w
