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
