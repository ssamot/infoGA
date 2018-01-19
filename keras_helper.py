import keras.backend as K
import numpy as np


def set_trainable_weight(model, weights):
    """Sets the weights of the model.

    # Arguments
        model: a keras neural network model
        weights: A list of Numpy arrays with shapes and types matching
            the output of `model.trainable_weights`.
    """
    tuples = []
    for layer in model.layers:
        num_param = len(layer.trainable_weights)
        layer_weights = weights[:num_param]
        for sw, w in zip(layer.trainable_weights, layer_weights):
            tuples.append((sw, w))
        weights = weights[num_param:]
    K.batch_set_value(tuples)

def get_trainable_weights(model):
    """Gets the weights of the model.

    # Arguments
        model: a keras neural network model

    # Returns
        a Numpy array of weights
    """
    W_list = K.get_session().run(model.trainable_weights)
    W_flattened_list = [k.flatten() for k in W_list]
    W = np.concatenate(W_flattened_list)
    return W