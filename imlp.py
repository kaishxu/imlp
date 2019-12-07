from keras import backend as K
from keras import activations
from keras.engine.topology import Layer

import numpy as np

class ActLayer(Layer):
    def __init__(self, activation, **kwargs):
        self.activation = activations.get(activation)
        super(ActLayer, self).__init__(**kwargs)

    def call(self, x):
        center, radius = x
        tmp_c = (self.activation(center-radius) + self.activation(center+radius))/2
        tmp_r = (self.activation(center+radius) - self.activation(center-radius))/2
        return [tmp_c, tmp_r]
    
    def compute_output_shape(self, input_shape):
        return input_shape

def iLoss(y_true, y_pred, beta=0.5):
#     global beta
    error_c = K.sum(K.square(y_true[0] - y_pred[0]))
    error_r = K.sum(K.square(y_true[1] - y_pred[1]))
    return beta * error_c + (1 - beta) * error_r