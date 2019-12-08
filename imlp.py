from keras import backend as K
from keras import activations
from keras.layers import Input, Dense
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np

class iAct(Layer):
    def __init__(self, activation, **kwargs):
        self.activation = activations.get(activation)
        super(iAct, self).__init__(**kwargs)

    def call(self, x):
        center, radius = x
        tmp_c = (self.activation(center-radius) + self.activation(center+radius))/2
        tmp_r = (self.activation(center+radius) - self.activation(center-radius))/2
        return [tmp_c, tmp_r]
    
    def compute_output_shape(self, input_shape):
        return input_shape

class iLoss(Layer):
    def __init__(self, beta, **kwargs):
        self.beta = beta
        super(iLoss, self).__init__(**kwargs)
        
    def loss(self, y_true, y_pred):
        error_c = K.sum(K.square(y_true[0] - y_pred[0]))
        error_r = K.sum(K.square(y_true[1] - y_pred[1]))
        return self.beta * error_c + (1 - self.beta) * error_r

def get_model(input_dim, output_dim, num_units, activation, beta, num_hidden_layers=1):
    center_x = Input((input_dim,), name='center_input')
    radius_x = Input((input_dim,), name='radius_input')

    for i in range(num_hidden_layers):
        c = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(center_x)
        r = Dense(num_units[i], use_bias=False, kernel_initializer='he_normal')(radius_x)
        c, r = iAct(activation[i])([c, r])
    
    c = Dense(output_dim, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(c)
    r = Dense(output_dim, use_bias=False, kernel_initializer='he_normal')(r)
    c, r = iAct('relu')([c, r])
    loss_layer = iLoss(beta)

    model = Model(inputs=[center_x, radius_x], outputs=[c, r])
    model.compile(loss=loss_layer.loss, optimizer='adam', metrics=['accuracy'])
    return model