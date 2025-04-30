import numpy as np
from layer import Layer
from storage import param_storage

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_id = param_storage.create_entry(
            (output_size, 1),
            'dense',
            {'input_size': input_size, 'output_size': output_size}
        )
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2./input_size)
        param_storage.save_weights(self.layer_id, self.weights)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + param_storage.get_bias(self.layer_id)

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        param_storage.update_bias(self.layer_id, learning_rate * output_gradient)
        return input_gradient
    
    def load_weights(self):
        self.weights = param_storage.get_weights(self.layer_id)