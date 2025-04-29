import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        exp = np.exp(input - np.max(input))
        self.output = exp / np.sum(exp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.empty_like(output_gradient)
        for i, (single_output, single_gradient) in enumerate(zip(self.output, output_gradient)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            input_gradient[i] = np.dot(jacobian, single_gradient)
        return input_gradient