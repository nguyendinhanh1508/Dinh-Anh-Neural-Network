import numpy as np
from layer import Layer
from scipy import signal
from storage import param_storage

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_shape = (
            depth, 
            self.input_height - kernel_size + 1, 
            self.input_width - kernel_size + 1
        )
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.layer_id = param_storage.create_entry(
            (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1),
            'convolutional',
            {'input_shape': input_shape, 'kernel_size': kernel_size, 'depth': depth}
        )
        scale = np.sqrt(2. / (self.input_depth * kernel_size * kernel_size))
        self.kernels = np.random.randn(*self.kernels_shape) * scale
        param_storage.save_weights(self.layer_id, self.kernels)

    def forward(self, input):
        self.input = input
        self.output = np.copy(param_storage.get_bias(self.layer_id))
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernels_gradient
        param_storage.update_bias(self.layer_id, learning_rate * output_gradient)
        return input_gradient
    
    def load_weights(self):
        self.kernels = param_storage.get_weights(self.layer_id)
    