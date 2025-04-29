import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 1000
learning_rate = 0.1

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        
        error += mse(y, output)

        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print('%d/%d, error*%f' % (e + 1, epochs, error))