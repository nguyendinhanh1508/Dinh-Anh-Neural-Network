import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from storage import param_storage
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Sigmoid, Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime, cross_entropy, cross_entropy_prime

def preprocess_data(x, y, limit):
    indices = []
    for i in range(10):
        idx = np.where(y == i)[0][:limit]
        indices.extend(idx)
    indices = np.random.permutation(indices)
    x, y = x[indices], y[indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes=10)
    y = y.reshape(len(y), 10, 1)
    return x, y

def calculate_accuracy(network, x, y):
    correct = 0
    for x_sample, y_sample in zip(x, y):
        output = x_sample
        for layer in network:
            output = layer.forward(output)
        if np.argmax(output) == np.argmax(y_sample):
            correct += 1
    return correct / len(x) * 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 50)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Tanh(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Tanh(),
    Dense(100, 10),
    Softmax()
]

if param_storage.load():
    for layer in network:
        if hasattr(layer, 'layer_id'):
            layer.load_weights()
    initial_acc = calculate_accuracy(network, x_test, y_test)
    print(f"Loaded previous state. Initial test accuracy: {initial_acc:.2f}%")
else:
    print("No previous state found.")

epochs = 1000
learning_rate = 0.01

for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        output = x
        for layer in network:
            output = layer.forward(output)
        
        error += cross_entropy(y, output)

        grad = cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
        
    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error = {error}")
    param_storage.save()

tests = 0
correct = 0

for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    tests += 1
    if np.argmax(output) == np.argmax(y):
        correct += 1
        
print(f"Training finished, current accuracy: {correct / tests * 100}%")