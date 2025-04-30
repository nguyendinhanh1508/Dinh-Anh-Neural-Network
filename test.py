import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from storage import param_storage
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Softmax

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

def build_network():
    return [
        Convolutional((1, 28, 28), 3, 5),
        Tanh(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Tanh(),
        Dense(100, 10),
        Softmax()
    ]

def load_weights(network):
    if param_storage.load():
        for layer in network:
            if hasattr(layer, 'load_weights'):
                layer.load_weights()
        print("Loaded saved weights")
    else:
        print("No saved weights found - using random initialization")

(_, _), (x_test, y_test) = mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, limit=100)
network = build_network()
load_weights(network)
correct = 0
tests = 0

for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    tests += 1
    if np.argmax(output) == np.argmax(y):
        correct += 1
accuracy = correct / tests * 100
print(f"\nAccuracy: {accuracy:.2f}%")