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
    # Get equal samples for all digits 0-9
    indices = []
    for i in range(10):
        idx = np.where(y == i)[0][:limit]
        indices.extend(idx)
    indices = np.random.permutation(indices)
    x, y = x[indices], y[indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes=10)  # Now 10 classes
    y = y.reshape(len(y), 10, 1)  # Output shape is now (10, 1)
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

# Load more data since we're classifying more digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)  # 100 samples per digit
x_test, y_test = preprocess_data(x_test, y_test, 50)  # 50 test samples per digit

network = [
    Convolutional((1, 28, 28), 3, 5),  # Input: 1 channel, 28x28, 3x3 kernel, 5 filters
    Tanh(),  # Using Tanh for better gradient flow
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Tanh(),
    Dense(100, 10),  # Output layer now has 10 neurons
    Softmax()  # Use Softmax for multi-class classification
]

# if param_storage.load(network):
#     initial_acc = calculate_accuracy(network, x_test, y_test)
#     print(f"Loaded previous biases. Initial test accuracy: {initial_acc:.2f}%")
# else:
#     print("No previous biases found. Starting fresh.")

epochs = 20
learning_rate = 0.01

for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        output = x
        for layer in network:
            output = layer.forward(output)
        
        error += cross_entropy(y, output)  # Using cross_entropy instead of binary_cross_entropy

        grad = cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
        
    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error = {error}")

tests = 0
correct = 0

for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    tests += 1
    if np.argmax(output) == np.argmax(y):
        correct += 1
        
# param_storage.save(network)
print(f"accuracy {correct / tests * 100}%")
